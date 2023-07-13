import numpy as np
from pycuda.gpuarray import GPUArray
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time
import os
from pycuda.driver import PointerHolderBase
import torch
import ctypes
from ctypes import RTLD_GLOBAL
from run_nerf_helpers import get_embedder

# gpu_n = '6'
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu_n  # args.gpu_no

N_SAMPLES = 8
N_POINT_RAY_ENC = 48
NUM_NEIGHBOR = 4

class Holder(PointerHolderBase):

    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor
        self.gpudata = tensor.data_ptr()

    def get_pointer(self):
        return self.tensor.data_ptr()

    # without an __index__ method, arithmetic calls to the GPUArray backed by this pointer fail
    # not sure why, this needs to return some integer, apparently
    def __index__(self):
        return self.gpudata

    def __int__(self):
        return self.gpudata

# dict to map between torch and numpy dtypes
dtype_map = {
    # signed integers
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.short: np.int16,
    torch.int32: np.int32,
    torch.int: np.int32,
    torch.int64: np.int64,
    torch.long: np.int64,

    # unsinged inters
    torch.uint8: np.uint8,

    # floating point
    torch.float: np.float32,
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.half: np.float16,
    torch.float64: np.float64,
    torch.double: np.float64
}


def torch_dtype_to_numpy(dtype):
    '''Convert a torch ``dtype`` to an equivalent numpy ``dtype``, if it is also available in pycuda.
    Parameters
    ----------
    dtype   :   np.dtype
    Returns
    -------
    torch.dtype
    Raises
    ------
    ValueError
        If there is not PyTorch equivalent, or the equivalent would not work with pycuda
    '''

    from pycuda.compyte.dtypes import dtype_to_ctype
    if dtype not in dtype_map:
        raise ValueError(f'{dtype} has no PyTorch equivalent')
    else:
        candidate = dtype_map[dtype]
        # we can raise exception early by checking of the type can be used with pycuda. Otherwise
        # we realize it only later when using the array
        try:
            _ = dtype_to_ctype(candidate)
        except ValueError:
            raise ValueError(f'{dtype} cannot be used in pycuda')
        else:
            return candidate


def numpy_dtype_to_torch(dtype):
    '''Convert numpy ``dtype`` to torch ``dtype``. The first matching one will be returned, if there
    are synonyms.
    Parameters
    ----------
    dtype   :   torch.dtype
    Returns
    -------
    np.dtype
    '''
    for dtype_t, dtype_n in dtype_map.items():
        if dtype_n == dtype_t:
            return dtype_t


def tensor_to_gpuarray(tensor):
    '''Convert a :class:`torch.Tensor` to a :class:`pycuda.gpuarray.GPUArray`. The underlying
    storage will be shared, so that modifications to the array will reflect in the tensor object.
    Parameters
    ----------
    tensor  :   torch.Tensor
    Returns
    -------
    pycuda.gpuarray.GPUArray
    Raises
    ------
    ValueError
        If the ``tensor`` does not live on the gpu
    '''
    if not tensor.is_cuda:
        raise ValueError('Cannot convert CPU tensor to GPUArray (call `cuda()` on it)')
    else:
        array = GPUArray(tensor.shape, dtype=torch_dtype_to_numpy(tensor.dtype),
                         gpudata=Holder(tensor))
        return array


def gpuarray_to_tensor(gpuarray):
    '''Convert a :class:`pycuda.gpuarray.GPUArray` to a :class:`torch.Tensor`. The underlying
    storage will NOT be shared, since a new copy must be allocated.
    Parameters
    ----------
    gpuarray  :   pycuda.gpuarray.GPUArray
    Returns
    -------
    torch.Tensor
    '''
    shape = gpuarray.shape
    dtype = gpuarray.dtype
    out_dtype = numpy_dtype_to_torch(dtype)
    out = torch.zeros(shape, dtype=out_dtype).cuda()
    gpuarray_copy = tensor_to_gpuarray(out)
    byte_size = gpuarray.itemsize * gpuarray.size
    pycuda.driver.memcpy_dtod(gpuarray_copy.gpudata, gpuarray.gpudata, byte_size)
    return out

class MMEngine(object):
    def _load_engine(self, model_path):
        with open(model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        print("Allocating buffer for engine I/O")
        outputs = []
        bindings = []
        # print("Batch size: ", self.engine.max_batch_size)
        out_ptr = 0
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                print("Input name: {}| Size: {}| Type: {}".format(binding, size, dtype))
                # NOTE init input binding with dummy int
                bindings.append(1000)
            else:
                print("Output name: {}| Size:{} | Type: {}".format(binding, size, dtype))
                out = self.out_ptrs[out_ptr]
                out_ptr += 1
                device_mem = GPUArray(out.shape, dtype=torch_dtype_to_numpy(out.dtype),
                         gpudata=Holder(out))
                
                outputs.append(device_mem)
                bindings.append(int(device_mem.gpudata))
        return outputs, bindings

    def __init__(self, load_model, batch=756*1008, in_ch=6*N_POINT_RAY_ENC):
        t1 = time.time()
        self.batch_size = batch
        self.in_ch = in_ch
        self.shape_of_output = [
                        (batch, 3),# mm_rgb
                        (batch, N_SAMPLES), # mm_density_add
                        (batch, N_SAMPLES), # mm_density_mul
                        (batch, N_SAMPLES), # depth_values
                        ]

        self.trt_logger = trt.Logger(trt.Logger.INFO)

        self.mm_density_mul = torch.zeros((batch * N_SAMPLES), dtype=torch.float32).cuda()
        self.mm_density_add = torch.zeros((batch * N_SAMPLES), dtype=torch.float32).cuda()
        self.mm_rgb = torch.zeros((batch * 3), dtype=torch.float32).cuda()
        self.depth_values = torch.zeros((batch * N_SAMPLES), dtype=torch.float32).cuda()
        
        # NOTE INPUT GPU MEM
        self.input_gpu = gpuarray.empty(batch*self.in_ch, dtype=np.float32)
        # NOTE INPUT HOST MEM
        self.host_mem = cuda.pagelocked_empty((batch,self.in_ch), dtype=np.float32)

        # NOTE OUT GPU TENSOR
        self.out_ptrs = [self.mm_rgb, self.mm_density_add, self.mm_density_mul, self.depth_values]
        try:
            self.engine = self._load_engine(load_model)
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.outputs, self.bindings = self._allocate_buffers()
            print("[MM_Engine] Model loaded in {:.3}s".format(time.time() - t1))
        except Exception as e:
            raise RuntimeError('Build engine failed:', e) from e

    def bind_input(self, input):
        # NOTE preprocess img
        self.host_mem = input
        self.input_gpu.set_async(self.host_mem, stream=self.stream)

        # NOTE modify input binding
        self.bindings[0] = self.input_gpu.gpudata

    def run(self):
        # NOTE execute engine
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        self.stream.synchronize()
        mm_rgb, mm_density_add, mm_density_mul, depth_values  = [output.reshape(shape) for output, shape in zip(self.out_ptrs, self.shape_of_output)]        
        return mm_rgb, mm_density_add, mm_density_mul, depth_values

class RefineEngine(object):
    def _load_engine(self, model_path):
        with open(model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        print("Allocating buffer for engine I/O")
        outputs = []
        bindings = []
        # print("Batch size: ", self.engine.max_batch_size)
        out_ptr = 0
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                print("Input name: {}| Size: {}| Type: {}".format(binding, size, dtype))
                # NOTE init input binding with dummy int
                bindings.append(1000)
            else:
                print("Output name: {}| Size:{} | Type: {}".format(binding, size, dtype))
                out = self.out_ptrs[out_ptr]
                out_ptr += 1
                device_mem = GPUArray(out.shape, dtype=torch_dtype_to_numpy(out.dtype),
                         gpudata=Holder(out))
                
                outputs.append(device_mem)
                bindings.append(int(device_mem.gpudata))
        return outputs, bindings

    def __init__(self, load_model, batch=756*1008, in_ch=(3*NUM_NEIGHBOR) * N_SAMPLES + 6*(N_SAMPLES)):
        t1 = time.time()
        self.batch_size = batch
        self.in_ch = in_ch
        self.shape_of_output = [
                        (batch, N_SAMPLES),# refine_depth_values
                        (batch, 3), # refine_rgb
                        (batch, 3*N_SAMPLES), # points_offset
                        ]

        self.trt_logger = trt.Logger(trt.Logger.INFO)

        self.refine_depth_values = torch.zeros((batch * N_SAMPLES), dtype=torch.float32).cuda()
        self.refine_rgb = torch.zeros((batch * 3), dtype=torch.float32).cuda()
        self.points_offset = torch.zeros((batch * 3*N_SAMPLES), dtype=torch.float32).cuda()
        
        # NOTE INPUT GPU MEM
        self.input_gpu = gpuarray.empty(batch*self.in_ch, dtype=np.float32)
        # NOTE INPUT HOST MEM
        self.host_mem = cuda.pagelocked_empty((batch,self.in_ch), dtype=np.float32)
        self.input_gpu_host_mem = None

        # NOTE OUT GPU TENSOR
        self.out_ptrs = [self.refine_depth_values, self.refine_rgb, self.points_offset]
        try:
            self.engine = self._load_engine(load_model)
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.outputs, self.bindings = self._allocate_buffers()
            print("[MM_Engine] Model loaded in {:.3}s".format(time.time() - t1))
        except Exception as e:
            raise RuntimeError('Build engine failed:', e) from e

    def bind_input(self, input, warmup = False):
        if warmup:
            self.input_gpu_host_mem = input
            self.input_gpu = tensor_to_gpuarray(self.input_gpu_host_mem)
            self.bindings[0] = self.input_gpu.gpudata 
        else:
            self.input_gpu_host_mem.copy_(input)

    def run(self):
        # NOTE execute engine
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)

        self.stream.synchronize()
        refine_depth_values, refine_rgb, points_offset  = [output.reshape(shape) for output, shape in zip(self.out_ptrs, self.shape_of_output)]
        return refine_depth_values, refine_rgb, points_offset

class NeRFEngine(object):
    def _load_engine(self, model_path):
        with open(model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        print("Allocating buffer for engine I/O")
        outputs = []
        bindings = []
        # print("Batch size: ", self.engine.max_batch_size)
        out_ptr = 0
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                print("Input name: {}| Size: {}| Type: {}".format(binding, size, dtype))
                # NOTE init input binding with dummy int
                bindings.append(1000)
            else:
                print("Output name: {}| Size:{} | Type: {}".format(binding, size, dtype))
                out = self.out_ptrs[out_ptr]
                out_ptr += 1
                device_mem = GPUArray(out.shape, dtype=torch_dtype_to_numpy(out.dtype),
                         gpudata=Holder(out))
                
                outputs.append(device_mem)
                bindings.append(int(device_mem.gpudata))
        return outputs, bindings

    def __init__(self, load_model, batch=756*1008*N_SAMPLES, in_ch=[63,27]):
        t1 = time.time()
        self.batch_size = batch
        self.in_ch = in_ch
        self.shape_of_output = [(batch, 4)]

        self.trt_logger = trt.Logger(trt.Logger.INFO)

        self.out = torch.zeros((batch * 4), dtype=torch.float32).cuda()
        
        # NOTE INPUT GPU MEM
        self.input_gpu = gpuarray.empty(batch*self.in_ch[0], dtype=np.float32)
        self.input_host_mem = cuda.pagelocked_empty((batch,self.in_ch[0]), dtype=np.float32)
        self.input_gpu_host_mem = None

        self.input_dir_gpu = gpuarray.empty(batch*self.in_ch[1], dtype=np.float32)
        self.input_dir_host_mem = cuda.pagelocked_empty((batch,self.in_ch[1]), dtype=np.float32)

        # NOTE OUT GPU TENSOR
        self.out_ptrs = [self.out]
        # self.out_ptrs = [self.hm, self.features]
        try:
            self.engine = self._load_engine(load_model)
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.outputs, self.bindings = self._allocate_buffers()
            print("[NeRF_Engine] Model loaded in {:.3}s".format(time.time() - t1))
        except Exception as e:
            raise RuntimeError('Build engine failed:', e) from e

    def bind_input(self, input, warmup = False):
        if warmup:
            self.input_gpu_host_mem = input
            self.input_gpu = tensor_to_gpuarray(self.input_gpu_host_mem)
            self.bindings[0] = self.input_gpu.gpudata 
        else:
            self.input_gpu_host_mem.copy_(input)

    def bind_input_dir(self, input_dir):
        self.input_dir_host_mem = input_dir
        self.input_dir_gpu.set_async(self.input_dir_host_mem, stream=self.stream)
        self.bindings[1] = self.input_dir_gpu.gpudata 

    def run(self):
        # self.stream.synchronize()
        # NOTE execute engine
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)

        self.stream.synchronize()
        out  = [output.reshape(shape) for output, shape in zip(self.out_ptrs, self.shape_of_output)]
        return out[0]

# if __name__ == '__main__':
#     # setup
#     far_thresh, near_thresh = 1., 0.
#     embed_fn, _ = get_embedder(10, 0)
#     mm_input = np.load('dummy_inputs/mm_input.npy')
#     ray_origins = torch.from_numpy(np.load('dummy_inputs/ray_origins.npy')).cuda()
#     ray_directions = torch.from_numpy(np.load('dummy_inputs/ray_directions.npy')).cuda()
#     depth_values_gt = np.load("dummy_inputs/depth_values.npy")
#     depth_values0_gt = np.load("dummy_inputs/depth_values_0.npy")
#     depth_densities_gt = np.load("dummy_inputs/depth_densities.npy")
#     dst_gt = np.load("dummy_inputs/dst.npy")
#     mm_rgb_gt = np.load("dummy_inputs/mm_rgb.npy")
#     pts = np.load("dummy_inputs/pts.npy")

#     input = np.load("dummy_inputs/nerf_input.npy")
#     # input_holder = torch.zeros((input.shape[0], input.shape[1]), dtype=torch.float32).cuda()
#     input_holder = torch.from_numpy(input).flatten().cuda()
#     input_dir = np.load("dummy_inputs/nerf_input_dir.npy")
#     output = np.load("dummy_inputs/nerf_output.npy")

#     # create_engine
#     mm_engine = MMEngine('logs_minmax/fern_4_ressample_rt/minmaxrays_net_fp16.trt')
#     nerf_engine = NeRFEngine('logs_minmax/fern_4_ressample_rt/nerf_fp16.trt')

#     # bind inputs
#     mm_engine.bind_input(mm_input) # bind mm_input to mm engine
#     nerf_engine.bind_input_dir(input_dir) # bind viewdirs to nerf  
#     nerf_engine.bind_input(input_holder, warmup=True)

#     dst, depth_values, depth_densities, mm_rgb = mm_engine.run()
    
    
#     depth_values = torch.sigmoid(depth_values) * (far_thresh - near_thresh) + near_thresh
#     depth_densities = torch.relu(depth_densities)
#     dst = torch.relu(dst)
#     sort_out = torch.sort(depth_values, dim=-1)
#     depth_values = sort_out[0]
#     depth_densities = depth_densities.view(-1)[sort_out[1].view(-1)].view(depth_densities.shape)
#     query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
#     flat_query_points = query_points.view(-1, 3)
#     embed_xyz = embed_fn(flat_query_points).flatten()

    
#     nerf_engine.bind_input(embed_xyz) # bind query points to nerf
#     for _ in range(5):
#         t = time.time()
#         raw = nerf_engine.run()
#         print('nerf', time.time()-t)
#     # np.testing.assert_allclose(raw.cpu().numpy(), output, rtol=1e-03, atol=1e-05)


    