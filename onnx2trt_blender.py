import tensorrt as trt
import sys
import os

gpu_n = '5'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_n  # args.gpu_no
print(f'Training on GPU {gpu_n}')
CHUNK_RATIO = 4

def get_engine(onnx_file_path="", engine_file_path="", fp16_mode=True, int8_mode=False, save_engine=True,max_batch_size=1, in_ch = 90, is_nerf = True):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        logger = trt.Logger(trt.Logger.VERBOSE)
        EXPLICIT_BATCH = []
        EXPLICIT_BATCH.append(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        with trt.Builder(logger) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, logger) as parser, builder.create_builder_config() as config:
            # config.max_workspace_size = 4194304
            config.max_workspace_size = 20 << 30
            builder.max_batch_size = max_batch_size
            if fp16_mode:
                config.set_flag(trt.BuilderFlag.FP16)

            profile = builder.create_optimization_profile()

            if is_nerf:
                # ! nerf input: input 63, input_dir 27
                # profile.set_shape("input", (max_batch_size, in_ch[0]),(max_batch_size, in_ch[0]),(max_batch_size, in_ch[0]))
                # profile.set_shape("input_dir", (max_batch_size, in_ch[1]),(max_batch_size, in_ch[1]),(max_batch_size, in_ch[1]))
                profile.set_shape("input", (max_batch_size//CHUNK_RATIO, in_ch),(max_batch_size//CHUNK_RATIO, in_ch),(max_batch_size//CHUNK_RATIO, in_ch))
            else:
                # ! refine input  + mm input   
                profile.set_shape("input", (max_batch_size//CHUNK_RATIO, in_ch),(max_batch_size//CHUNK_RATIO, in_ch),(max_batch_size//CHUNK_RATIO, in_ch))

            config.add_optimization_profile(profile)

            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())

            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

            engine = builder.build_engine(network, config)
            print("Completed creating Engine")

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    return build_engine(max_batch_size, save_engine)


if __name__ == '__main__':
    N_samples = 16
    N_point_ray_enc = 49
    num_neighbor = 4
    root_dir = 'logs_epi_blender/lego_16sampler_trt'


    # convert nerf: Batch size nsamples*h*w
    ONNX_NAME = os.path.join(root_dir, 'nerf.onnx')
    TRT_NAME = os.path.join(root_dir, 'nerf_fp16.trt')
    get_engine(onnx_file_path=ONNX_NAME,engine_file_path=TRT_NAME, max_batch_size=800*800*N_samples, in_ch=(3+3) * num_neighbor + 2 + 63 + 27, fp16_mode=True)
    # get_engine(onnx_file_path=ONNX_NAME,engine_file_path=TRT_NAME, max_batch_size=756*1008*8, in_ch=[3,3], fp16_mode=True)

    # convert mm rays: Batch h*w, order: mm_density_add, mm_density_mul, mm_rgb, depth_values
    ONNX_NAME = os.path.join(root_dir, 'minmaxrays_net.onnx')
    TRT_NAME = os.path.join(root_dir, 'minmaxrays_net_fp16.trt')
    get_engine(onnx_file_path=ONNX_NAME,engine_file_path=TRT_NAME, max_batch_size=800*800, in_ch=2 + 3 * N_point_ray_enc, fp16_mode=True, is_nerf = False)

    # convert refine model: Batch size h*w, order: refine_depth_values, refine_rgb, points_offset
    ONNX_NAME = os.path.join(root_dir, 'refine_net.onnx')
    TRT_NAME = os.path.join(root_dir, 'refine_net_fp16.trt')
    get_engine(onnx_file_path=ONNX_NAME,engine_file_path=TRT_NAME, max_batch_size=800*800, in_ch=2 + 3 * N_samples + 3 * num_neighbor * N_samples + 3 * num_neighbor, fp16_mode=True, is_nerf = False)