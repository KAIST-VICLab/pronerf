import os
import sys

print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
try:
    from ptflops import flops_counter
except ImportError:
    flops_counter = None
import inverse_warp
try:
    import line_profiler  # noqa: F401
except ImportError:
    line_profiler = None

import matplotlib.pyplot as plt

from run_nerf_helpers import *
try:
    from trt_infer_v2 import *
    TRT_IMPORT_ERROR = None
except ImportError as exc:
    TRT_IMPORT_ERROR = exc
    NeRFEngine = MMEngine = RefineEngine = None

from load_llff import load_llff_data, load_llff_data_infer

PROFILING = False
if PROFILING:
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='configs/fern.txt',
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs_trt/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--use_trt", action='store_true',
                        help='use trt inference')
    parser.add_argument("--export_only", action='store_true',
                        help='export ONNX models and exit before rendering')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netskips", type=int, default=[4], nargs='*',
                        help='skip connection layers')
    parser.add_argument("--a_mmrgb", type=float, default=0,
                        help='weight for mm rgb loss')
    parser.add_argument("--a_p", type=float, default=0,
                        help='weight for perceptual loss')
    parser.add_argument("--a_mmdisp", type=float, default=0,
                        help='weight for mm disp loss')
    parser.add_argument("--mmnetdepth", type=int, default=8,
                        help='layers in mm network')
    parser.add_argument("--mmnetwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--mmnetskips", type=int, default=[4], nargs='*',
                        help='skip connection layers')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help='regularization on weights')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--full_image", action='store_true',
                        help='train with full image')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='stage 2 checkpoint used for PyTorch/ONNX/TensorRT inference')
    parser.add_argument("--nerf_engine_path", type=str, default=None,
                        help='nerf trt model')
    parser.add_argument("--mm_engine_path", type=str, default=None,
                        help='mm engine path')
    parser.add_argument("--refine_engine_path", type=str, default=None,
                        help='refine engine path')
    parser.add_argument("--num_neighbor", type=int, default=4,
                    help='num neighbor frames')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--N_point_ray_enc", type=int, default=32,
                        help='number of points to represent a ray')
    parser.add_argument("--k_ref", type=int, default=4,
                        help='number reference epipolar lines')
    parser.add_argument("--rand_crop_size", type=int, default=100,
                        help='size of random crop')
    parser.add_argument("--mm_emb", action='store_true',
                        help='embed ray point encoding')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='dataset loader; this release supports llff')
    parser.add_argument("--white_bkgd", action='store_true',
                        help='render with a white background')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=5000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=10000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=10000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=10000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--max_images", type=int, default=None,
                        help='optional number of test images to render; useful for smoke tests')

    return parser

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)

    outputs_flat = fn(embedded, embedded_dirs)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render(rays, or_rays, sh, **kwargs):
    # Render and reshape
    all_ret = render_rays(rays, or_rays, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map0', 'rgb_map1', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, near=0., far=1., or_near=1., or_far=10.):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs0 = []
    rgbs1 = []
    depths = []
    psnrs = []
    image_macs_pp = []

    t1, t2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for i, c2w in enumerate(tqdm(render_poses)):
        render_kwargs['target_pose'] = c2w

        # Prepare the target rays and their original camera-space form.
        rays_o, rays_d = get_rays(H, W, K, c2w)
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

        or_rays_o = torch.reshape(rays_o, [-1, 3]).float()
        or_rays_d = torch.reshape(rays_d, [-1, 3]).float()
        or_near, or_far = or_near * torch.ones_like(or_rays_d[..., :1]), or_far * torch.ones_like(or_rays_d[..., :1])
        or_rays = torch.cat([or_rays_o, or_rays_d, or_near, or_far], -1)
        or_rays = torch.cat([or_rays, viewdirs], -1)

        ro1, rd1 = torch.transpose(or_rays_o, 0, 1).unsqueeze(0), torch.transpose(or_rays_d, 0, 1).unsqueeze(0)  # 1, 3, H*W
        ro1 = torch.cat([ro1, torch.ones(ro1.shape[0],1,ro1.shape[2], device = ro1.device)], dim=1)
        rd1 = torch.cat([rd1, torch.zeros(rd1.shape[0],1,rd1.shape[2], device = rd1.device)], dim=1)

        ro1, rd1 = ro1.expand(render_kwargs['N_samples'] * render_kwargs['num_neighbor'], -1, -1), rd1.expand(render_kwargs['N_samples'] * render_kwargs['num_neighbor'], -1, -1)
        render_kwargs['ro1'] = ro1
        render_kwargs['rd1'] = rd1

        sh = rays_d.shape # [..., 3]
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()

        near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        rays = torch.cat([rays, viewdirs], -1)

        # The sampler predicts sparse depths from Plucker ray features.
        pts, _ = compute_query_points_from_rays(
            rays_o, rays_d, 0., 1., render_kwargs['N_point_ray_enc'], randomize=False)
        plucker_pts = render_kwargs['embed_rays'](pts, rays_d[:,None,:].expand(-1,render_kwargs['N_point_ray_enc'],-1)) # nump_pts + origin
        plucker_pts = plucker_pts.view(-1, (render_kwargs['N_point_ray_enc'])*6)
        render_kwargs['mm_input'] = plucker_pts

        # Select nearby source views for epipolar color features.
        rel_cam_dist = torch.sum((c2w[None,:, 3] - render_kwargs['poses'][:, :, 3]) ** 2, 1) ** (1 / 2)
        _, rel_cam_idx = torch.sort(rel_cam_dist.detach(), dim=0)
        ref_nos = rel_cam_idx[:render_kwargs['num_neighbor']]
        render_kwargs['ref_nos'] = ref_nos

        neighbor_images = torch.Tensor(render_kwargs['images'])[ref_nos].to(device)
        ref_pose = render_kwargs['poses'][ref_nos]

        trans_ones = torch.eye(3).to(device)
        trans_ones[1,1] = -1
        trans_ones[2,2] = -1
        ref_K = render_kwargs['ref_K']
        project_mat = torch.bmm(trans_ones[None].expand(ref_pose.shape[0],-1,-1), ref_pose)
        project_mat = torch.bmm(ref_K[None].expand(ref_pose.shape[0],-1,-1), project_mat)

        ref_rgb = (neighbor_images.permute(0, 3, 1, 2))
        ref_rgb_sh = ref_rgb.shape
        ref_rgb = ref_rgb.unsqueeze(1).expand(-1,render_kwargs['N_samples'],-1,-1,-1).contiguous().view(ref_rgb_sh[0]*render_kwargs['N_samples'],ref_rgb_sh[1],ref_rgb_sh[2],ref_rgb_sh[3])
        ref_pose_sh = project_mat.shape
        ref_pose = project_mat.unsqueeze(1).expand(-1,render_kwargs['N_samples'],-1,-1).contiguous().view(ref_pose_sh[0]*render_kwargs['N_samples'],ref_pose_sh[1],ref_pose_sh[2])
        render_kwargs['ref_pose'] = ref_pose
        render_kwargs['ref_rgb'] = ref_rgb


        # TensorRT engines keep persistent device buffers, so bind static inputs once.
        if render_kwargs['use_trt']:
            input_dirs = viewdirs[:, None].repeat(1,render_kwargs['N_samples'],1)
            input_dirs_flat = input_dirs.view(-1,3)
            embedded_dirs = render_kwargs['embeddirs_fn'](input_dirs_flat)
            render_kwargs['nerf_engine'].bind_input_dir(embedded_dirs.cpu().numpy()) # bind viewdirs to nerf

            input_holder = torch.zeros(embedded_dirs.shape[0], 63).flatten().cuda() # bind dummy input xyz to nerf
            render_kwargs['nerf_engine'].bind_input(input_holder, warmup=True)
            _ = render_kwargs['nerf_engine'].run()

            render_kwargs['mm_engine'].bind_input(plucker_pts.cpu().numpy()) # bind mm_input to mm engine
            refine_input_holder = torch.zeros(plucker_pts.shape[0], (3* render_kwargs['num_neighbor']) * render_kwargs['N_samples'] + 6*(render_kwargs['N_samples'])).cuda() #  bind input to refine engine
            render_kwargs['refine_engine'].bind_input(refine_input_holder, warmup=True)
            _ = render_kwargs['refine_engine'].run()

        if render_kwargs['count_flops']:
            for k in ['network_fine', 'min_max_ray_net', 'refine_net']:
                render_kwargs[k].start_flops_count(ost=None, verbose=False, ignore_list=[])


        # Measure the full sparse-depth, refinement, and NeRF rendering path.
        for _ in range(20):
            t1.record()
            rgb0, rgb1, depth_map, _ = render(rays, or_rays, sh, **render_kwargs)
            t2.record()
            torch.cuda.synchronize(device=device)
            print('Render path time:', t1.elapsed_time(t2))

        total_macs = 0
        if render_kwargs['count_flops']:
            for k in [ 'min_max_ray_net', 'refine_net']:
                macs, params = render_kwargs[k].compute_average_flops_cost()
                if k == 'network_fine':
                    macs *= render_kwargs['N_samples']
                total_macs += macs
                render_kwargs[k].stop_flops_count()
                print(k, macs)
            image_macs_pp.append(total_macs)
        print('Total flops:', total_macs*2)

        # Save the final image and depth outputs for release evaluation.
        rgbs0.append(rgb0.cpu().numpy())
        rgbs1.append(rgb1.cpu().numpy())
        depths.append(depth_map.cpu().numpy())

        if gt_imgs is not None and render_factor == 0:
            p = mse2psnr(img2mse(rgb1, torch.Tensor(gt_imgs[i]).to(device)))
            psnrs.append(p)

        if savedir is not None:
            rgb8 = to8b(rgbs1[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            rgb8 = to8b(depths[-1]/np.max(depths[-1]))
            filename = os.path.join(savedir, 'depth_{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs0 = np.stack(rgbs0, 0)
    rgbs1 = np.stack(rgbs1, 0)
    depths = np.stack(depths, 0)

    if len(psnrs) > 0:
        mean_psnr = 0
        for this_psnr in psnrs:
            mean_psnr = mean_psnr + this_psnr / len(psnrs)
        print(psnrs)
        print(f'Mean Test PSNR {mean_psnr.detach().item()}')

    return rgbs0, rgbs1, depths, depths

def model2onnx(model, in_ch, savedir, model_name, batch_size):
    """This function converts torch nn module to onnx format"""
    model.eval()
    model_device = next(model.parameters()).device
    fn = os.path.join(savedir, '{}.onnx'.format(model_name))
    if model_name == 'nerf':
        dummy_input = torch.randn(in_ch[0], device=model_device)[None].repeat(batch_size,1)
        dummy_input_dir = torch.randn(in_ch[1], device=model_device)[None].repeat(batch_size,1)
        torch.onnx.export(model, (dummy_input, dummy_input_dir), fn, verbose=True,
                                export_params=True, input_names = ['input', 'input_dir'],
                                    output_names = ['output'],
                                    dynamic_axes={'input' : {0 : 'batch_size'}, 'input_dir' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
    elif model_name == 'minmaxrays_net':
        dummy_input = torch.randn(in_ch, device=model_device)[None].repeat(batch_size,1)
        torch.onnx.export(model, (dummy_input, ), fn, verbose=True,
                                export_params=True, input_names = ['input'],
                                    output_names = ['mm_rgb', 'mm_density_add', 'mm_density_mul', 'depth_values'],
                                    dynamic_axes={'input' : {0 : 'batch_size'},
                                                    'mm_rgb' : {0 : 'batch_size'},
                                                    'mm_density_add' : {0 : 'batch_size'},
                                                    'mm_density_mul' : {0 : 'batch_size'},
                                                    'depth_values' : {0 : 'batch_size'}})
    elif model_name == 'refine_net':
        dummy_input = torch.randn(in_ch, device=model_device)[None].repeat(batch_size,1)
        torch.onnx.export(model, (dummy_input, ), fn, verbose=True,
                                export_params=True, input_names = ['input'],
                                    output_names = ['refine_depth_values', 'refine_rgb', 'points_offset'],
                                    dynamic_axes={'input' : {0 : 'batch_size'},
                                                    'refine_depth_values' : {0 : 'batch_size'},
                                                    'refine_rgb' : {0 : 'batch_size'},
                                                    'points_offset' : {0 : 'batch_size'}})
    else:
        print('NO MODEL NAME:', model_name)
    

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    embed_rays = Pluecker()

    input_ch_views = 0
    embeddirs_fn = None
    grad_vars = []
    grad_vars_nerf = []

    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRFTRT(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    if not args.use_trt:
        model.to(device)

    model_fine = None
    model_fine = DoNeRFTRT(D=args.netdepth, W=args.netwidth,
                    n_in=input_ch + input_ch_views, n_out=output_ch, skip='auto')
    if not args.use_trt:
        model_fine.to(device)
    grad_vars_nerf.append({'params': model_fine.parameters(),
                    'weight_decay': args.weight_decay, 'lr': args.lrate})

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    model_mmray = MinMaxRaySamplerTRT_Net(D=args.mmnetdepth, W=args.mmnetwidth,
                                input_ch=2 + input_ch * args.N_point_ray_enc if args.mm_emb else
                                6 * args.N_point_ray_enc,
                                output_ch=3 * args.N_samples + 3, skips=args.mmnetskips, N_samples = args.N_samples)
    grad_vars.append({'params': model_mmray.parameters(),
                     'weight_decay': args.weight_decay, 'lr': args.lrate})
    
    
    model_refine = MinMaxRayEpiSamplerTRT_Net(D=args.mmnetdepth, W=args.mmnetwidth,
                                input_ch=input_ch * args.N_samples if args.mm_emb else
                                6 * (0+args.N_samples) + 3 * args.num_neighbor * args.N_samples,
                                output_ch=4 * args.N_samples + 3, skips=args.mmnetskips, N_samples = args.N_samples)
    grad_vars.append({'params': model_refine.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lrate})
    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    optimizer_nerf = torch.optim.Adam(params=grad_vars_nerf, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ckpt['network_fn_state_dict'])
        model_mmray.load_state_dict(ckpt['mmr_network_fn_state_dict'])
        model_refine.load_state_dict(ckpt['refine_net_state_dict'])
        model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    # Export ONNX modules used to build the TensorRT engines.
    model2onnx(model_fine, in_ch = [input_ch, input_ch_views], savedir = os.path.join(args.basedir, args.expname), model_name='nerf', batch_size=1)
    model2onnx(model_mmray, in_ch = (6) * args.N_point_ray_enc, savedir = os.path.join(args.basedir, args.expname), model_name='minmaxrays_net',batch_size=1)
    model2onnx(model_refine, in_ch = (3*args.num_neighbor) * args.N_samples + 6*(args.N_samples), savedir = os.path.join(args.basedir, args.expname), model_name='refine_net',batch_size=1)

    # TensorRT engines are optional; PyTorch inference is still supported.
    nerf_engine, mm_engine, refine_engine = None, None, None
    if args.use_trt:
        if TRT_IMPORT_ERROR is not None:
            raise ImportError(
                "TensorRT inference requested, but TensorRT/PyCUDA imports failed. "
                "Install requirements-trt.txt and verify CUDA/TensorRT paths."
            ) from TRT_IMPORT_ERROR
        nerf_engine =  NeRFEngine(os.path.join(args.basedir, args.expname, 'nerf_fp16.trt'))
        mm_engine = MMEngine(os.path.join(args.basedir, args.expname, 'minmaxrays_net_fp16.trt'))
        refine_engine = RefineEngine(os.path.join(args.basedir, args.expname, 'refine_net_fp16.trt'))
    elif flops_counter is not None:
        model_mmray = flops_counter.add_flops_counting_methods(model_mmray)
        model_refine = flops_counter.add_flops_counting_methods(model_refine)
        model_fine = flops_counter.add_flops_counting_methods(model_fine)
        model = flops_counter.add_flops_counting_methods(model)


    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'min_max_ray_net': model_mmray,
        'refine_net': model_refine,
        'N_point_ray_enc': args.N_point_ray_enc,
        'embed_fn': embed_fn,
        'embeddirs_fn': embeddirs_fn,
        'embed_rays':embed_rays,
        'randomize': True,
        'nerf_engine':nerf_engine,
        'mm_engine':mm_engine,
        'refine_engine':refine_engine,
        'num_neighbor': args.num_neighbor,
        'use_trt': args.use_trt,
        'count_flops': (not args.use_trt) and flops_counter is not None,

    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['randomize'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, optimizer_nerf

def compute_query_points_from_rays(
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        near_thresh: float,
        far_thresh: float,
        N_point_ray_enc,
        randomize=True
) -> (torch.Tensor, torch.Tensor):

    # Linear
    depth_values = torch.linspace(
        near_thresh, far_thresh, N_point_ray_enc).to(ray_origins)
    depth_values = depth_values.unsqueeze(0)

    query_points = ray_origins[..., None, :] + \
        ray_directions[..., None, :] * depth_values[..., :, None]
    return query_points, depth_values

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, mm_density_add=None, mm_density_mul=None, iter=1e6):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    inf_dist = torch.ones(dists[..., :1].shape, device = dists.device)*(1e10)
    dists = torch.cat([dists, inf_dist], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]

    alpha = raw2alpha(raw[...,3] + mm_density_add, dists)  # [N_rays, N_samples]
    alpha = alpha*torch.relu(mm_density_mul)

    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    return rgb_map, disp_map, acc_map, weights, depth_map

def render_rays(ray_batch, or_ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                min_max_ray_net=None,
                refine_net=None,
                N_point_ray_enc=0,
                embed_fn=None,
                embeddirs_fn=None,
                randomize=True,
                verbose=False,
                pytest=False,
                **kwargs):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    if kwargs['use_trt']:
        _, mm_density_add, mm_density_mul, depth_values = kwargs['mm_engine'].run()
    else:
        _, mm_density_add, mm_density_mul, depth_values = min_max_ray_net(kwargs['mm_input'])

    # Stage-1 sampler output is sorted before refinement and volume rendering.
    depth_values = depth_values * (far - near) + near
    sort_out = torch.sort(depth_values, dim=-1)
    depth_values = sort_out[0]
    mm_density_add = torch.gather(mm_density_add, dim =1, index = sort_out[1])
    mm_density_mul = torch.gather(mm_density_mul, dim =1, index = sort_out[1])

    depth_values_3d = 1/(1-depth_values - 1e-5)
    num_pts = N_samples
    num_neighbor = kwargs['num_neighbor']
    k_ref = kwargs['num_neighbor']
    ref_rgb = kwargs['ref_rgb']
    ref_pose = kwargs['ref_pose']
    ro1 = kwargs['ro1']
    rd1 = kwargs['rd1']

    # Project sparse samples into source views to form epipolar features.
    warp_H = 1
    warp_W = N_rays
    depths = depth_values_3d[None,None,:,:].expand(k_ref,-1,-1,-1) # k_ref, H, W, N_point_ray_enc
    depths = (depths.permute(0, 3, 1, 2)).reshape(-1, warp_H, warp_W)  # k_ref * N_point_ray_enc, H, W

    warps, _ = inverse_warp.inverse_warp_rod1_rt2_coords_trt(ref_rgb, depths, ro1, rd1, ref_pose, padding_mode='zeros')
    valid_warps_flat = warps.view(1, k_ref, num_pts, 3, warp_H, warp_W)
    
    epi_features = (valid_warps_flat.view(num_pts*num_neighbor, 3, warp_H*warp_W).permute(2,0,1)).reshape(-1, 3*num_pts*num_neighbor) # N rays, 3*num_pts
    epi_pts = rays_o[..., None, :] + rays_d[..., None, :] * depth_values[..., :, None]
    plucker_embed = kwargs['embed_rays'](epi_pts, rays_d[:, None, :].repeat(1, num_pts, 1)) # nump_pts + origin
    plucker_embed = plucker_embed.view(-1, num_pts, 6).view(N_rays, -1)

    # The refinement network adjusts sampler depths and offsets before NeRF rendering.
    refine_input = torch.cat([plucker_embed, epi_features], dim =1)


    if kwargs['use_trt']:
        kwargs['refine_engine'].bind_input(refine_input)
        refine_depth_values, _, points_offset = kwargs['refine_engine'].run()
    else:
        refine_depth_values, _, points_offset = refine_net(refine_input)

    
    points_offset = points_offset.view(N_rays, N_samples,3)

    mids = .5 * (depth_values[...,1:] + depth_values[...,:-1])
    upper = torch.cat([mids, 0.5*(far+depth_values[...,-1:])], -1) # upper cat far
    lower = torch.cat([0.5*(near+depth_values[...,:1]), mids], -1) # lower cat near
    refine_depth_values = lower + (upper - lower) * refine_depth_values
    epi_z_vals = refine_depth_values

    query_points_nerf = rays_o[..., None, :] + rays_d[..., None,
                                                    :] * epi_z_vals[..., :, None]
    query_points_nerf = query_points_nerf + (1e-2)*points_offset


    if kwargs['use_trt']:
        flat_query_points = query_points_nerf.view(-1, 3)
        embed_xyz = embed_fn(flat_query_points).flatten()
        kwargs['nerf_engine'].bind_input(embed_xyz)
        raw = kwargs['nerf_engine'].run()
        raw = raw.view(N_rays, N_samples, -1)
    else:
        raw = network_query_fn(query_points_nerf, viewdirs, network_fine)


    rgb_map, _, _, _, depth_map = raw2outputs(raw, epi_z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, mm_density_add=mm_density_add, mm_density_mul=mm_density_mul, iter=1e6)
    ret = {'rgb_map0': rgb_map, 'rgb_map1': rgb_map,'depth_map': depth_map}
    return ret


def train():

    parser = config_parser()
    args = parser.parse_args()
    if args.dataset_type != 'llff':
        raise ValueError('This cleaned release supports only dataset_type=llff.')

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test, i_ref = load_llff_data_infer(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        render_poses = render_poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    _, render_kwargs_test, start, _, _, _ = create_nerf(args)
    if args.export_only:
        print('Exported ONNX models; export_only requested, skipping render.')
        return

    bds_dict = {
        'near' : near,
        'far' : far,
    }

    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    poses = torch.Tensor(poses).to(device)

    K_ten = torch.Tensor(K.copy()).to(device)
    render_kwargs_test['i_train'] = i_train

    render_kwargs_test['images'] = images[i_ref]
    render_kwargs_test['poses'] = poses[i_ref]

    render_kwargs_test['ref_K'] = K_ten

    testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
        'test' if args.render_test else 'path', start))
    os.makedirs(testsavedir, exist_ok=True)
    if args.max_images is not None:
        i_test = i_test[:args.max_images]
    print('test poses shape', poses[i_test].shape)
    with torch.no_grad():
        render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
    print('Saved test set')



if __name__=='__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
