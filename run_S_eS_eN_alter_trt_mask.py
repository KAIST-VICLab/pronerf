import os
import sys
import threading

gpu_n = '6'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_n  # args.gpu_no
print(f'Training on GPU {gpu_n}')
import cv2

import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from ptflops import flops_counter
import inverse_warp
import line_profiler

import matplotlib.pyplot as plt

from run_nerf_helpers import *
from trt_infer_v2_mask import *

from load_llff import load_llff_data_infer, load_llff_cimgs
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

PROFILING = True
if PROFILING:
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
else:
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t1, t2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
np.random.seed(0)

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
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--pretrain_path", type=str, default=None,
                        help='specific weights npy file to reload pretrain network')
    parser.add_argument("--pretrain_depth_path", type=str, default=None,
                        help='specific weights npy file to reload pretrain depth')
    parser.add_argument("--num_neighbor", type=int, default=4,
                    help='num neighbor frames')
    
    parser.add_argument("--mask_path", type=str, default=None,
                        help='specific weights npy file to reload for mask network')
    parser.add_argument("--avr_path", type=str, default=None,
                        help='specific weights npy file to reload for avr network')
    
    # engine_path
    parser.add_argument("--nerf_engine_path", type=str, default=None,
                        help='nerf trt model')
    parser.add_argument("--mm_engine_path", type=str, default=None,
                        help='mm engine path')
    parser.add_argument("--refine_engine_path", type=str, default=None,
                        help='refine engine path')
    parser.add_argument("--mask_engine_path", type=str, default=None,
                        help='mask trt model')
    parser.add_argument("--avr_mm_engine_path", type=str, default=None,
                        help='avr mm engine path')
    parser.add_argument("--avr_refine_engine_path", type=str, default=None,
                        help='avr refine engine path')


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
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

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
        # embedded = torch.cat([embedded, embedded_dirs], -1)

    # outputs_flat = batchify(fn, netchunk)(embedded)
    outputs_flat = fn(embedded, embedded_dirs)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render(rays, or_rays, sh, **kwargs):
    # Render and reshape
    all_ret = render_rays(rays, or_rays, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map0', 'rgb_map1']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, near=0., far=1., or_near=1., or_far=10.):

    H, W, focal = hwf
    render_kwargs['compress_rate'] = 2/3.0
    render_kwargs['avr_N_samples'] = 8

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs0 = []
    rgbs1 = []
    avr_masks, shader_masks = [], []
    psnrs = []
    image_macs_pp = []

    # ssims = []
    # lpips_res = []
    # lpips_vgg = lpips.LPIPS(net="vgg").cuda()
    # lpips_vgg = lpips_vgg.eval()
    time1, time2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for i, c2w in enumerate(tqdm(render_poses)):
        render_kwargs['target_pose'] = c2w

        # Step 1: prepare render input
        rays_o, rays_d = get_rays(H, W, K, c2w)
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()
        N_rays = viewdirs.shape[0]

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

        
        pts, _ = compute_query_points_from_rays(
            rays_o, rays_d, 0., 1., render_kwargs['N_point_ray_enc'], randomize=False)  # ! this is ndc space
        plucker_pts = render_kwargs['embed_rays'](pts, rays_d[:,None,:].expand(-1,render_kwargs['N_point_ray_enc'],-1)) # nump_pts + origin
        plucker_pts = plucker_pts.view(-1, (render_kwargs['N_point_ray_enc'])*6)
        render_kwargs['mm_input'] = plucker_pts

        # nearest cam id
        rel_cam_dist = torch.sum((c2w[None,:, 3] - render_kwargs['poses'][:, :, 3]) ** 2, 1) ** (1 / 2)
        _, rel_cam_idx = torch.sort(rel_cam_dist.detach(), dim=0)
        ref_nos = rel_cam_idx[:render_kwargs['num_neighbor']]
        render_kwargs['ref_nos'] = ref_nos

        neighbor_images = torch.Tensor(render_kwargs['images'])[ref_nos].to(device)
        ref_pose = render_kwargs['poses'][ref_nos]
        render_kwargs['ref_pose_in'] = ref_pose[None].view(1, -1, 3*4).expand(int(N_rays*render_kwargs['compress_rate']), -1, -1)

        trans_ones = torch.eye(3).to(device)
        trans_ones[1,1] = -1
        trans_ones[2,2] = -1
        ref_K = render_kwargs['ref_K']

        # estimate w2c pose
        R_c2w = ref_pose[:, :, 0:3]
        t_c2w = ref_pose[:, :, 3, None]
        R_w2c = torch.transpose(R_c2w, 2, 1)
        t_w2c = -torch.bmm(R_w2c, t_c2w)

        w2c2 = torch.cat([R_w2c, t_w2c], dim = -1)

        project_mat = torch.bmm(trans_ones[None].expand(w2c2.shape[0],-1,-1), w2c2)
        project_mat = torch.bmm(ref_K[None].expand(w2c2.shape[0],-1,-1), project_mat)

        ref_rgb = (neighbor_images.permute(0, 3, 1, 2))

        shader_ref_rgb = torch.repeat_interleave(ref_rgb, repeats=render_kwargs['N_samples'], dim=0)
        shader_ref_pose = torch.repeat_interleave(project_mat, repeats=render_kwargs['N_samples'], dim=0)
        render_kwargs['shader_ref_pose'] = shader_ref_pose
        render_kwargs['shader_ref_rgb'] = shader_ref_rgb

        avr_ref_rgb = torch.repeat_interleave(ref_rgb, repeats=render_kwargs['avr_N_samples'], dim=0)
        avr_ref_pose = torch.repeat_interleave(project_mat, repeats=render_kwargs['avr_N_samples'], dim=0)
        render_kwargs['avr_ref_pose'] = avr_ref_pose
        render_kwargs['avr_ref_rgb'] = avr_ref_rgb

        avr_ref_rgb_1c = torch.round(avr_ref_rgb * 255)
        avr_ref_rgb_1c = (avr_ref_rgb_1c[:, 0] + (2**8) * avr_ref_rgb_1c[:, 1] + (2**16) * avr_ref_rgb_1c[:, 2]).unsqueeze(1)
        render_kwargs['avr_ref_rgb_1c'] = avr_ref_rgb_1c

        shader_ref_rgb_1c = torch.round(shader_ref_rgb * 255)
        shader_ref_rgb_1c = (shader_ref_rgb_1c[:, 0] + (2**8) * shader_ref_rgb_1c[:, 1] + (2**16) * shader_ref_rgb_1c[:, 2]).unsqueeze(1)
        render_kwargs['shader_ref_rgb_1c'] = shader_ref_rgb_1c

        # avr_ref_pose = torch.repeat_interleave(project_mat, repeats=render_kwargs['avr_N_samples'], dim=0)
        # render_kwargs['avr_w2c'] = avr_ref_pose

        input_dirs = viewdirs[:, None].repeat(1,render_kwargs['N_samples'],1)
        render_kwargs['embedded_dirs'] = render_kwargs['embeddirs_fn'](input_dirs)


        # Step 2 IMPORTANT: Bind input to gpu array
        if render_kwargs['use_trt']:
            
            input_dir_holder = torch.zeros(int(render_kwargs['N_samples']*N_rays*(1-render_kwargs['compress_rate'])), 27).cuda()
            render_kwargs['nerf_engine'].bind_input_dir(input_dir_holder, warmup=True)

            input_holder = torch.zeros(int(render_kwargs['N_samples']*N_rays*(1-render_kwargs['compress_rate'])), 63).cuda() # bind dummy input xyz to nerf
            render_kwargs['nerf_engine'].bind_input(input_holder, warmup=True)
            
            _ = render_kwargs['nerf_engine'].run()

            mm_input_holder = torch.zeros(int(N_rays*(1-render_kwargs['compress_rate'])), plucker_pts.shape[-1]).cuda()
            render_kwargs['mm_engine'].bind_input(mm_input_holder, warmup=True)
            _ = render_kwargs['mm_engine'].run()
            
            refine_input_holder = torch.zeros(int(N_rays*(1-render_kwargs['compress_rate'])), (3* render_kwargs['num_neighbor']) * render_kwargs['N_samples'] + 6*(render_kwargs['N_samples'])).cuda() #  bind input to refine engine
            render_kwargs['refine_engine'].bind_input(refine_input_holder, warmup=True)
            _ = render_kwargs['refine_engine'].run()
            
            # bind mask
            mask_input_holder = torch.zeros(plucker_pts.shape).cuda()
            render_kwargs['mask_engine'].bind_input(mask_input_holder, warmup=True)
            _ = render_kwargs['mask_engine'].run()

            # bind avr
            avr_mm_input_holder = torch.zeros(int(N_rays*render_kwargs['compress_rate']), plucker_pts.shape[-1]).cuda()
            render_kwargs['avr_mm_engine'].bind_input(avr_mm_input_holder, warmup=True)
            _ = render_kwargs['avr_mm_engine'].run()

            avr_refine_input_holder = torch.zeros(int(N_rays*render_kwargs['compress_rate']), 6 * render_kwargs['avr_N_samples'] + 3 * render_kwargs['num_neighbor'] * render_kwargs['avr_N_samples'] +
                                         3 * render_kwargs['avr_N_samples'] + 3 + (3 * 4) * render_kwargs['num_neighbor']).cuda() #  bind input to refine engine
            render_kwargs['avr_refine_engine'].bind_input(avr_refine_input_holder, warmup=True)
            _ = render_kwargs['avr_refine_engine'].run()

        # # Step 3: Warm up run
        # rgb0, rgb1, depth_map, extras = render(rays, or_rays, sh, **render_kwargs)

        # # count flops
        # if render_kwargs['count_flops']:
        #     for k in ['network_fine', 'min_max_ray_net', 'refine_net']:
        #         render_kwargs[k].start_flops_count(ost=None, verbose=False, ignore_list=[])


        # Step 4: Measure inference time
        for _ in range(5):
            time1.record()
            rgb0, rgb1, extras = render(rays, or_rays, sh, **render_kwargs)
            time2.record()
            torch.cuda.synchronize(device=device)
            print('Render path time:', time1.elapsed_time(time2))

        # total_macs = 0
        # if render_kwargs['count_flops']:
        #     for k in [ 'min_max_ray_net', 'refine_net']:
        #         macs, params = render_kwargs[k].compute_average_flops_cost()
        #         if k == 'network_fine':
        #             macs *= render_kwargs['N_samples']
        #         total_macs += macs
        #         render_kwargs[k].stop_flops_count()
        #         print(k, macs)
        #     image_macs_pp.append(total_macs)
        # print('Total flops:', total_macs*2)

        # Step 5: For logging purposes
        rgbs0.append(rgb0.cpu().numpy())
        rgbs1.append(rgb1.cpu().numpy())
        avr_masks.append(extras['avr_mask'].cpu().numpy())
        shader_masks.append(extras['shader_mask'].cpu().numpy())


        if gt_imgs is not None and render_factor == 0:
            p = mse2psnr(img2mse(rgb1, torch.Tensor(gt_imgs[i]).to(device)))
            psnrs.append(p)

            # # ssims
            # ssim = img2ssim(rgb1.permute(2, 0, 1)[None], (gt_imgs[i]).permute(2, 0, 1)[None].cuda())
            # ssims.append(ssim.cpu().numpy())

            # # lpips
            # scaled_gt = (gt_imgs[i]).permute(2, 0, 1)[None] * 2.0 - 1.0
            # scaled_pred = rgb1.permute(2, 0, 1)[None] * 2.0 - 1.0
            # lpips_val = lpips_vgg(scaled_gt.cuda(), scaled_pred.cuda())
            # lpips_res.append(lpips_val.detach().squeeze().cpu().numpy())

        if savedir is not None:
            rgb8 = to8b(rgbs1[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            rgb8 = to8b(avr_masks[-1])
            filename = os.path.join(savedir, 'avr_{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            rgb8 = to8b(shader_masks[-1])
            filename = os.path.join(savedir, 'shader_{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            rgb8 = to8b(rgbs0[-1])
            filename = os.path.join(savedir, 'mask_{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs0 = np.stack(rgbs0, 0)
    rgbs1 = np.stack(rgbs1, 0)
    # depths = np.stack(depths, 0)

    if len(psnrs) > 0:
        mean_psnr = 0
        for this_psnr in psnrs:
            mean_psnr = mean_psnr + this_psnr / len(psnrs)
        print(psnrs)
        print(f'Mean Test PSNR {mean_psnr.detach().item()}')

    # print('LPIPS', np.array(lpips_res).mean())
    # print('SSIMS', np.array(ssims).mean())
    return rgbs0, rgbs1, None, None

def model2onnx(model, in_ch, savedir, model_name, batch_size):
    """This function converts torch nn module to onnx format"""
    model.eval()
    fn = os.path.join(savedir, '{}.onnx'.format(model_name))
    if model_name == 'nerf':
        dummy_input = torch.randn(in_ch[0])[None].repeat(batch_size,1)
        dummy_input_dir = torch.randn(in_ch[1])[None].repeat(batch_size,1)
        torch.onnx.export(model, (dummy_input, dummy_input_dir), fn, verbose=True,
                                export_params=True, input_names = ['input', 'input_dir'],
                                    output_names = ['output'],
                                    dynamic_axes={'input' : {0 : 'batch_size'}, 'input_dir' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
    elif model_name == 'minmaxrays_net':
        dummy_input = torch.randn(in_ch)[None].repeat(batch_size,1)
        torch.onnx.export(model, (dummy_input, ), fn, verbose=True,
                                export_params=True, input_names = ['input'],
                                    output_names = ['output'],
                                    dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    elif model_name == 'refine_net':
        dummy_input = torch.randn(in_ch)[None].repeat(batch_size,1)
        torch.onnx.export(model, (dummy_input, ), fn, verbose=True,
                                export_params=True, input_names = ['input'],
                                    output_names = ['output'],
                                    dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    elif model_name == 'mask_net':
        dummy_input = torch.randn(in_ch)[None].repeat(batch_size,1)
        torch.onnx.export(model, (dummy_input, ), fn, verbose=True,
                                export_params=True, input_names = ['input'],
                                    output_names = ['output'],
                                    dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    elif model_name == 'avr_minmaxrays_net':
        dummy_input = torch.randn(in_ch)[None].repeat(batch_size,1)
        torch.onnx.export(model, (dummy_input, ), fn, verbose=True,
                                export_params=True, input_names = ['input'],
                                    output_names = ['output'],
                                    dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    elif model_name == 'avr_refine_net':
        dummy_input = torch.randn(in_ch)[None].repeat(batch_size,1)
        torch.onnx.export(model, (dummy_input, ), fn, verbose=True,
                                export_params=True, input_names = ['input'],
                                    output_names = ['output'],
                                    dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    else:
        print('NO MODEL NAME:', model_name)
    

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    embed_rays = Pluecker()

    input_ch_views = 0
    embeddirs_fn = None

    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    model = DoNeRFTRT(D=args.netdepth, W=args.netwidth,
                    n_in=input_ch + input_ch_views, n_out=output_ch, skip='auto')
    
    if not args.use_trt:
        model.to(device)

    model_fine = None

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    model_mmray = MinMaxRay_Net(D=args.mmnetdepth, W=args.mmnetwidth,
                                input_ch=2 + input_ch * args.N_point_ray_enc if args.mm_emb else
                                6 * args.N_point_ray_enc,
                                output_ch=3 * args.N_samples + 3, skips=args.mmnetskips)    
    
    model_refine = MinMaxRay_Net(D=args.mmnetdepth, W=args.mmnetwidth,
                                input_ch=input_ch * args.N_samples if args.mm_emb else
                                6 * (0+args.N_samples) + 3 * args.num_neighbor * args.N_samples,
                                output_ch=4 * args.N_samples + 3, skips=args.mmnetskips)
    
    model_mask = MinMaxRay_Net(D=3, W=args.mmnetwidth,
                                input_ch=6 * args.N_point_ray_enc,
                                output_ch=1, skips=args.mmnetskips)

    avr_N_samples = 8
    model_avr_mmray = MinMaxRay_Net(D=args.mmnetdepth, W=args.mmnetwidth,
                                input_ch=6 * args.N_point_ray_enc,
                                output_ch=2 * avr_N_samples+ 3, skips=args.mmnetskips)
    
    model_avr_refine = MinMaxRay_Net(D=args.mmnetdepth, W=args.mmnetwidth,
                                input_ch=6 * avr_N_samples + 3 * args.num_neighbor * avr_N_samples + 3 * avr_N_samples + 3 + (3 * 4) * args.num_neighbor,
                                output_ch=4*args.num_neighbor * avr_N_samples + args.num_neighbor, skips=args.mmnetskips)

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        # load mask
        mask_ckpt = torch.load(args.mask_path)

        # load avr
        avr_ckpt = torch.load(args.avr_path)

        # # if not (args.ft_path is not None and args.ft_path!='None'):
        start = ckpt['global_step']

        # Load model
        model.load_state_dict(ckpt['network_fine_state_dict'])
        model_mmray.load_state_dict(ckpt['mmr_network_fn_state_dict'])
        model_refine.load_state_dict(ckpt['refine_net_state_dict'])

        # Load mask network
        model_mask.load_state_dict(mask_ckpt['mmr_network_fn_state_dict'])

        # Load avr network
        model_avr_mmray.load_state_dict(avr_ckpt['mmr_network_fn_state_dict'])
        model_avr_refine.load_state_dict(avr_ckpt['refine_net_state_dict'])


    # # export to onnx
    # model2onnx(model, in_ch = [input_ch, input_ch_views], savedir = os.path.join(args.basedir, args.expname), model_name='nerf', batch_size=1)
    # model2onnx(model_mmray, in_ch = (6) * args.N_point_ray_enc, savedir = os.path.join(args.basedir, args.expname), model_name='minmaxrays_net',batch_size=1)
    # model2onnx(model_refine, in_ch = (3*args.num_neighbor) * args.N_samples + 6*(args.N_samples), savedir = os.path.join(args.basedir, args.expname), model_name='refine_net',batch_size=1)

    # model2onnx(model_mask, in_ch = 6 * args.N_point_ray_enc, savedir = os.path.join(args.basedir, args.expname), model_name='mask_net', batch_size=1)
    # model2onnx(model_avr_mmray, in_ch = 6 * args.N_point_ray_enc, savedir = os.path.join(args.basedir, args.expname), model_name='avr_minmaxrays_net',batch_size=1)
    # model2onnx(model_avr_refine, in_ch = 6 * avr_N_samples + 3 * args.num_neighbor * avr_N_samples + 3 * avr_N_samples + 3 + (3 * 4) * args.num_neighbor, savedir = os.path.join(args.basedir, args.expname), model_name='avr_refine_net',batch_size=1)
    # breakpoint()
    ##########################

    # # TRT Engine
    nerf_engine, mm_engine, refine_engine = None, None, None
    if args.use_trt:
        nerf_engine =  NeRFEngine(os.path.join(args.basedir, args.expname, 'nerf_fp16.trt'))
        mm_engine = MMEngine(os.path.join(args.basedir, args.expname, 'minmaxrays_net_fp16.trt'))
        refine_engine = RefineEngine(os.path.join(args.basedir, args.expname, 'refine_net_fp16.trt'))

        mask_engine =  MaskEngine(os.path.join(args.basedir, args.expname, 'mask_net_fp16.trt'))
        avr_mm_engine = AVRMMEngine(os.path.join(args.basedir, args.expname, 'avr_minmaxrays_net_fp16.trt'))
        avr_refine_engine = AVRRefineEngine(os.path.join(args.basedir, args.expname, 'avr_refine_net_fp16.trt'))
    else:
        model_mmray = flops_counter.add_flops_counting_methods(model_mmray)
        model_refine = flops_counter.add_flops_counting_methods(model_refine)
        model = flops_counter.add_flops_counting_methods(model)

        model_mask = flops_counter.add_flops_counting_methods(model_mask)
        model_avr_mmray = flops_counter.add_flops_counting_methods(model_avr_mmray)
        model_avr_refine = flops_counter.add_flops_counting_methods(model_avr_refine)

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
        'mask_engine':mask_engine,
        'avr_mm_engine':avr_mm_engine,
        'avr_refine_engine':avr_refine_engine,
        'num_neighbor': args.num_neighbor,
        'use_trt': args.use_trt,
        'count_flops': not args.use_trt,
        'avr_min_max_ray_net': model_avr_mmray,
        'avr_refine_net': model_avr_refine,
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

    return render_kwargs_train, render_kwargs_test, start, None, None, None

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

# @profile
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
    or_rays_o, or_rays_d = or_ray_batch[:, 0:3], or_ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    plucker_pts = kwargs['mm_input']

    # 1. Mask pipline
    # t1.record()
    kwargs['mask_engine'].bind_input(plucker_pts)
    mm_mask = kwargs['mask_engine'].run()
    mm_mask = torch.sigmoid(mm_mask)

    compress_rate = kwargs['compress_rate']
    _, avr_index = torch.topk(mm_mask, k = int(N_rays*compress_rate), dim = 0)
    avr_index = avr_index.squeeze(-1)

    _, shader_index = torch.topk(mm_mask, k = int(N_rays*(1-compress_rate)), dim = 0, largest = False)
    shader_index = shader_index.squeeze(-1)
    # t2.record()
    # torch.cuda.synchronize()
    # print('mask:', t1.elapsed_time(t2))

    # 3 avr pipeline
    # t1.record()
    avr_N_samples = 8
    avr_N_rays = int(N_rays*compress_rate)
    avr_plucker_pts = plucker_pts[avr_index]
    avr_rays_o, avr_rays_d = rays_o[avr_index], rays_d[avr_index] 
    avr_or_rays_o, avr_or_rays_d = or_rays_o[avr_index], or_rays_d[avr_index] 
    avr_viewdirs = viewdirs[avr_index]
    avr_bounds = bounds[avr_index]
    avr_near, avr_far = near[avr_index], far[avr_index]

    avr_rgb_map = render_avr(avr_plucker_pts,
                             8,
                            avr_near,
                            avr_far,
                            avr_N_rays,
                            avr_or_rays_o,
                            avr_or_rays_d,
                            avr_rays_o,
                            avr_rays_d,
                            avr_viewdirs,
                            avr_index,
                            **kwargs
                            )
    # t2.record()
    # torch.cuda.synchronize()
    # print('avr:', t1.elapsed_time(t2))

    # 2 shader pipeline
    # t1.record()
    shader_N_rays = int(N_rays*(1-compress_rate))
    shader_plucker_pts = plucker_pts[shader_index]
    shader_rays_o, shader_rays_d = rays_o[shader_index], rays_d[shader_index] 
    shader_viewdirs = viewdirs[shader_index]
    shader_embedded_dirs = kwargs['embedded_dirs'][shader_index].view(-1, 27)
    shader_bounds = bounds[shader_index]
    shader_near, shader_far = near[shader_index], far[shader_index]
    shader_or_rays_o, shader_or_rays_d = or_rays_o[shader_index], or_rays_d[shader_index] 

    
    shader_rgb_map = render_shader(min_max_ray_net, 
                                   refine_net,
                                   shader_plucker_pts,
                                   N_samples,
                                   shader_near,
                                   shader_far,
                                   shader_N_rays,
                                   shader_or_rays_o,
                                   shader_or_rays_d,
                                   shader_rays_o,
                                   shader_rays_d,
                                   shader_viewdirs,
                                   network_query_fn,
                                   network_fn,
                                   white_bkgd,
                                   pytest,
                                   embed_fn,
                                   shader_embedded_dirs,
                                   shader_index,
                                   **kwargs)
    # t2.record()
    # torch.cuda.synchronize()
    # print('shader:', t1.elapsed_time(t2))


    rgb_map = torch.zeros((N_rays, 3))
    avr_mask = torch.zeros((N_rays, 1))
    shader_mask = torch.zeros((N_rays, 1))

    rgb_map[shader_index] = shader_rgb_map
    rgb_map[avr_index] = avr_rgb_map

    shader_mask[shader_index] = 1.0
    avr_mask[avr_index] = 1.0

    ret = {'rgb_map0': mm_mask, 'rgb_map1': rgb_map, 'shader_mask': shader_mask, 'avr_mask': avr_mask}
    return ret

# @profile
def render_shader(min_max_ray_net, 
                  refine_net, 
                  shader_plucker_pts, 
                  N_samples,
                  shader_near,
                  shader_far,
                  shader_N_rays,
                  shader_or_rays_o,
                  shader_or_rays_d,
                  shader_rays_o,
                  shader_rays_d,
                  shader_viewdirs,
                  network_query_fn,
                  network_fn,
                  white_bkgd,
                  pytest,
                embed_fn,
                shader_embedded_dirs,
                shader_index,
                  **kwargs
                  ):
    
    kwargs['mm_engine'].bind_input(shader_plucker_pts)
    min_max_rays = kwargs['mm_engine'].run()

    mm_rgb = torch.sigmoid(min_max_rays[:, 3*N_samples:])
    mm_density_add = min_max_rays[:, N_samples:2*N_samples]
    mm_density_mul = min_max_rays[:, 2*N_samples:3*N_samples]

    depth_values = torch.sigmoid(min_max_rays[:, :N_samples]) * (shader_far - shader_near) + shader_near  # B, Nsamples, H, W
    sort_out = torch.sort(depth_values, dim=-1)
    depth_values = sort_out[0]  # ! depth values are sorted, ndc space
    mm_density_add = torch.gather(mm_density_add, dim =1, index = sort_out[1])
    mm_density_mul = torch.gather(mm_density_mul, dim =1, index = sort_out[1])
    depth_values_3d = 1/(1-depth_values - 1e-5)  #! convert ndc zval to 3d zval

    with torch.no_grad():
        num_pts = N_samples
        num_neighbor = kwargs['num_neighbor']
        k_ref = kwargs['images'].shape[0]
        
        ref_rgb = kwargs['shader_ref_rgb']
        ref_rgb_1c = kwargs['avr_ref_rgb_1c']
        ref_pose = kwargs['shader_ref_pose']

        ro1 = kwargs['ro1'][:,:,shader_index]
        rd1 = kwargs['rd1'][:,:,shader_index]

        warp_H = 1
        warp_W = shader_N_rays
        depths = depth_values_3d[None,None,:,:].expand(k_ref,-1,-1,-1) # k_ref, H, W, N_point_ray_enc
        depths = (depths.permute(0, 3, 1, 2)).reshape(-1, warp_H, warp_W)  # k_ref * N_point_ray_enc, H, W

        warps, _ = inverse_warp.inverse_warp_rod1_rt2_coords_trt(ref_rgb, depths, ro1, rd1, ref_pose, padding_mode='zeros')
        # warps, _ = inverse_warp.inverse_warp_rod1_rt2_coords_trt_1c(ref_rgb_1c, depths, ro1, rd1, ref_pose, padding_mode='zeros')
        warps_flat = warps.view(1, k_ref, num_pts, 3, warp_H, warp_W)

        epi_features = (warps_flat.view(num_pts*num_neighbor, 3, warp_H*warp_W).permute(2,0,1)).reshape(-1, 3*num_pts*num_neighbor) # N rays, 3*num_pts

    epi_pts = shader_rays_o[..., None, :] + shader_rays_d[..., None, :] * depth_values[..., :, None]
    plucker_embed = kwargs['embed_rays'](epi_pts, shader_rays_d[:, None, :].expand(-1, num_pts, -1)) # nump_pts + origin
    plucker_embed = plucker_embed.view(-1, num_pts, 6).view(shader_N_rays, -1)
    refine_input = torch.cat([plucker_embed, epi_features], dim =1)

    kwargs['refine_engine'].bind_input(refine_input)
    refine_output = kwargs['refine_engine'].run()

    refine_depth_values = torch.sigmoid(refine_output[:,:N_samples])
    points_offset = torch.tanh(refine_output[:, N_samples:4*N_samples]).view(shader_N_rays, N_samples,3)

    mids = .5 * (depth_values[...,1:] + depth_values[...,:-1])
    upper = torch.cat([mids, 0.5*(shader_far+depth_values[...,-1:])], -1) # upper cat far
    lower = torch.cat([0.5*(shader_near+depth_values[...,:1]), mids], -1) # lower cat near
    refine_depth_values = lower + (upper - lower) * refine_depth_values
    epi_z_vals = refine_depth_values

    query_points_nerf = shader_rays_o[..., None, :] + shader_rays_d[..., None,
                                                    :] * epi_z_vals[..., :, None]  # ! this is ndc space
    query_points_nerf = query_points_nerf + (1e-2) * points_offset

    flat_query_points = query_points_nerf.view(-1, 3)
    embed_xyz = embed_fn(flat_query_points)

    kwargs['nerf_engine'].bind_input(embed_xyz) # bind query points to nerf
    kwargs['nerf_engine'].bind_input_dir(shader_embedded_dirs) # bind query points to nerf

    raw = kwargs['nerf_engine'].run()
    raw = raw.view(shader_N_rays, N_samples, -1)

    iter = kwargs.get('iter', 1e6)
    raw_noise_std = 0
    shader_rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, epi_z_vals, shader_rays_d, raw_noise_std, white_bkgd,
                                                                 pytest=pytest, mm_density_add=mm_density_add,
                                                                 mm_density_mul=mm_density_mul, iter=iter)
    return shader_rgb_map

# @profile
def render_avr(avr_plucker_pts, 
                N_samples,
                avr_near,
                avr_far,
                avr_N_rays,
                avr_or_rays_o,
                avr_or_rays_d,
                avr_rays_o,
                avr_rays_d,
                avr_viewdirs,
                avr_index,
                **kwargs
                  ):
    
    kwargs['avr_mm_engine'].bind_input(avr_plucker_pts)
    F_theta0_out = kwargs['avr_mm_engine'].run()
    depth_values = torch.sigmoid(F_theta0_out[:, :N_samples]) * (avr_far - avr_near) + avr_near

    sort_out = torch.sort(depth_values, dim=-1)
    depth_values = sort_out[0] # ! depth values are sorted, ndc space

    # Get epipolar colors
    depth_values_3d = 1 / (1 - depth_values - 1e-6)  # ! convert ndc zval to 3d zval
    with torch.no_grad():
        num_pts = N_samples
        num_neighbor = kwargs['num_neighbor']
        k_ref = kwargs['images'].shape[0]

        ref_rgb = kwargs['avr_ref_rgb']
        ref_rgb_1c = kwargs['avr_ref_rgb_1c']

        ref_pose = kwargs['avr_ref_pose']
        

        ro1 = kwargs['ro1'][:,:,avr_index]
        rd1 = kwargs['rd1'][:,:,avr_index]

    warp_H = 1
    warp_W = avr_N_rays
    depths = depth_values_3d[None, None, :, :].expand(k_ref,-1,-1,-1)  # k_ref, H, W, N_point_ray_enc
    depths = (depths.permute(0, 3, 1, 2)).reshape(-1, warp_H, warp_W)  # k_ref * N_point_ray_enc, H, W

    warps, _ = inverse_warp.inverse_warp_rod1_rt2_coords_trt(ref_rgb, depths, ro1, rd1, ref_pose, padding_mode='border')
    # warps, _ = inverse_warp.inverse_warp_rod1_rt2_coords_trt_1c(ref_rgb_1c, depths, ro1, rd1, ref_pose, padding_mode='border')
    
    warps_flat = warps.view(1, k_ref, num_pts, 3, warp_H, warp_W)

    ref_pose_in = kwargs['ref_pose_in']
    epi_features = (warps_flat.view(num_neighbor, num_pts, 3, warp_H * warp_W).permute(3, 1, 0, 2)). \
        reshape(-1, num_pts, num_neighbor * 3)
    epi_var = torch.abs(torch.var(epi_features.view(avr_N_rays, N_samples, num_neighbor, 3), dim=2))

    epi_pts = avr_rays_o[..., None, :] + avr_rays_d[..., None, :] * depth_values[..., :, None]
    plucker_embed = kwargs['embed_rays'](epi_pts, avr_rays_d[:, None, :].expand(-1, num_pts, -1)) # nump_pts + origin
    plucker_embed = plucker_embed.view(-1, num_pts, 6)
    net_input = torch.cat((plucker_embed.view(avr_N_rays, -1), epi_features.view(avr_N_rays, -1), epi_var.view(avr_N_rays, -1),
                           avr_viewdirs.view(avr_N_rays, -1), ref_pose_in.view(avr_N_rays, -1)), -1)

    kwargs['avr_refine_engine'].bind_input(net_input)
    F_theta1_output = kwargs['avr_refine_engine'].run()

    weights0 = F_theta1_output[:, 0:num_neighbor * N_samples]
    combine = torch.sigmoid(F_theta1_output[:, num_neighbor * N_samples:num_neighbor * N_samples+num_neighbor])
    combine = combine / torch.sum(combine, dim=1, keepdim=True)

    # Approximated volumetric rendering (AVR)
    weights = F.softmax(weights0.view(avr_N_rays, N_samples, num_neighbor), dim=1)
    refine_rgbs = torch.sum(weights.view(avr_N_rays, N_samples, num_neighbor, 1) *
                       epi_features.view(avr_N_rays, N_samples, num_neighbor, 3), dim=1)
    mean_depth = torch.sum(depth_values[:, :, None] * weights.view(avr_N_rays, N_samples, num_neighbor), dim=-1)
    mean_depth = torch.mean(mean_depth, dim=1, keepdim=True)

    avr_rgb_map = torch.sum(combine[..., None] * refine_rgbs, 1)
    return avr_rgb_map

def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test, i_ref = load_llff_data_infer(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify, num_neighbor=args.num_neighbor)

        low_imgs = load_llff_cimgs(args.datadir, args.factor)

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
        # i_train = np.array([i for i in np.arange(int(images.shape[0])) if
        #                 (i not in i_test and i not in i_val and i!=23 and i!=25)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

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

    bds_dict = {
        'near' : near,
        'far' : far,
    }

    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    # images = torch.Tensor(images)
    poses = torch.Tensor(poses).to(device)
    low_imgs = torch.Tensor(low_imgs).to(device)

    # update train val id
    K_ten = torch.Tensor(K.copy()).to(device)
    render_kwargs_test['i_train'] = i_train
    render_kwargs_test['images'] = low_imgs[i_ref]
    render_kwargs_test['poses'] = poses[i_ref]
    render_kwargs_test['ref_K'] = K_ten

    testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
        'test' if args.render_test else 'path', start))
    os.makedirs(testsavedir, exist_ok=True)

    os.makedirs(testsavedir, exist_ok=True)
    print('test poses shape', poses[i_test].shape)
    with torch.no_grad():
        render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
    print('Saved test set')


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()