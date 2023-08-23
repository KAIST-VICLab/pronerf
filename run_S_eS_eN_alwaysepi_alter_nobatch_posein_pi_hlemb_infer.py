import os, sys

gpu_n = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_n  # args.gpu_no
print(f'Training on GPU {gpu_n}')

import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import inverse_warp
from math import log
from skimage.metrics import structural_similarity
from ssim_torch import ssim as r2l_ssim_func

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data, load_blender_data_infer
from load_LINEMOD import load_LINEMOD_data

from trt_infer_v2_blender import *
import geopoly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='configs/lego/lego_epinerf.txt',
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs_epi_blender/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

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
    parser.add_argument("--a_acc", type=float, default=0,
                        help='weight for accumulation regularization')
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
    parser.add_argument("--chunk", type=int, default=1024 * 64,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--full_image", action='store_true',
                        help='train with full image')
    parser.add_argument("--warp_gray", action='store_true',
                        help='warp grayscale image')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--N_point_ray_enc", type=int, default=32,
                        help='number of points to represent a ray')
    parser.add_argument("--k_ref", type=int, default=0,
                        help='number reference epipolar lines')
    parser.add_argument("--N_n", type=int, default=4,
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
    parser.add_argument("--use_trt", action='store_true',
                        help='use trt engine')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
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
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat[:, 0:3])
    if inputs_flat.shape[1] > 3:  # add epipolar features
        embedded = torch.cat((embedded, inputs_flat[:, 3::]), -1)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].repeat(1, inputs.shape[1], 1)  # .expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def run_network_gaussian(mean, cov, epi_feat, viewdirs, fn, embed_fn, embeddirs_fn, pos_basis_t, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    min_deg_point = 0
    max_deg_point = 10

    lifted_means, lifted_vars = (
                    lift_and_diagonalize(mean, cov, pos_basis_t))
    
    inputs = integrated_pos_enc(lifted_means, lifted_vars,
                                    min_deg_point, max_deg_point)

    if epi_feat is not None:
        embedded = torch.cat((inputs, epi_feat), -1)
    else:
        embedded = inputs
    embedded = torch.reshape(embedded, [-1, embedded.shape[-1]])

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].repeat(1, inputs.shape[1], 1)  # .expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, or_rays, target_pose, ref_poses, ref_rgbs, p_uv,
                  H, W, K, ref_K, k_ref, N_n, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    rays_radii = kwargs['rays_radii']
    for i in range(0, rays_flat.shape[0], chunk):
        kwargs['rays_radii'] = rays_radii[i:i + chunk]
        ret = render_rays(rays_flat[i:i + chunk], or_rays[i:i + chunk], p_uv[i:i + chunk], target_pose, ref_poses,
                          ref_rgbs, H, W, K, ref_K, k_ref, N_n, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(rays, or_rays, sh, **kwargs):

    # Render and reshape
    # all_ret = render_rays(rays, or_rays, **kwargs)
    all_ret = {}
    chunk_ratio = 4
    chunk=800*800 // chunk_ratio
    p_uv = kwargs['p_uv']
    mm_input = kwargs['mm_input']
    ro1 = kwargs['ro1']
    rd1 = kwargs['rd1']
    ref_pose_in = kwargs['ref_pose_in']
    rays_radii = kwargs['rays_radii']
    for i in range(0, rays.shape[0], chunk):
        kwargs['p_uv'] = p_uv[i:i + chunk]
        kwargs['mm_input'] = mm_input[i:i + chunk]
        kwargs['ro1'] = ro1[...,i:i + chunk]
        kwargs['rd1'] = rd1[...,i:i + chunk]
        kwargs['ref_pose_in'] = ref_pose_in[i:i + chunk]
        kwargs['rays_radii'] = rays_radii[i:i + chunk]

        ret = render_rays(rays[i:i + chunk], or_rays[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}


    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map0', 'rgb_map1', 'depth_map', 'nerf_depth']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, k_ref, N_n, chunk, ref_poses, ref_rgbs, render_kwargs,
                p_uv=None, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf

    chunk_ratio = 4
    chunk=800*800 // chunk_ratio

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs0 = []
    rgbs1 = []
    depths = []
    nerf_depths = []
    psnrs = []
    psnrs0 = []
    rgbs_n0 = []
    ssims = []
    nex_ssims = []
    r2l_ssims = []
    lpips_res = []
    lpips_res_alex = []

    render_kwargs['train_nerf'] = False
    render_kwargs['train_sampler'] = True
    # render_kwargs['ref_poses'] = ref_poses
    # render_kwargs['ref_rgbs'] = ref_rgbs


    t = time.time()
    time1, time2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i, c2w in enumerate(tqdm(render_poses)):
        t = time.time()
        render_kwargs['target_pose'] = c2w[:3, :4]
        render_kwargs['Hfull'] = H
        render_kwargs['Wfull'] = W

        
        # prepare input
        rays_o, rays_d = get_rays(H, W, K, c2w)
        rays_radii = get_rays_radii(H, W, K, c2w)

        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        N_rays = viewdirs.shape[0]

        sh = rays_d.shape  # [..., 3]

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
        rays_radii = torch.reshape(rays_radii, [-1, 1]).float()
        render_kwargs['rays_radii'] = rays_radii

        near, far = intersect_sphere(rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        rays = torch.cat([rays, viewdirs], -1)
        or_rays = rays

        ro1, rd1 = torch.transpose(rays_o, 0, 1).unsqueeze(0), torch.transpose(rays_d, 0, 1).unsqueeze(0)  # 1, 3, H*W
        ro1 = torch.cat([ro1, torch.ones(ro1.shape[0],1,ro1.shape[2], device = ro1.device)], dim=1)
        rd1 = torch.cat([rd1, torch.zeros(rd1.shape[0],1,rd1.shape[2], device = rd1.device)], dim=1)

        ro1, rd1 = ro1.expand(render_kwargs['N_samples'] * N_n, -1, -1), rd1.expand(render_kwargs['N_samples'] * N_n, -1, -1)
        render_kwargs['ro1'] = ro1
        render_kwargs['rd1'] = rd1

        render_kwargs['p_uv'] = p_uv

        # mm input
        pts, _ = compute_query_points_from_rays(rays_o, rays_d, near, far, render_kwargs['N_point_ray_enc'], False)
        m = pts
        m = m.view(N_rays, -1)
        m = torch.cat((m, p_uv), -1)
        render_kwargs['mm_input'] = m

        render_kwargs['N_n'] = N_n
        render_kwargs['ref_poses'] = ref_poses
        render_kwargs['ref_rgbs'] = ref_rgbs
        render_kwargs['ref_K'] = torch.Tensor(K.copy()).to(device)

        # nearest cam id
        rel_cam_dist = torch.sum((c2w[None,:, 3] - render_kwargs['ref_poses'][:, :, 3]) ** 2, 1) ** (1 / 2)
        _, rel_cam_idx = torch.sort(rel_cam_dist.detach(), dim=0)
        ref_nos = rel_cam_idx[:render_kwargs['N_n']]
        render_kwargs['ref_nos'] = ref_nos

        neighbor_images = torch.Tensor(render_kwargs['ref_rgbs'])[ref_nos].to(device)
        ref_pose = render_kwargs['ref_poses'][ref_nos]
        render_kwargs['ref_pose_in'] = ref_pose[:, :, 3].view(1, N_n * 3).expand(N_rays, N_n * 3)

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

        ref_rgb = torch.repeat_interleave(ref_rgb, repeats=render_kwargs['N_samples'], dim=0)
        ref_pose = torch.repeat_interleave(project_mat, repeats=render_kwargs['N_samples'], dim=0)
        render_kwargs['ref_pose'] = ref_pose
        render_kwargs['ref_rgb'] = ref_rgb

        if render_kwargs['use_trt']:
            # input_holder = torch.zeros(chunk*render_kwargs['N_samples'], 473).cuda() # bind dummy input xyz to nerf
            # render_kwargs['nerf_engine'].bind_input(input_holder, warmup=True)
            # _ = render_kwargs['nerf_engine'].run()

            input_holder = torch.zeros(chunk*render_kwargs['N_samples'], 116).cuda() # bind dummy input xyz to nerf
            render_kwargs['nerf_engine'].bind_input(input_holder, warmup=True)
            _ = render_kwargs['nerf_engine'].run()

            mm_input_holder = torch.zeros(chunk, 2 + 3 * render_kwargs['N_point_ray_enc']).cuda()
            render_kwargs['mm_engine'].bind_input(mm_input_holder, warmup=True)
            _ = render_kwargs['mm_engine'].run()
            
            refine_input_holder = torch.zeros(chunk, 2 + 3 * render_kwargs['N_samples'] + 3 * render_kwargs['N_n'] * render_kwargs['N_samples'] + 3 * render_kwargs['N_n']).cuda() #  bind input to refine engine
            render_kwargs['refine_engine'].bind_input(refine_input_holder, warmup=True)
            _ = render_kwargs['refine_engine'].run()

        # for _ in range(5):
        #     time1.record()
        #     rgb0, rgb1, depth, nerf_depth, extras = render(rays, or_rays, sh, **render_kwargs)
        #     time2.record()
        #     torch.cuda.synchronize(device=device)
        #     print('Render path time:', time1.elapsed_time(time2))

        rgb0, rgb1, depth, nerf_depth, extras = render(rays, or_rays, sh, **render_kwargs)

        rgbs0.append(rgb0.cpu().numpy())
        rgbs1.append(rgb1.cpu().numpy())
        depths.append(depth.cpu().numpy())
        nerf_depths.append(nerf_depth.cpu().numpy())

        if 'mm_rgb' in extras.keys():
            rgbs_n0.append(extras['mm_rgb'].cpu().numpy())

        if gt_imgs is not None and render_factor == 0:
            p = mse2psnr(img2mse(rgb1, gt_imgs[i]))
            psnrs.append(p)
            p0 = mse2psnr(img2mse(rgb0, gt_imgs[i]))
            psnrs0.append(p0)

            # ssims
            ssim = img2ssim(rgb1.cpu(), (gt_imgs[i]).cpu())
            r2l_ssim = r2l_ssim_func(
                rgb1.cpu()[None].permute(0, 3, 1, 2), (gt_imgs[i]).cpu()[None].permute(0, 3, 1, 2))
            nex_ssim = structural_similarity(rgb1.cpu().numpy(), (gt_imgs[i]).cpu(
            ).numpy(), win_size=11, multichannel=True, gaussian_weights=True)
            ssims.append(ssim)
            nex_ssims.append(nex_ssim)
            r2l_ssims.append(r2l_ssim)

            # lpips
            lpips_val = rgb_lpips(
                (gt_imgs[i]).cpu().numpy(), rgb1.cpu().numpy(), 'vgg', device)
            lpips_res.append(lpips_val)

            lpips_val = rgb_lpips(
                (gt_imgs[i]).cpu().numpy(), rgb1.cpu().numpy(), 'alex', device)
            lpips_res_alex.append(lpips_val)

        if savedir is not None:
            rgb8 = to8b(rgbs1[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            # rgb8 = to8b(rgbs0[-1])
            # filename = os.path.join(savedir, '{:03d}_rgb0.png'.format(i))
            # imageio.imwrite(filename, rgb8)

            # if 'mm_rgb' in extras.keys():
            #     rgb8 = to8b(rgbs_n0[-1])
            #     filename = os.path.join(savedir, '{:03d}_rgb00.png'.format(i))
            #     imageio.imwrite(filename, rgb8)

            # dmap = to8b(nerf_depths[-1] / np.percentile(nerf_depths[-1], 99))
            # filename = os.path.join(savedir, '{:03d}_nerf_depth.png'.format(i))
            # imageio.imwrite(filename, dmap)

            # dmap = to8b(depths[-1] / np.percentile(depths[-1], 99))
            # filename = os.path.join(savedir, '{:03d}_depth.png'.format(i))
            # imageio.imwrite(filename, dmap)

            # if gt_imgs is not None and render_factor == 0:
            #     rgb8 = to8b(np.abs(rgbs1[-1] - gt_imgs[i].cpu().numpy()))
            #     filename = os.path.join(savedir, '{:03d}_error.png'.format(i))
            #     imageio.imwrite(filename, rgb8)

    rgbs0 = np.stack(rgbs0, 0)
    rgbs1 = np.stack(rgbs1, 0)
    depths = np.stack(depths, 0)
    nerf_depths = np.stack(nerf_depths, 0)

    if len(rgbs_n0) > 0:
        rgbs_n0 = np.stack(rgbs_n0, 0)
        print(rgbs_n0.shape)

    if len(psnrs) > 0:
        mean_psnr = 0
        for this_psnr in psnrs:
            mean_psnr = mean_psnr + this_psnr / len(psnrs)
        print(f'Mean Test PSNR RGB1 {mean_psnr.detach().item()}')
        # mean_psnr = 0
        # for this_psnr in psnrs0:
        #     mean_psnr = mean_psnr + this_psnr / len(psnrs0)
        # print(f'Mean Test PSNR RGB0 {mean_psnr.detach().item()}')

    print('LPIPS vgg', round(np.array(lpips_res).mean(),3))
    print('LPIPS alex', round(np.array(lpips_res_alex).mean(),3))
    print('SSIMS', round(np.array(ssims).mean(),3))
    print('NEX SSIMS', round(np.array(nex_ssims).mean(),3))
    print('R2l SSIMS', round(np.array(r2l_ssims).mean(),3))

    return rgbs0, rgbs1, depths, nerf_depths, rgbs_n0

def model2onnx(model, in_ch, savedir, model_name, batch_size):
    """This function converts torch nn module to onnx format"""
    model.eval()
    fn = os.path.join(savedir, '{}.onnx'.format(model_name))
    if model_name == 'nerf':
        dummy_input = torch.randn(in_ch)[None].repeat(batch_size,1)
        torch.onnx.export(model, (dummy_input,), fn, verbose=True,
                                export_params=True, input_names = ['input'],
                                    output_names = ['output'],
                                    dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
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
    else:
        print('NO MODEL NAME:', model_name)


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn0, input_ch0 = get_embedder(args.multires, args.i_embed)
    embed_fn, input_ch = get_N_embedder(32)
    embed_rays = Pluecker()

    grad_vars = []
    input_ch_views = 0
    embeddirs_fn = None
    model_fine = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = args.netskips
    # model = NeRF_epiR1(D=args.netdepth, W=args.netwidth, input_ch=420,
    #                   input_ch_epi=(3+3) * args.N_n + 2, output_ch=output_ch,
    #                   skips=skips, input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    model = NeRF_epiR1(D=args.netdepth, W=args.netwidth, input_ch=input_ch0,
                      input_ch_epi=(3+3) * args.N_n + 2, output_ch=output_ch,
                      skips=skips, input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars.append({'params': model.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lrate})
    grad_vars.append({'params': embed_fn.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lrate})

    pos_basis_t = torch.from_numpy(geopoly.generate_basis('icosahedron', 2).T.copy()).float().to(device)
    # network_query_fn = lambda mean, cov, epi_feat, viewdirs, network_fn: run_network_gaussian(mean, cov, epi_feat, viewdirs, network_fn,
    #                                                                     embed_fn=embed_fn0,
    #                                                                     embeddirs_fn=embeddirs_fn,
    #                                                                     pos_basis_t = pos_basis_t,
    #                                                                     netchunk=args.netchunk)
    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn0,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)

    s_grad_vars = []
    s_grad_vars.append({'params': model.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lrate})
    s_grad_vars.append({'params': embed_fn.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lrate})

    model_mmray = MinMaxRay_Net(D=args.mmnetdepth, W=args.mmnetwidth,
                                input_ch=2 + input_ch * args.N_point_ray_enc
                                if args.mm_emb else 2 + 3 * args.N_point_ray_enc,
                                output_ch=3 * args.N_samples + 3, skips=args.mmnetskips)
    s_grad_vars.append({'params': model_mmray.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lrate})

    model_refine = MinMaxRay_Net(D=args.mmnetdepth, W=args.mmnetwidth,
                                 input_ch=2 + input_ch * args.N_samples + 3 * args.N_n * args.N_samples + 3 * args.N_n
                                 if args.mm_emb else 2 + 3 * args.N_samples + 3 * args.N_n * args.N_samples +
                                                     3 * args.N_n,
                                 output_ch=4 * args.N_samples + 3, skips=args.mmnetskips)
    s_grad_vars.append({'params': model_refine.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lrate})

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    s_optimizer = torch.optim.Adam(params=s_grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        s_optimizer.load_state_dict(ckpt['s_optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        model_mmray.load_state_dict(ckpt['mmr_network_fn_state_dict'])
        model_refine.load_state_dict(ckpt['refine_net_state_dict'])
        # embed_fn.load_state_dict(ckpt['embed_fn'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])


    # # export to onnx
    # model2onnx(model, in_ch = (3+3) * args.N_n + 2 + 63 + input_ch_views, savedir = os.path.join(args.basedir, args.expname), model_name='nerf', batch_size=1)
    # model2onnx(model_mmray, in_ch = 2 + 3 * args.N_point_ray_enc, savedir = os.path.join(args.basedir, args.expname), model_name='minmaxrays_net',batch_size=1)
    # model2onnx(model_refine, in_ch = 2 + 3 * args.N_samples + 3 * args.N_n * args.N_samples + 3 * args.N_n, savedir = os.path.join(args.basedir, args.expname), model_name='refine_net',batch_size=1)
    # breakpoint()

    ##########################

    nerf_engine, mm_engine, refine_engine = None, None, None
    if args.use_trt:
        nerf_engine =  NeRFEngine(os.path.join(args.basedir, args.expname, 'nerf_fp16.trt'))
        mm_engine = MMEngine(os.path.join(args.basedir, args.expname, 'minmaxrays_net_fp16.trt'))
        refine_engine = RefineEngine(os.path.join(args.basedir, args.expname, 'refine_net_fp16.trt'))

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
        'embed_fn': embed_fn0,
        'embeddirs_fn': embeddirs_fn,
        'randomize': True,
        'nerf_engine':nerf_engine,
        'mm_engine':mm_engine,
        'refine_engine':refine_engine,
        'use_trt': args.use_trt,
        'embed_rays': embed_rays,
        'pos_basis_t':pos_basis_t,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['randomize'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, s_optimizer


def intersect_sphere(rays_o, rays_d, origin=None, radius=2.5):
    if origin is None:
        origin = torch.zeros_like(rays_o)
    rays_o = rays_o - origin
    o = rays_o
    d = rays_d

    dot_o_o = torch.bmm(o[:, None], o[:, :, None]).squeeze(-1)
    dot_d_d = torch.bmm(d[:, None], d[:, :, None]).squeeze(-1)
    dot_o_d = torch.bmm(o[:, None], d[:, :, None]).squeeze(-1)

    a = dot_d_d
    b = 2 * dot_o_d
    c = dot_o_o - radius * radius
    disc = b * b - 4 * a * c
    t1 = (-b + torch.sqrt(disc + 1e-8)) / (2 * a)
    t2 = (-b - torch.sqrt(disc + 1e-8)) / (2 * a)

    t, _ = torch.sort(torch.cat([t1, t2], dim=-1), dim=-1)

    # sort t1 and t2 in order
    # the_far = t[...,1][...,None]
    # far_pt = rays_o[..., None, :] + rays_d[..., None, :] * the_far[..., :, None]
    # below_plane = (far_pt[...,-1] < 0).type_as(rays_o)
    # max_far = -rays_o[..., None, -1] / rays_d[..., None, -1]
    # the_far = the_far * (1 - below_plane) + max_far * below_plane
    return t[..., 0][..., None], t[..., 1][..., None]


def compute_query_points_from_rays(
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        near_thresh: float,
        far_thresh: float,
        N_point_ray_enc,
        randomize=True
) -> (torch.Tensor, torch.Tensor):
    z_step = torch.linspace(0., 1., N_point_ray_enc).to(ray_origins)
    depth_values = z_step * (far_thresh - near_thresh) + near_thresh
    # depth_values = torch.linspace(near_thresh, far_thresh, N_point_ray_enc).to(ray_origins)
    # depth_values = depth_values.unsqueeze(0)

    if randomize is True:
        noise_shape = [ray_origins.shape[0], N_point_ray_enc]
        noise_ = (1 / 5) * torch.normal(0.0, 1.0, size=noise_shape).to(ray_origins)
        noise_ = noise_ * (far_thresh - near_thresh) / N_point_ray_enc
        depth_values = noise_ + depth_values
        depth_values[depth_values < 0] = 0
        # depth_values[depth_values > far_thresh] = far_thresh
        depth_values, _ = torch.sort(depth_values, dim=-1)
    # else:
    # depth_values = depth_values.repeat(ray_origins.shape[0], 1)

    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
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

    rgb_map = rgb_map + (1. - acc_map[..., None])
    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch, or_ray_batch, p_uv, target_pose, ref_poses, ref_rgbs, Hfull, Wfull, ref_K,
                 N_n,
                network_fn,
                network_query_fn,
                N_samples,
                white_bkgd=False,
                raw_noise_std=0.,
                min_max_ray_net=None,
                refine_net=None,
                N_point_ray_enc=0,
                embed_fn=None,
                embeddirs_fn=None,
                train_sampler=False,
                pytest=False,
                **kwargs):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    near, far = ray_batch[:, 6:7], ray_batch[:, 7:8]  # [-1,1]

    m = kwargs['mm_input']

    if kwargs['use_trt']:
        kwargs['mm_engine'].bind_input(m)
        min_max_rays = kwargs['mm_engine'].run()
    else:
        min_max_rays = min_max_ray_net(m)

    mm_density_add = min_max_rays[:, N_samples:2 * N_samples]
    mm_density_mul = min_max_rays[:, 2 * N_samples:3 * N_samples]
    mm_rgb = torch.sigmoid(min_max_rays[:, 3 * N_samples:])

    depth_values = torch.sigmoid(min_max_rays[:, :N_samples]) * (far - near) + near  # B, Nsamples, H, W
    sort_out = torch.sort(depth_values, dim=-1)
    depth_values = sort_out[0]  # ! depth values are sorted, ndc space
    mm_density_add = torch.gather(mm_density_add, dim=1, index=sort_out[1])
    mm_density_mul = torch.gather(mm_density_mul, dim=1, index=sort_out[1])

    N_point_ray_enc = N_samples

    # This code no grad???
    depths = depth_values.view(1, 1, N_rays, N_point_ray_enc).expand(N_n, -1, -1, -1)  # k_ref, H, W, N_point_ray_enc
    depths = torch.permute(depths, (0, 3, 1, 2)).reshape(-1, 1, N_rays)  # k_ref * N_point_ray_enc, H, W

    C_color = 3
    ref_rgb = kwargs['ref_rgb']
    ref_pose = kwargs['ref_pose']

    ro1 = kwargs['ro1']
    rd1 = kwargs['rd1']
    warps, _ = inverse_warp.inverse_warp_rod1_rt2_coords_trt(ref_rgb, depths, ro1, rd1, ref_pose, padding_mode='zeros')
    
    warps_flat = warps.clone().view(1, N_n, N_point_ray_enc, C_color, 1, N_rays)

    warps_flat = torch.permute(warps_flat.view(N_n, N_point_ray_enc, C_color, N_rays),
                               (3, 1, 0, 2)).reshape(N_rays, N_point_ray_enc, C_color * N_n)
    # This code^ no grad???

    epi_pts = rays_o[..., None, :] + rays_d[..., None, :] * depth_values[..., :, None]

    m = epi_pts

    warps_flat_1 = warps_flat.view(N_rays, N_samples, -1)
    input_ref_poses = kwargs['ref_pose_in']
    # input_ref_poses = ref_poses[ref_nos, :, 3].view(1, N_n * 3).expand(N_rays, N_n * 3)
    net_input = torch.cat((m.view(N_rays, -1), warps_flat_1.view(N_rays, -1), p_uv, input_ref_poses), -1)

    if kwargs['use_trt']:
        kwargs['refine_engine'].bind_input(net_input)
        refine_output = kwargs['refine_engine'].run()
    else:
        refine_output = refine_net(net_input)

    points_offset = torch.tanh(refine_output[:, N_samples:4 * N_samples]).view(N_rays, N_samples, 3)
    refine_rgb = torch.sigmoid(refine_output[:, 4 * N_samples:])

    # Intersample refinement
    refine_depth_values = torch.sigmoid(refine_output[:, :N_samples])
    mids = .5 * (depth_values[..., 1:] + depth_values[..., :-1])
    upper = torch.cat([mids, 0.5 * (far + depth_values[..., -1:])], -1)  # upper cat far
    lower = torch.cat([0.5 * (near + depth_values[..., :1]), mids], -1)  # lower cat near
    refine_depth_values = lower + (upper - lower) * refine_depth_values

    mean_sample = torch.mean(refine_depth_values, dim=1, keepdim=True)

    # add point offset
    query_points_nerf = rays_o[..., None, :] + rays_d[..., None, :] * refine_depth_values[..., :, None]
    if train_sampler:
        query_points_nerf = query_points_nerf + 4 * (1e-2) * points_offset
    # gaussians_mean, gaussians_cov = rays_to_gaussian_embed(refine_depth_values, rays_o, rays_d, kwargs['rays_radii'], near, far)

    N_n0 = N_n
    N_point_ray_enc = N_samples

    depths = refine_depth_values.view(1, 1, N_rays, N_point_ray_enc).expand(N_n0, -1, -1, -1)  # k_ref, H, W, N_point_ray_enc
    depths = torch.permute(depths, (0, 3, 1, 2)).reshape(-1, 1, N_rays)  # k_ref * N_point_ray_enc, H, W

    warps, _ = inverse_warp.inverse_warp_rod1_rt2_coords_trt(ref_rgb, depths, ro1, rd1, ref_pose, padding_mode='zeros')
    
    warps_flat = warps.clone().view(1, N_n0, N_point_ray_enc, C_color, 1, N_rays)
    warps_flat = torch.permute(warps_flat.view(N_n0, N_point_ray_enc, C_color, N_rays),
                               (3, 1, 0, 2)).reshape(N_rays, N_point_ray_enc, C_color * N_n0)

    input_ref_posesn = kwargs['ref_pose_in']
    input_ref_posesn = input_ref_posesn[:, None].expand(N_rays, N_samples, N_n * 3)
    p_uv_n = p_uv.view(N_rays, 1, 2).expand(N_rays, N_samples, 2)

    if kwargs['use_trt']:
        inputs = torch.cat((query_points_nerf, warps_flat.view(N_rays, N_samples, -1), input_ref_posesn, p_uv_n), -1)
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = embed_fn(inputs_flat[:, 0:3])
        embedded = torch.cat((embedded, inputs_flat[:, 3::]), -1)
        input_dirs = viewdirs[:, None].repeat(1, inputs.shape[1], 1)  # .expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
        kwargs['nerf_engine'].bind_input(embedded)
        raw = kwargs['nerf_engine'].run()
        raw = raw.view(N_rays, N_samples, -1)

    else:
        raw = network_query_fn(
            torch.cat((query_points_nerf, warps_flat.view(N_rays, N_samples, -1), input_ref_posesn, p_uv_n), -1),
            viewdirs, network_fn)

    # if kwargs['use_trt']:
    #     min_deg_point = 0
    #     max_deg_point = 10
    #     pos_basis_t = kwargs['pos_basis_t']

    #     lifted_means, lifted_vars = (
    #                     lift_and_diagonalize(gaussians_mean, gaussians_cov, pos_basis_t))
        
    #     inputs = integrated_pos_enc(lifted_means, lifted_vars,
    #                                     min_deg_point, max_deg_point)
    #     epi_feat = torch.cat((warps_flat.view(N_rays, N_samples, -1), input_ref_posesn, p_uv_n), -1)
    #     embedded = torch.cat((inputs, epi_feat), -1)
    #     embedded = torch.reshape(embedded, [-1, embedded.shape[-1]])

    #     input_dirs = viewdirs[:, None].repeat(1, inputs.shape[1], 1)  # .expand(inputs.shape)
    #     input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
    #     embedded_dirs = embeddirs_fn(input_dirs_flat)
    #     embedded = torch.cat([embedded, embedded_dirs], -1)
    #     kwargs['nerf_engine'].bind_input(embedded)
    #     raw = kwargs['nerf_engine'].run()
    #     raw = raw.view(N_rays, N_samples, -1)

    # else:
    #     raw = network_query_fn(
    #         gaussians_mean,
    #         gaussians_cov,
    #         torch.cat((warps_flat.view(N_rays, N_samples, -1), input_ref_posesn, p_uv_n), -1),
    #         viewdirs, network_fn)

    raw_noise_std = 0
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, refine_depth_values, rays_d, raw_noise_std,
                                                                    white_bkgd, pytest=pytest,
                                                                    mm_density_add=mm_density_add,
                                                                    mm_density_mul=mm_density_mul)

    ret = {'rgb_map0': refine_rgb, 'rgb_map1': rgb_map, 'depth_map': mean_sample, 'nerf_depth': depth_map,
           'mm_rgb': mm_rgb}

    return ret


def train():
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
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
            near = 1e-6
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        # images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        images, poses, render_poses, hwf, i_split, i_ref = load_blender_data_infer(args.datadir, args.half_res, args.testskip)
        poses = poses[:, :3, :4]
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        alpha_channel = images[..., -1]

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res,
                                                                                    args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
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
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, s_optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand

    # Get all rays ready in memory
    print('get rays')
    rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
    print('done, concats')
    rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only [N_train, H, W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.astype(np.float32)
    print('done')

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)

    poses_ref = torch.stack([poses[i] for i in i_ref], 0) # only ref
    # poses_train = torch.stack([poses[i] for i in i_train], 0) # only ref

    rays_rgb = torch.Tensor(rays_rgb).to(device)
    alpha_channel = torch.Tensor(alpha_channel).to(device)
    alpha_channel = torch.stack([alpha_channel[i] for i in i_train], 0)
    K_ten = torch.Tensor(K.copy()).to(device)
    ref_K = K_ten.clone()
    p_u, p_v = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    the_p_uv = torch.stack((p_u.t() / W, p_v.t() / H), -1).type_as(rays_rgb)

    # Warp only grayscale
    if args.warp_gray:
        the_ref_rgb = torch.mean(rays_rgb[:, :, :, 2, :], dim=3, keepdim=True)
    else:
        # the_ref_rgb = rays_rgb[:, :, :, 2, :]
        the_ref_rgb = rays_rgb[i_ref, :, :, 2, :]

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            render_kwargs_test['train_nerf'] = True
            render_kwargs_test['train_sampler'] = True

            testsavedir = os.path.join(basedir, expname,
                                       'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            print('test poses shape', poses[i_test].shape)
            print(images.shape)
            # render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.k_ref, args.N_n, args.chunk,
            #             ref_poses=poses_ref, p_uv=the_p_uv.view(-1, 2),
            #             ref_rgbs=the_ref_rgb, render_kwargs=render_kwargs_test,
            #             gt_imgs=images[i_test], savedir=testsavedir)
            render_path(torch.Tensor(render_poses[:,:3,:4]).to(device), hwf, K, args.k_ref, args.N_n, args.chunk,
                        ref_poses=poses_ref, p_uv=the_p_uv.view(-1, 2),
                        ref_rgbs=the_ref_rgb, render_kwargs=render_kwargs_test,
                        gt_imgs=None, savedir=testsavedir)
            return

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
