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
from loss_functions import perceptual_loss, vgg
import inverse_warp
from math import log

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

# TODO ray - > few sample points - > epipolar lines - > rgb0, sample points, or nerual rendering0 -> Nerf

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='configs/fern.txt',
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs_epi/',
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

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
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
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render(H, h, W, w, K, ref_K, k_ref, coords=None, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           target_pose=None, ref_poses=None, ref_rgbs=None,
           near=0., far=1., or_near=1., or_far=10.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if coords is None:
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1)  # (H, W, 2)
        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
        coords = coords.type_as(rays_o) / W

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    # Create original ray batch
    or_rays_o = torch.reshape(rays_o, [-1, 3]).float()
    or_rays_d = torch.reshape(rays_d, [-1, 3]).float()
    or_near, or_far = or_near * torch.ones_like(or_rays_d[..., :1]), or_far * torch.ones_like(or_rays_d[..., :1])
    or_rays = torch.cat([or_rays_o, or_rays_d, or_near, or_far], -1)
    if use_viewdirs:
        or_rays = torch.cat([or_rays, viewdirs], -1)

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = render_rays(rays, or_rays, coords, target_pose, ref_poses, ref_rgbs, h, w, ref_K, k_ref, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map0', 'rgb_map1', 'depth_map', 'mean_warp']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, k_ref, chunk, ref_poses, ref_rgbs, render_kwargs,
                gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs0 = []
    rgbs1 = []
    depths = []
    mean_warps = []
    psnrs = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        # print(i, time.time() - t)
        t = time.time()
        rgb0, rgb1, depth, mean_warp, extras = render(H, H, W, W, K, torch.Tensor(K.copy()).to(device), k_ref,
                                                       target_pose=c2w, ref_poses=ref_poses,
                                                       ref_rgbs=ref_rgbs, chunk=chunk, c2w=c2w[:3, :4],
                                                       **render_kwargs)
        rgbs0.append(rgb0.cpu().numpy())
        rgbs1.append(rgb1.cpu().numpy())
        depths.append(depth.cpu().numpy())
        mean_warps.append(mean_warp.cpu().numpy())

        # if i == 0:
        #     print(rgb0.shape)

        if gt_imgs is not None and render_factor==0:
            p = mse2psnr(img2mse(rgb1, gt_imgs[i]))
            psnrs.append(p)

        if savedir is not None:
            rgb8 = to8b(rgbs1[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs0 = np.stack(rgbs0, 0)
    rgbs1 = np.stack(rgbs1, 0)
    depths = np.stack(depths, 0)
    mean_warps = np.stack(mean_warps, 0)

    if len(psnrs) > 0:
        mean_psnr = 0
        for this_psnr in psnrs:
            mean_psnr = mean_psnr + this_psnr / len(psnrs)
        print(f'Mean Test PSNR {mean_psnr.detach().item()}')

    return rgbs0, rgbs1, depths, mean_warps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    grad_vars = []
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = args.netskips
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars.append({'params': model.parameters(), 'weight_decay': 0, 'lr': args.lrate})

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars.append({'params': model_fine.parameters(), 'weight_decay': 0, 'lr': args.lrate})

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)

    # Input is ray points. Output is RGB0, sample positions
    model_mmray = MinMaxRayS1Conv_Net(D=args.mmnetdepth, W=args.mmnetwidth,
                                      input_ch=args.k_ref * 3 + (input_ch + args.k_ref * 3) * args.N_point_ray_enc
                                      if args.mm_emb else 3 * args.N_point_ray_enc,
                                      output_ch=2 + 3, skips=args.mmnetskips)
    grad_vars.append({'params': model_mmray.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lrate})

    # Input is average epipolar line and sample positions in 3d space, output is vol rendering (RGBs, Aplhas, and z_vals)
    model_refine = MinMaxRayS1Conv_Net(D=args.mmnetdepth, W=args.mmnetwidth,
                                  input_ch=args.k_ref * 3 + input_ch * args.N_point_ray_enc if args.mm_emb else
                                  3 +(3 + args.k_ref * 3) * args.N_samples,
                                  output_ch=args.N_samples * (1 + 1 + 3), skips=args.mmnetskips)
    grad_vars.append({'params': model_refine.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lrate})

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

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

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        model_mmray.load_state_dict(ckpt['mmr_network_fn_state_dict'])
        model_refine.load_state_dict(ckpt['refine_net_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

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
        'embed_fn': embed_fn if args.mm_emb else None,
        'randomize':True
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

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def compute_query_points_from_rays(
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        near_thresh: float,
        far_thresh: float,
        N_point_ray_enc,
        randomize = True
) -> (torch.Tensor, torch.Tensor):
    # Linear
    # depth_values = torch.linspace(near_thresh, far_thresh, N_point_ray_enc).to(ray_origins)
    # depth_values = depth_values.unsqueeze(0)

    # Exp.
    depth_values = torch.linspace(1.0, 0.0, steps=N_point_ray_enc).view(1, -1).type_as(ray_origins)
    depth_values = near_thresh * torch.exp(log(far_thresh / near_thresh) * (1-depth_values))

    if randomize is True:
        noise_shape = list(depth_values.shape)
        noise_ = (1/6) * torch.normal(0.0, 1.0, size=noise_shape).to(ray_origins)
        noise_ = noise_ * (far_thresh - near_thresh) / N_point_ray_enc
        depth_values = noise_ + depth_values
        depth_values[depth_values < 0] = 0
        depth_values, _ = torch.sort(depth_values, dim=-1)

    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
    return query_points, depth_values


def sraw2outputs(raw, z_vals, white_bkgd=False, rgb0=None):
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
    if rgb0 is None:
        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    else:
        rgb = torch.sigmoid(rgb0) + raw[..., :3]  # [N_rays, N_samples, 3]

    weights = F.softmax(raw[..., 3], 1)
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1) + 1e-6))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, rgb0=None,
                depth_densities=None, use_res_dense=True, pytest=False):
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
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    if rgb0 is None:
        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    else:
        rgb = torch.sigmoid(rgb0) + raw[..., :3]  # [N_rays, N_samples, 3]

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    if depth_densities is not None:
        if use_res_dense:
            # alpha = 1. - torch.exp(-(F.relu(raw[..., 3] + noise) + depth_densities) * dists)
            alpha = raw2alpha(raw[..., 3] + depth_densities + noise, dists)  # [N_rays, N_samples]
        else:
            alpha = raw2alpha(depth_densities + noise, dists)  # [N_rays, N_samples]
    else:
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1) + 1e-6))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch, or_ray_batch, coords, target_pose, ref_poses, ref_rgbs, H, W, K, k_ref,
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
                randomize=True,
                verbose=False,
                pytest=False):
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
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[0, 0, 0], bounds[0, 0, 1]  # [-1,1]

    or_rays_o, or_rays_d = or_ray_batch[:, 0:3], or_ray_batch[:, 3:6]  # [N_rays, 3] each
    or_bounds = torch.reshape(or_ray_batch[..., 6:8], [-1, 1, 2])
    or_near, or_far = or_bounds[0, 0, 0], or_bounds[0, 0, 1]  # [-1,1]

    pts, _ = compute_query_points_from_rays(or_rays_o, or_rays_d, or_near, or_far, N_point_ray_enc, randomize)

    pts = pts.view(-1, N_point_ray_enc * 3)
    if embed_fn is not None:
        pts = embed_fn(pts)
    mm_input = torch.permute(pts.view(1, H, W, -1), (0, 3, 1, 2))
    min_max_rays = min_max_ray_net(mm_input)
    min_max_rays = torch.permute(min_max_rays.view(2 + 3, H*W), (1, 0))

    # Depth levels
    min_rays = torch.sigmoid(min_max_rays[:, 0, None]) * (or_far - or_near) + or_near
    max_rays = torch.sigmoid(min_max_rays[:, 1, None]) * (or_far + 1e-6 - min_rays) + min_rays
    rgb0 = min_max_rays[:, 2::]

    # log depth quant
    epi_z_vals = torch.linspace(1.0, 0.0, steps=N_samples).view(1, -1).type_as(or_rays_o)
    epi_z_vals = min_rays * torch.exp(torch.log(max_rays / min_rays) * (1 - epi_z_vals))
    mean_epi_depth = torch.mean(epi_z_vals, 1, True)
    epi_pts = or_rays_o[..., None, :] + or_rays_d[..., None, :] * epi_z_vals[..., :, None]

    epi_pts = epi_pts.view(-1, N_samples * 3)
    if embed_fn is not None:
        epi_pts = embed_fn(epi_pts)

    with torch.no_grad():
        # k_ref = ref_rgbs.shape[0]
        # ref_nos = range(k_ref)
        ref_nos = random.choices(range(ref_rgbs.shape[0]), k=k_ref)
        if randomize:
            random.shuffle(ref_nos)
        ref_rgb = ref_rgbs[ref_nos].view(k_ref, H, W, 3)
        ref_rgb = torch.permute(ref_rgb, (0, 3, 1, 2))
        ref_rgb = torch.repeat_interleave(ref_rgb, repeats=N_samples, dim=0)
        ref_pose = ref_poses[ref_nos]
        ref_pose = torch.repeat_interleave(ref_pose, repeats=N_samples, dim=0)

        ro1, rd1 = torch.transpose(or_rays_o, 0, 1).unsqueeze(0), torch.transpose(or_rays_d, 0, 1).unsqueeze(0)  # 1, 3, H*W
        ro1, rd1 = ro1.repeat(N_samples * k_ref, 1, 1), rd1.repeat(N_samples * k_ref, 1, 1)
        K = K.unsqueeze(0).repeat(N_samples * k_ref, 1, 1)
        inv_K = torch.inverse(K)

    depths = epi_z_vals.view(1, H, W, N_samples).repeat(k_ref, 1, 1, 1) # k_ref, H, W, N_point_ray_enc
    depths = torch.permute(depths, (0, 3, 1, 2)).reshape(-1, H, W)  # k_ref * N_point_ray_enc, H, W

    warps = inverse_warp.inverse_warp_rod1_rt2(ref_rgb, depths, ro1, rd1, ref_pose, K, inv_K, padding_mode='zeros')
    invalid_warp = (torch.sum(warps.detach(), 1, True) == 0).type_as(warps)
    warps = warps * (1 - invalid_warp) - invalid_warp # make invalid regions -1
    warps_flat = warps.clone().view(1, k_ref, N_samples, 3, H, W)

    mm_input = torch.cat((warps_flat.view(1, k_ref*N_samples * 3, H, W),
                          torch.permute(epi_pts.view(1, H, W, N_samples * 3), (0, 3, 1, 2)),
                          torch.permute(rgb0.view(1, H, W, 3), (0, 3, 1, 2))
                          ), 1)
    min_max_rays = refine_net(mm_input)
    min_max_rays = torch.permute(min_max_rays.view(N_samples * (1 + 1 + 3), H*W), (1, 0))

    # Depth levels
    z_vals = torch.sigmoid(min_max_rays[:, 0:N_samples]) * (far - near) + near
    # Opacities
    z_densities = min_max_rays[:, N_samples:N_samples*2]
    # RGB priors
    z_rgbs = min_max_rays[:, N_samples*2::].view(rays_o.shape[0], N_samples, 3)
    mean_rgb = torch.mean(z_rgbs, dim=1)

    ret = {'rgb_map0': rgb0, 'rgb_map1': mean_rgb, 'depth_map':mean_epi_depth, 'mean_warp':mean_rgb}

    if k_ref > 0:
        warps = torch.permute(warps, (0, 2, 3, 1))
        for i in range(warps.shape[0]):
            ret[f'warp_{i}'] = warps[i].view(-1, 3)

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

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
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 1.
        far = 7.

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
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname,
                                       'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            render_outputs = render_path(render_poses, hwf, K, args.k_ref, args.chunk, render_kwargs_test, gt_imgs=images,
                                  savedir=testsavedir, render_factor=args.render_factor)
            rgbs = render_outputs[1]
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

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
    poses_train = torch.stack([poses[i] for i in i_train], 0)
    rays_rgb = torch.Tensor(rays_rgb).to(device)
    K_ten = torch.Tensor(K.copy()).to(device)

    N_iters = 2000000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = start + 1
    a_sch = 1000
    for i in trange(start, N_iters):
        # Random from one image
        img_i = random.choice(range(rays_rgb.shape[0]))

        # Reference data
        ref_poses = [poses_train[n_ref] for n_ref in range(rays_rgb.shape[0]) if n_ref != img_i]
        ref_poses = torch.stack(ref_poses, 0)
        ref_rgbs = [rays_rgb[n_ref, :, :, 2, :] for n_ref in range(rays_rgb.shape[0]) if n_ref != img_i]
        ref_rgbs = torch.stack(ref_rgbs, 0)

        # target data
        these_rays_rgb = rays_rgb[img_i]
        rays_o = these_rays_rgb[:, :, 0, :]
        rays_d = these_rays_rgb[:, :, 1, :]
        target = these_rays_rgb[:, :, 2, :]
        pose = poses_train[img_i, :3, :4]
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1)  # (H, W, 2)

        # Random crop
        if i < args.precrop_iters:
            dH = int(H // 2 * args.precrop_frac)
            dW = int(W // 2 * args.precrop_frac)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                    torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                ), -1)
            x1 = W // 2 - dW
            y1 = H // 2 - dH
            ref_K = K_ten.clone()
            ref_K[0, 2] = K_ten[0, 2] - x1
            ref_K[1, 2] = K_ten[1, 2] - y1
            if i == start:
                print(f"[Config] Center cropping of size "
                      f"{2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
        else:
            th, tw = args.rand_crop_size, args.rand_crop_size
            x1 = random.randint(0, W - tw)
            y1 = random.randint(0, H - th)
            coords = coords[y1: y1 + th, x1: x1 + tw, :]
            ref_K = K_ten.clone()
            ref_K[0, 2] = ref_K[0, 2] - x1
            ref_K[1, 2] = ref_K[1, 2] - y1

        th, tw, _ = coords.shape
        # print(f'coords shape {coords.shape}')
        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
        select_coords = coords.long()  # (N_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = torch.stack([rays_o, rays_d], 0)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        ref_rgbs = ref_rgbs[:, select_coords[:, 0], select_coords[:, 1]] # (N_ref, N_rand, 3)
        # print(f'reference RGBs shape {ref_rgb.shape}')

        #####  Core optimization loop  #####
        rgb0, rgb1, depth, mean_warp, extras = render(H, th, W, tw, K, ref_K, args.k_ref, target_pose=pose, ref_poses=ref_poses,
                                                     ref_rgbs=ref_rgbs, coords=coords.type_as(batch_rays)/W,
                                                     chunk=args.chunk, rays=batch_rays, verbose=i < 10,
                                                     retraw=True, **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb1, target_s)

        perc_loss = 0
        if args.a_p > 0 and i >= a_sch:
            # reshape for vgg (B, C, H, W)
            rgb_predicted_ = torch.permute(rgb1.view(th, tw, 3).clone(), (2, 0, 1)).unsqueeze(0)
            target_img_ = torch.permute(target_s.view(th, tw, 3).clone(), (2, 0, 1)).unsqueeze(0)
            perc_loss = args.a_p * perceptual_loss(vgg(rgb_predicted_), vgg(target_img_))

        loss = img_loss + perc_loss + img2mse(rgb0, target_s)# + img2mse(mean_warp, target_s)
        psnr = mse2psnr(img_loss)

        # if 'mean_rbg0' in extras and args.a_mmrgb > 0:
        #     img_loss0 = img2mse(extras['mean_rbg0'], target_s)
        #     loss = loss + img_loss0

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # Rest is logging
        if i % 500 == 0:
            if not os.path.exists(os.path.join(basedir, expname, 'debug')):
                os.makedirs(os.path.join(basedir, expname, 'debug'))

            rgb8 = to8b(target_s.view(th, tw, 3).detach().cpu().numpy())
            filename = os.path.join(basedir, expname, 'debug', 'check_target.png')
            imageio.imwrite(filename, rgb8)

            rgb8 = to8b(rgb1.view(th, tw, 3).detach().cpu().numpy())
            filename = os.path.join(basedir, expname, 'debug', 'check_rg1.png')
            imageio.imwrite(filename, rgb8)

            rgb8 = to8b(rgb0.view(th, tw, 3).detach().cpu().numpy())
            filename = os.path.join(basedir, expname, 'debug', 'check_rgb0.png')
            imageio.imwrite(filename, rgb8)

            rgb8 = to8b(mean_warp.view(th, tw, 3).detach().cpu().numpy())
            filename = os.path.join(basedir, expname, 'debug', 'check_mwarp.png')
            imageio.imwrite(filename, rgb8)

            np_depth = depth.view(th, tw).detach().cpu().numpy()
            rgb8 = to8b(np_depth / np.percentile(np_depth, 99))
            filename = os.path.join(basedir, expname, 'debug', 'check_depth.png')
            imageio.imwrite(filename, rgb8)

            if args.k_ref > 0:
                for num_ref in range(args.k_ref * args.N_samples):
                    rgb8 = to8b(extras[f'warp_{num_ref}'].view(th, tw, 3).detach().cpu().numpy())
                    filename = os.path.join(basedir, expname, 'debug', f'check_warp_{num_ref}.png')
                    imageio.imwrite(filename, rgb8)

        if i % args.i_weights == 0:
            print(f'New learning rate: {new_lrate}')
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if args.N_importance > 0:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'mmr_network_fn_state_dict': render_kwargs_train['min_max_ray_net'].state_dict(),
                    'refine_net_state_dict': render_kwargs_train['refine_net'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'mmr_network_fn_state_dict': render_kwargs_train['min_max_ray_net'].state_dict(),
                    'refine_net_state_dict': render_kwargs_train['refine_net'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                r_out = render_path(render_poses, hwf, K, args.k_ref, args.chunk, ref_poses=poses_train,
                                    ref_rgbs=rays_rgb[:, :, :, 2, :], render_kwargs=render_kwargs_test)
                rgbs0, rgbs1, depths, mean_warps = r_out[0], r_out[1], r_out[2], r_out[3]
            print('Done, saving', rgbs0.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb0.mp4', to8b(rgbs0), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'rgb1.mp4', to8b(rgbs1), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'mean_warps.mp4', to8b(mean_warps), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'depth.mp4', to8b(depths / np.percentile(depths, 99)), fps=30, quality=8)
            print(f'Mean depth {np.mean(depths)}')

        # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.k_ref, args.chunk, ref_poses=poses_train,
                            ref_rgbs=rays_rgb[:, :, :, 2, :], render_kwargs=render_kwargs_test,
                            gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
