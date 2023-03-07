import os
import sys

gpu_n = '1'
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
import inverse_warp
import lpips

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(3407)
DEBUG = False

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='configs/fern_finetune.txt',
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs_epi_RR/',
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
    parser.add_argument("--pretrain_path", type=str, default=None,
                        help='specific weights npy file to reload pretrain network')
    parser.add_argument("--pretrain_depth_path", type=str, default=None,
                        help='specific weights npy file to reload pretrain depth')
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
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, or_rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    batch_rays_nearest_id = kwargs['batch_rays_nearest_id']
    for i in range(0, rays_flat.shape[0], chunk):
        kwargs['batch_rays_nearest_id'] = batch_rays_nearest_id[i:i+chunk]
        ret = render_rays(rays_flat[i:i+chunk],or_rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True, 
                  near=0., far=1.,or_near=1., or_far=10.,
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

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    # Create original ray batch
    or_rays_o = torch.reshape(rays_o, [-1, 3]).float()
    or_rays_d = torch.reshape(rays_d, [-1, 3]).float()
    or_near, or_far = or_near * torch.ones_like(or_rays_d[..., :1]), or_far * torch.ones_like(or_rays_d[..., :1])
    or_rays = torch.cat([or_rays_o, or_rays_d, or_near, or_far], -1)
    if use_viewdirs:
        or_rays = torch.cat([or_rays, viewdirs], -1)

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, or_rays, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map0', 'rgb_map1', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

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
    ssims = []
    lpips_res = []
    img2mse_np = lambda x, y : np.mean((x - y) ** 2)
    mse2psnr_np = lambda x : -10. * np.log(x) / np.log([10.])
    lpips_vgg = lpips.LPIPS(net="vgg").cuda()
    lpips_vgg = lpips_vgg.eval()

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        # print(i, time.time() - t)
        # t = time.time()
        # ! compute nearest id
        poses_train = render_kwargs['poses_np']
        dists = np.sum(np.square(c2w.cpu().numpy()[:3,3] - poses_train[:,:3,3]), -1)
        
        nearest_pose = np.argsort(dists)[0:1+render_kwargs['num_neighbor']]
        rays_nearest_id = nearest_pose[None, None,:].repeat(H, axis=0).repeat(W, axis=1).reshape(-1,render_kwargs['num_neighbor'] + 1)
        rays_nearest_id = torch.Tensor(rays_nearest_id).to(device)
        render_kwargs['batch_rays_nearest_id'] = rays_nearest_id
        render_kwargs['target_pose'] = c2w

        rgb0, rgb1, depth_map, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs0.append(rgb0.cpu().numpy())
        rgbs1.append(rgb1.cpu().numpy())
        depths.append(depth_map.cpu().numpy())

        if gt_imgs is not None and render_factor == 0:
            p = mse2psnr(img2mse(rgb1, gt_imgs[i]))
            # p = -10. * np.log10(np.mean(np.square(rgb1.cpu().numpy() - gt_imgs[i].cpu().numpy())))
            # p = mse2psnr_np(img2mse_np(debug_rgb[i], gt_imgs[i].cpu().numpy()))
            psnrs.append(p)
            error = (rgb1 - gt_imgs[i])**2
            error = error.cpu().numpy()
            error = (error - np.min(error)) / (max(np.max(error) - np.min(error), 1e-8))
            error = cv2.applyColorMap((error * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

            # ssims
            ssim = img2ssim(rgb1.permute(2, 0, 1)[None], (gt_imgs[i]).permute(2, 0, 1)[None].cuda())
            ssims.append(ssim.cpu().numpy())

            # lpips
            scaled_gt = (gt_imgs[i]).permute(2, 0, 1)[None] * 2.0 - 1.0
            scaled_pred = rgb1.permute(2, 0, 1)[None] * 2.0 - 1.0
            lpips_val = lpips_vgg(scaled_gt.cuda(), scaled_pred.cuda())
            lpips_res.append(lpips_val.detach().squeeze().cpu().numpy())

        if savedir is not None:
            rgb8 = to8b(rgbs1[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            rgb8 = to8b(gt_imgs[i].cpu().numpy())
            filename = os.path.join(savedir, 'gt_{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            filename = os.path.join(savedir, 'err{:03d}.png'.format(i))
            imageio.imwrite(filename, error)

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
    # if len(psnrs) > 0:
    #     psnrs = np.array(psnrs)
    #     print(f'Mean Test PSNR {psnrs.mean()}')
    print('LPIPS', np.array(lpips_res).mean())
    print('SSIMS', np.array(ssims).mean())
    return rgbs0, rgbs1, depths, depths


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    embed_rays = Pluecker()
    pretrain_ckpt = torch.load(args.pretrain_path)

    input_ch_views = 0
    embeddirs_fn = None
    grad_vars = []
    grad_vars_nerf = []

    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    model.to(device)

    model_fine = None
    # model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
    #                     input_ch=input_ch, output_ch=output_ch, skips=skips,
    #                     input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    model_fine = DoNeRF(D=args.netdepth, W=args.netwidth,
                    n_in=input_ch + input_ch_views, n_out=output_ch, skip='auto')

    model_fine.to(device)
    model_fine.load_state_dict(pretrain_ckpt['network_fn_state_dict'])
    grad_vars_nerf.append({'params': model_fine.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lrate})
    grad_vars.append({'params': model_fine.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lrate})

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    model_mmray = MinMaxRay_Net(D=args.mmnetdepth, W=args.mmnetwidth,
                                input_ch=2 + input_ch * args.N_point_ray_enc if args.mm_emb else
                                6 * args.N_point_ray_enc,
                                output_ch=3 * args.N_samples + 3, skips=args.mmnetskips)
    model_mmray.load_state_dict(pretrain_ckpt['mmr_network_fn_state_dict'])
    grad_vars.append({'params': model_mmray.parameters(),
                     'weight_decay': args.weight_decay, 'lr': args.lrate})
    
    
    model_refine = MinMaxRay_Net(D=args.mmnetdepth, W=args.mmnetwidth,
                                input_ch=input_ch * args.N_samples if args.mm_emb else
                                6 * (0+args.N_samples) + 3 * args.num_neighbor * args.N_samples,
                                output_ch=4 * args.N_samples + 3, skips=args.mmnetskips)
    model_refine.load_state_dict(pretrain_ckpt['refine_net_state_dict'])
    grad_vars.append({'params': model_refine.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lrate})

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    optimizer_nerf = torch.optim.Adam(params=grad_vars_nerf, lr=args.lrate, betas=(0.9, 0.999))

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

        # # if not (args.ft_path is not None and args.ft_path!='None'):
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        optimizer_nerf.load_state_dict(ckpt['optimizer_nerf_state_dict'])

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
        'embed_fn': embed_fn,
        'embeddirs_fn': embeddirs_fn,
        'embed_rays':embed_rays,
        'randomize': True,
        'num_neighbor': args.num_neighbor
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

    # # Exp.
    # depth_values = torch.linspace(1.0, 0.0, steps=N_point_ray_enc).view(1, -1).type_as(ray_origins)
    # depth_values = near_thresh * torch.exp(torch.log(far_thresh / near_thresh) * (1-depth_values))

    # if randomize is True:
    #     noise_shape = list(depth_values.shape)
    #     noise_ = (1 / 6) * torch.normal(0.0, 1.0,
    #                                     size=noise_shape).to(ray_origins)
    #     noise_ = noise_ * (far_thresh - near_thresh) / N_point_ray_enc
    #     depth_values = noise_ + depth_values
    #     depth_values[depth_values < near_thresh] = near_thresh
    #     depth_values[depth_values > far_thresh] = far_thresh
    #     depth_values, _ = torch.sort(depth_values, dim=-1)

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
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std
        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # if mm_density_add is not None:
    if False:
        alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
        if True:
            alpha = alpha*torch.relu(mm_density_mul)
            # alpha = alpha*torch.sigmoid(mm_density_mul)
    else:
        alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

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

    with torch.no_grad():
        pts, _ = compute_query_points_from_rays(
            rays_o, rays_d, 0., 1., N_point_ray_enc, randomize=False)  # ! this is ndc space
    
    # 1. mm take point encoding and predict N samples points
    plucker_pts = kwargs['embed_rays'](pts, rays_d[:,None,:].repeat(1,N_point_ray_enc,1)) # nump_pts + origin
    plucker_pts = plucker_pts.view(-1, (N_point_ray_enc)*6)

    # pts = pts.view(-1, N_point_ray_enc * 3)
    min_max_rays = min_max_ray_net(plucker_pts)
    mm_rgb = torch.sigmoid(min_max_rays[:, 3*N_samples:])
    mm_density_add = min_max_rays[:, N_samples:2*N_samples]
    mm_density_mul = min_max_rays[:, 2*N_samples:3*N_samples]

    depth_values = torch.sigmoid(min_max_rays[:, :N_samples]) * (far - near) + near  # B, Nsamples, H, W
    sort_out = torch.sort(depth_values, dim=-1)
    depth_values = sort_out[0]  # ! depth values are sorted, ndc space
    mm_density_add = torch.gather(mm_density_add, dim =1, index = sort_out[1])
    mm_density_mul = torch.gather(mm_density_mul, dim =1, index = sort_out[1])

    depth_values_3d = 1/(1-depth_values - 1e-5)  #! convert ndc zval to 3d zval

    or_rays_o, or_rays_d = or_ray_batch[:, 0:3], or_ray_batch[:, 3:6]  # [N_rays, 3] each
    or_bounds = torch.reshape(or_ray_batch[..., 6:8], [-1, 1, 2])
    or_near, or_far = or_bounds[0, 0, 0], or_bounds[0, 0, 1]  # [-1,1]

    with torch.no_grad():
        num_pts = N_samples
        num_neighbor = kwargs['num_neighbor']
        k_ref = kwargs['images'].shape[0]
        ref_rgbs = kwargs['images']
        ref_K = kwargs['ref_K']
        ref_poses = kwargs['poses']
        
        if randomize:
            current_id = kwargs['batch_rays_nearest_id'][:,0].long()
            target_pose = ref_poses[current_id]
        else:
            target_pose = kwargs['target_pose'][None].repeat(N_rays,1,1)

        rel_cam_dist = torch.sum((target_pose[:,None,:, 3] - ref_poses[:, :, 3]) ** 2, 2) ** (1 / 2)
        _, rel_cam_idx = torch.sort(rel_cam_dist.detach(), dim=1)

        if randomize:
            ref_nos = rel_cam_idx[:,1:]
            # Random but keep order, ! remove the first pose
            order_idx = torch.from_numpy(np.array(sorted(random.sample(range(ref_nos.shape[1]), num_neighbor)))).to(device)
            ref_nos = torch.gather(ref_nos, dim=1, index=order_idx[None].repeat(N_rays,1).long())
        else:
            # Nearest, testing do not remove first pose
            ref_nos = rel_cam_idx[:, 0:num_neighbor]

        ref_rgb = (ref_rgbs.permute(0, 3, 1, 2))
        ref_rgb = torch.repeat_interleave(ref_rgb, repeats=num_pts, dim=0)
        ref_pose = torch.repeat_interleave(ref_poses, repeats=num_pts, dim=0)

        ro1, rd1 = torch.transpose(or_rays_o, 0, 1).unsqueeze(0), torch.transpose(or_rays_d, 0, 1).unsqueeze(0)  # 1, 3, H*W
        ro1, rd1 = ro1.repeat(num_pts * k_ref, 1, 1), rd1.repeat(num_pts * k_ref, 1, 1)
        ref_K = ref_K.unsqueeze(0).repeat(num_pts * k_ref, 1, 1)
        inv_K = torch.inverse(ref_K)

        # Should we enable grad????
        # ! warp H and W will be 1, N_rays
        warp_H = 1
        warp_W = N_rays
        depths = depth_values_3d[None,None,:,:].repeat(k_ref,1,1,1) # k_ref, H, W, N_point_ray_enc
        depths = (depths.permute(0, 3, 1, 2)).reshape(-1, warp_H, warp_W)  # k_ref * N_point_ray_enc, H, W

        warps, _ = inverse_warp.inverse_warp_rod1_rt2_coords(ref_rgb, depths, ro1, rd1, ref_pose, ref_K, inv_K, padding_mode='zeros')
        warps_flat = warps.clone().view(1, k_ref, num_pts, 3, warp_H, warp_W)
        rays_valid_id = ref_nos.transpose(0, 1)[None,:,None,None,None].repeat(1, 1, num_pts,3,1,1)
        valid_warps_flat = torch.gather(warps_flat, dim=1, index = rays_valid_id.long()) # 1, validid, N samples, 3, 1, N rays

        valid_warp = (torch.sum(valid_warps_flat.detach(), 3, True) > 0).type_as(warps).repeat(1, 1, 1, 3, 1, 1)
        mean_sample_warp = torch.sum(valid_warp * valid_warps_flat.detach(), 1, True) / (torch.sum(valid_warp, 1, True) + 1e-6)
        valid_warps_flat = valid_warps_flat * valid_warp + mean_sample_warp * (1 - valid_warp)

        epi_features = (valid_warps_flat.view(num_pts*num_neighbor, 3, warp_H*warp_W).permute(2,0,1)).reshape(-1, 3*num_pts*num_neighbor) # N rays, 3*num_pts
    # Should we enable grad of ^????

    epi_pts = rays_o[..., None, :] + rays_d[..., None, :] * depth_values[..., :, None]

    plucker_embed = kwargs['embed_rays'](epi_pts, rays_d[:, None, :].repeat(1, num_pts, 1)) # nump_pts + origin
    plucker_embed = plucker_embed.view(-1, num_pts, 6).view(N_rays, -1)

    refine_input = torch.cat([plucker_embed, epi_features], dim =1)
    refine_output = refine_net(refine_input)
    refine_depth_values = torch.sigmoid(refine_output[:,:N_samples])
    refine_rgb = torch.sigmoid(refine_output[:, 4*N_samples:])
    points_offset = torch.tanh(refine_output[:, N_samples:4*N_samples]).view(N_rays, N_samples,3)

    mids = .5 * (depth_values[...,1:] + depth_values[...,:-1])
    upper = torch.cat([mids, 0.5*(far+depth_values[...,-1:])], -1) # upper cat far
    lower = torch.cat([0.5*(near+depth_values[...,:1]), mids], -1) # lower cat near
    refine_depth_values = lower + (upper - lower) * refine_depth_values

    # ! new nerf randomize exploration
    train_nerf = kwargs.get('train_nerf', False)
    if train_nerf and randomize:
        epi_z_vals = refine_depth_values
        noise_shape = list(epi_z_vals.shape)
        noise_ = (1 / 5) * torch.normal(0.0, 1.0, size=noise_shape).type_as(epi_z_vals)
        noise_ = torch.abs(noise_)
        max_noise = 1 - 2e-6
        noise_[noise_ >= max_noise] = max_noise
        if random.random() > 0.5:
            epi_z_vals_diff = torch.abs(
                epi_z_vals - torch.cat((epi_z_vals[:, 1:], far * torch.ones(N_rays, 1).type_as(epi_z_vals)), 1))
            noise_ = noise_ * epi_z_vals_diff
        else:
            epi_z_vals_diff = torch.abs(
                epi_z_vals - torch.cat((near * torch.ones(N_rays, 1), epi_z_vals[:, 0:-1].type_as(epi_z_vals)), 1))
            noise_ = -noise_ * epi_z_vals_diff
        epi_z_vals = epi_z_vals + noise_
    else:
        epi_z_vals = refine_depth_values

    query_points_nerf = rays_o[..., None, :] + rays_d[..., None,
                                                    :] * epi_z_vals[..., :, None]  # ! this is ndc space
    query_points_nerf = query_points_nerf + (1e-2) * points_offset
    raw = network_query_fn(query_points_nerf, viewdirs, network_fine)
    iter = kwargs.get('iter', 1e6)
    if not train_nerf:
        raw_noise_std = 0
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, epi_z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                 pytest=pytest, mm_density_add=mm_density_add,
                                                                 mm_density_mul=mm_density_mul, iter=iter)
    ret = {'rgb_map0': refine_rgb, 'rgb_map1': rgb_map, 'depth_map': depth_map, 'mm_rgb': mm_rgb}
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
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
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
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, optimizer_nerf = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # # Short circuit if only rendering out from trained model
    # if args.render_only:
    #     print('RENDER ONLY')
    #     with torch.no_grad():
    #         if args.render_test:
    #             # render_test switches to test poses
    #             images = torch.Tensor(images[i_test]).to(device)
    #         else:
    #             # Default is smoother render_poses path
    #             images = None

    #         testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
    #         os.makedirs(testsavedir, exist_ok=True)
    #         print('test poses shape', render_poses.shape)

    #         render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
    #         print('Done rendering', testsavedir)
    #         # imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

    #         return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb + depth + img_id, 3]
        rays_rgb = rays_rgb.astype(np.float32)

        # ! compute nearest id
        poses_train = poses[i_train]
        render_kwargs_test['poses_np'] = poses_train
        rays_nearest_id = []
        for pose_id in range(poses_train.shape[0]):
            dists = np.sum(np.square(poses_train[pose_id][:3,3] - poses_train[:,:3,3]), -1)
            nearest_pose = np.argsort(dists)[0:1+args.num_neighbor] # 4 nereast neighbor
            rays_nearest_id.append(nearest_pose)
        rays_nearest_id = np.stack(rays_nearest_id, axis = 0)
        rays_nearest_id = rays_nearest_id[:, None, None,:].repeat(H, axis=1).repeat(W, axis=2).reshape(-1,args.num_neighbor + 1)
            

        print('shuffle rays')
        rand_idx = np.random.permutation(rays_rgb.shape[0])
        rays_rgb = rays_rgb[rand_idx]
        rays_nearest_id = rays_nearest_id[rand_idx]

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)
        rays_nearest_id = torch.Tensor(rays_nearest_id).to(device)


    N_iters = 500000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # update train val id
    K_ten = torch.Tensor(K.copy()).to(device)
    render_kwargs_train['i_train'] = i_train
    render_kwargs_test['i_train'] = i_train
    render_kwargs_train['images'] = images[i_train]
    render_kwargs_test['images'] = images[i_train]
    render_kwargs_train['poses'] = poses[i_train]
    render_kwargs_test['poses'] = poses[i_train]
    render_kwargs_train['ref_K'] = K_ten
    render_kwargs_test['ref_K'] = K_ten

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()
        # Random over all images
        batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_s = batch[:2], batch[2]
        batch_rays_nearest_id = rays_nearest_id[i_batch:i_batch+N_rand]

        i_batch += N_rand
        if i_batch >= rays_rgb.shape[0]:
            rand_idx = np.random.permutation(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
            rays_nearest_id = rays_nearest_id[rand_idx]
            i_batch = 0

        #####  Core optimization loop  #####
        render_kwargs_train['iter'] = i
        render_kwargs_test['iter'] = i
        render_kwargs_test['train_nerf'] = False
        render_kwargs_train['batch_rays_nearest_id'] = batch_rays_nearest_id
        render_kwargs_train['train_nerf'] = True

        # ! train nerf
        # rgb0, rgb1, depth_map, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
        #                                         verbose=i < 10, retraw=True,
        #                                         **render_kwargs_train)
        #
        # optimizer_nerf.zero_grad()
        # img_loss = img2mse(rgb1, target_s)
        # loss = img_loss
        # # psnr = mse2psnr(img_loss)
        # loss.backward()
        # optimizer_nerf.step()

        # ! train sampling net
        render_kwargs_train['train_nerf'] = True
        rgb0, rgb1, depth_map, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb1, target_s)
        loss = img_loss
        if args.a_mmrgb > 0:
            rgb0_loss = img2mse(rgb0, target_s)
            mm_rgb_loss = img2mse(extras['mm_rgb'], target_s)
            loss = loss + args.a_mmrgb * (rgb0_loss + mm_rgb_loss)

        psnr = mse2psnr(img_loss)
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        for param_group in optimizer_nerf.param_groups:
            param_group['lr'] = new_lrate
        # ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            # print(f'New learning rate: {new_lrate}')
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if render_kwargs_train['network_fine'] is not None:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'mmr_network_fn_state_dict': render_kwargs_train['min_max_ray_net'].state_dict(),
                    'refine_net_state_dict': render_kwargs_train['refine_net'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_nerf_state_dict': optimizer_nerf.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
            else:
                torch.save({
                    'global_step': global_step,
                    'mmr_network_fn_state_dict': render_kwargs_train['min_max_ray_net'].state_dict(),
                    'refine_net_state_dict': render_kwargs_train['refine_net'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_nerf_state_dict': optimizer_nerf.state_dict(),
                }, path)
                print('Saved checkpoints at', path)

        # if (i % args.i_video == 0 and i > 0) or (args.render_only):
        #     # Turn on testing mode
        #     with torch.no_grad():
        #         r_out = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
        #         rgbs0, rgbs1, depths, depths0 = r_out[0], r_out[1], r_out[2], r_out[3]
        #     print('Done, saving', rgbs0.shape)
        #     if args.render_only:
        #         testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
        #             'test' if args.render_test else 'path', start))
        #         os.makedirs(testsavedir, exist_ok=True)
        #         moviebase = os.path.join(
        #             testsavedir, '{}_spiral_{:06d}_'.format(expname, i))
        #     else:
        #         moviebase = os.path.join(
        #             basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
        #     imageio.mimwrite(moviebase + 'rgb0.mp4',
        #                      to8b(rgbs0), fps=30, quality=8)
        #     imageio.mimwrite(moviebase + 'rgb1.mp4',
        #                      to8b(rgbs1), fps=30, quality=8)
        #     # imageio.mimwrite(moviebase + 'mean_warps.mp4', to8b(mean_warps), fps=30, quality=8)
        #     imageio.mimwrite(moviebase + 'depth.mp4', to8b(depths /
        #                      np.percentile(depths, 99)), fps=30, quality=8)
        #     imageio.mimwrite(moviebase + 'depth0.mp4', to8b(depths0 /
        #                      np.percentile(depths0, 99)), fps=30, quality=8)
        #     # print(f'Mean depth {np.mean(depths)}')
        #     if args.render_only:
        #         return

        if (i % args.i_testset == 0 and i > 0) or (args.render_test):
            if args.render_test:
                testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
                    'test' if args.render_test else 'path', start))
                os.makedirs(testsavedir, exist_ok=True)
            else:
                testsavedir = os.path.join(
                    basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')
            if (args.render_test):
                return


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()