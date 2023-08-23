import os
import sys
import threading

gpu_n = '7'
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
from skimage.metrics import structural_similarity
from ssim_torch import ssim as r2l_ssim_func

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data, load_llff_cimgs, load_llff_data_infer
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
    parser.add_argument("--basedir", type=str, default='./logs_infer/',
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
    parser.add_argument("--mask_path", type=str, default=None,
                        help='specific weights npy file to reload for mask network')
    parser.add_argument("--avr_path", type=str, default=None,
                        help='specific weights npy file to reload for avr network')

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
    parser.add_argument("--compress_rate", type=int, default=50,
                        help='compress_rate')

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
    # all_ret = {}
    # batch_rays_nearest_id = kwargs['batch_rays_nearest_id']
    # for i in range(0, rays_flat.shape[0], chunk):
    #     kwargs['batch_rays_nearest_id'] = batch_rays_nearest_id[i:i+chunk]
    #     ret = render_rays(rays_flat[i:i+chunk],or_rays_flat[i:i+chunk], **kwargs)
    #     for k in ret:
    #         if k not in all_ret:
    #             all_ret[k] = []
    #         all_ret[k].append(ret[k])

    # all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    # return all_ret

    ret = render_rays(rays_flat,or_rays_flat, **kwargs)
    return ret


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
    depths0 = []
    depths00 = []
    depth_diffs = []
    psnrs = []
    ssims = []
    nex_ssims = []
    r2l_ssims = []
    lpips_res = []
    lpips_res_alex = []


    t1, t2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
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

        t1.record()
        rgb0, rgb1, depth_map, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        t2.record()
        torch.cuda.synchronize(device=device)
        print('Render path time:', t1.elapsed_time(t2))

        rgbs0.append(rgb0.cpu().numpy())
        rgbs1.append(rgb1.cpu().numpy())
        depths.append(depth_map.cpu().numpy())
        # depths0.append(extras['z_vals'].cpu().numpy())
        # depths00.append(extras['z_vals0'].cpu().numpy())
        # depth_diffs.append(torch.abs(extras['z_vals0'] - extras['z_vals']).cpu().numpy())

        if gt_imgs is not None and render_factor == 0:
            rgb1_cc = color_correct(rgb1.cpu().numpy().astype(np.float64), gt_imgs[i].cpu().numpy())
            rgb1_ = rgb1.clone()
            rgb1 = (np.asarray(rgb1_cc))
            breakpoint()
            p = mse2psnr_np(img2mse_np(rgb1_cc, gt_imgs[i].cpu().numpy()))
            psnrs.append(p)

            # error = (rgb1 - gt_imgs[i])**2
            # error = error.cpu().numpy()
            # error = (error - np.min(error)) / (max(np.max(error) - np.min(error), 1e-8))
            # error = cv2.applyColorMap((error * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

            # ssims
            ssim = img2ssim(rgb1, (gt_imgs[i]).cpu().numpy())
            # r2l_ssim = r2l_ssim_func(
            #     rgb1.cpu()[None].permute(0, 3, 1, 2), (gt_imgs[i]).cpu()[None].permute(0, 3, 1, 2))
            nex_ssim = structural_similarity(rgb1, (gt_imgs[i]).cpu(
            ).numpy(), win_size=11, multichannel=True, gaussian_weights=True)
            ssims.append(ssim)
            nex_ssims.append(nex_ssim)
            # r2l_ssims.append(r2l_ssim)

            # lpips
            lpips_val = rgb_lpips((gt_imgs[i]).cpu().numpy(), rgb1, 'vgg', device)
            lpips_res.append(lpips_val)

            lpips_val = rgb_lpips(
                (gt_imgs[i]).cpu().numpy(), rgb1_.cpu().numpy(), 'alex', device)
            lpips_res_alex.append(lpips_val)


        if savedir is not None:
            rgb8 = to8b(rgbs1[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            # rgb8 = to8b(gt_imgs[i].cpu().numpy())
            # filename = os.path.join(savedir, 'gt_{:03d}.png'.format(i))
            # imageio.imwrite(filename, rgb8)

            # filename = os.path.join(savedir, 'err{:03d}.png'.format(i))
            # imageio.imwrite(filename, error)

            # rgb8 = to8b(depths[-1]/np.max(depths[-1]))
            # filename = os.path.join(savedir, 'depth_{:03d}.png'.format(i))
            # imageio.imwrite(filename, rgb8)

            # rgb8 = to8b(depths0[-1]/np.max(depths0[-1]))
            # filename = os.path.join(savedir, 'depth0_{:03d}.png'.format(i))
            # imageio.imwrite(filename, rgb8)

            # rgb8 = to8b(depths00[-1]/np.max(depths00[-1]))
            # filename = os.path.join(savedir, 'depth00_{:03d}.png'.format(i))
            # imageio.imwrite(filename, rgb8)

            # rgb8 = cv2.applyColorMap(((depth_diffs[-1] - np.min(depth_diffs[-1]))/(np.max(depth_diffs[-1]) - np.min(depth_diffs[-1])) * 255).astype(np.uint8), cv2.COLORMAP_JET)
            # filename = os.path.join(savedir, 'depthdiff_{:03d}.png'.format(i))
            # imageio.imwrite(filename, rgb8)

    rgbs0 = np.stack(rgbs0, 0)
    rgbs1 = np.stack(rgbs1, 0)
    depths = np.stack(depths, 0)
    if len(psnrs) > 0:
        mean_psnr = 0
        for this_psnr in psnrs:
            mean_psnr = mean_psnr + this_psnr / len(psnrs)
        print(psnrs)
        print(f'Mean Test PSNR {round(mean_psnr, 2)}')
        
    print('LPIPS vgg', round(np.array(lpips_res).mean(),3))
    print('LPIPS alex', round(np.array(lpips_res_alex).mean(),3))
    print('SSIMS', round(np.array(ssims).mean(),3))
    print('NEX SSIMS', round(np.array(nex_ssims).mean(),3))
    # print('R2l SSIMS', round(np.array(r2l_ssims).mean(),3))
    return rgbs0, rgbs1, depths, depths


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
    model = DoNeRF(D=args.netdepth, W=args.netwidth,
                    n_in=input_ch + input_ch_views, n_out=output_ch, skip='auto')
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
                                input_ch=6 * avr_N_samples + 3 * args.num_neighbor * avr_N_samples +
                                         3 * avr_N_samples + 3 + (3 * 4) * args.num_neighbor,
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

        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    compress_dict = {
        0: 0.0,
        50: 0.5,
        33: 1/3.0,
        66: 2/3.0,
        100: 1.0
    }

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
        'num_neighbor': args.num_neighbor,
        'mask_net': model_mask,
        'avr_min_max_ray_net': model_avr_mmray,
        'avr_refine_net': model_avr_refine,
        'compress_rate': compress_dict[args.compress_rate]
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

    return render_kwargs_train, render_kwargs_test, start, grad_vars, None, None


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

    if mm_density_add is not None:
        alpha = raw2alpha(raw[...,3] + noise + mm_density_add, dists)  # [N_rays, N_samples]
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
    or_rays_o, or_rays_d = or_ray_batch[:, 0:3], or_ray_batch[:, 3:6]  # [N_rays, 3] each

    with torch.no_grad():
        pts, _ = compute_query_points_from_rays(
            rays_o, rays_d, 0., 1., N_point_ray_enc, randomize=False)  # ! this is ndc space
    
    # 1. mm take point encoding and predict N samples points
    plucker_pts = kwargs['embed_rays'](pts, rays_d[:,None,:].repeat(1,N_point_ray_enc,1)) # nump_pts + origin
    plucker_pts = plucker_pts.view(-1, (N_point_ray_enc)*6)

    # mask mmray
    mm_mask = kwargs['mask_net'](plucker_pts)
    mm_mask = torch.sigmoid(mm_mask)
    compress_rate = kwargs['compress_rate']
    _, avr_index = torch.topk(mm_mask, k = int(N_rays*compress_rate), dim = 0)
    avr_index = avr_index.squeeze(-1)

    _, shader_index = torch.topk(mm_mask, k = int(N_rays*(1-compress_rate)), dim = 0, largest = False)
    shader_index = shader_index.squeeze(-1)

    # SHADER PIPELINE
    shader_N_rays = int(N_rays*(1-compress_rate))
    shader_plucker_pts = plucker_pts[shader_index]
    shader_rays_o, shader_rays_d = rays_o[shader_index], rays_d[shader_index] 
    shader_or_rays_o, shader_or_rays_d = or_rays_o[shader_index], or_rays_d[shader_index] 
    shader_viewdirs = viewdirs[shader_index]
    shader_bounds = bounds[shader_index]
    shader_near, shader_far = near[shader_index], far[shader_index]

    # AVR PIPELINE
    avr_N_samples = 8
    avr_N_rays = int(N_rays*compress_rate)
    avr_plucker_pts = plucker_pts[avr_index]
    avr_rays_o, avr_rays_d = rays_o[avr_index], rays_d[avr_index] 
    avr_or_rays_o, avr_or_rays_d = or_rays_o[avr_index], or_rays_d[avr_index] 
    avr_viewdirs = viewdirs[avr_index]
    avr_bounds = bounds[avr_index]
    avr_near, avr_far = near[avr_index], far[avr_index]

    rgb_map = torch.zeros((N_rays, 3))

    if shader_N_rays > 0:
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
                                    **kwargs)
        rgb_map[shader_index] = shader_rgb_map

    if avr_N_rays > 0:
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
                                **kwargs
                                )
        rgb_map[avr_index] = avr_rgb_map   

    
    ret = {'rgb_map0': rgb_map, 'rgb_map1': rgb_map, 'depth_map': rgb_map}
    return ret

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
                **kwargs
                  ):
    F_theta0_out = kwargs['avr_min_max_ray_net'](avr_plucker_pts)
    depth_values = torch.sigmoid(F_theta0_out[:, :N_samples]) * (avr_far - avr_near) + avr_near
    weights_f0 = F_theta0_out[:, N_samples:2 * N_samples]
    mm_rgb = F_theta0_out[:, 2 * N_samples:2 * N_samples + 3]
    sort_out = torch.sort(depth_values, dim=-1)
    depth_values = sort_out[0] # ! depth values are sorted, ndc space

    # Get epipolar colors
    depth_values_3d = 1 / (1 - depth_values - 1e-6)  # ! convert ndc zval to 3d zval
    with torch.no_grad():
        num_pts = N_samples
        num_neighbor = kwargs['num_neighbor']
        k_ref = kwargs['images'].shape[0]
        ref_rgbs = kwargs['images']
        ref_K = kwargs['ref_K']
        ref_poses = kwargs['poses']

        target_pose = kwargs['target_pose'][None].repeat(avr_N_rays, 1, 1)

        rel_cam_dist = torch.sum((target_pose[:, None, :, 3] - ref_poses[:, :, 3]) ** 2, 2) ** (1 / 2)
        _, rel_cam_idx = torch.sort(rel_cam_dist.detach(), dim=1)


        ref_nos = rel_cam_idx[:, 0:num_neighbor]

        ref_rgb = (ref_rgbs.permute(0, 3, 1, 2))
        ref_rgb = torch.repeat_interleave(ref_rgb, repeats=num_pts, dim=0)
        ref_pose = torch.repeat_interleave(ref_poses, repeats=num_pts, dim=0)

        ro1, rd1 = torch.transpose(avr_or_rays_o, 0, 1).unsqueeze(0), torch.transpose(avr_or_rays_d, 0, 1).unsqueeze(
            0)  # 1, 3, H*W
        ro1, rd1 = ro1.repeat(num_pts * k_ref, 1, 1), rd1.repeat(num_pts * k_ref, 1, 1)
        ref_K = ref_K.unsqueeze(0).repeat(num_pts * k_ref, 1, 1)
        inv_K = torch.inverse(ref_K)

    # This should be with grad enabled??????
    warp_H = 1
    warp_W = avr_N_rays
    depths = depth_values_3d[None, None, :, :].repeat(k_ref, 1, 1, 1)  # k_ref, H, W, N_point_ray_enc
    depths = (depths.permute(0, 3, 1, 2)).reshape(-1, warp_H, warp_W)  # k_ref * N_point_ray_enc, H, W
    warps, _ = inverse_warp.inverse_warp_rod1_rt2_coords(ref_rgb, depths, ro1, rd1, ref_pose, ref_K, inv_K,
                                                            padding_mode='border')
    warps_flat = warps.clone().view(1, k_ref, num_pts, 3, warp_H, warp_W)
    rays_valid_id = ref_nos.transpose(0, 1)[None, :, None, None, None].repeat(1, 1, num_pts, 3, 1, 1)
    warps_flat = torch.gather(warps_flat, dim=1, index=rays_valid_id.long())  # 1, validid, N samples, 3, 1, N rays

    rays_valid_id_pose = ref_nos[:, :, None].expand(-1, -1, 3*4)
    ref_pose_in = ref_poses[None].view(1, -1, 3*4).expand(avr_N_rays, -1, -1)
    ref_pose_in = torch.gather(ref_pose_in, dim=1, index=rays_valid_id_pose.long())

    epi_features = (warps_flat.view(num_neighbor, num_pts, 3, warp_H * warp_W).permute(3, 1, 0, 2)). \
        reshape(-1, num_pts, num_neighbor * 3)
    epi_var = torch.abs(torch.var(epi_features.view(avr_N_rays, N_samples, num_neighbor, 3), dim=2))

    weights = F.softmax(weights_f0, dim=1)
    rgb0 = torch.sum(weights.view(-1, N_samples, 1, 1) *
                       epi_features.view(avr_N_rays, N_samples, num_neighbor, 3), dim=1)
    rgb0 = torch.mean(rgb0, dim=1) + mm_rgb
    mean_depth0 = torch.sum(depth_values * weights, dim=-1, keepdim=True)

    epi_pts = avr_rays_o[..., None, :] + avr_rays_d[..., None, :] * depth_values[..., :, None]
    plucker_embed = kwargs['embed_rays'](epi_pts, avr_rays_d[:, None, :].repeat(1, num_pts, 1)) # nump_pts + origin
    plucker_embed = plucker_embed.view(-1, num_pts, 6)
    net_input = torch.cat((plucker_embed.view(avr_N_rays, -1), epi_features.view(avr_N_rays, -1), epi_var.view(avr_N_rays, -1),
                           avr_viewdirs.view(avr_N_rays, -1), ref_pose_in.view(avr_N_rays, -1)), -1)

    F_theta1_output = kwargs['avr_refine_net'](net_input)
    weights0 = F_theta1_output[:, 0:num_neighbor * N_samples]
    combine = torch.sigmoid(F_theta1_output[:, num_neighbor * N_samples:num_neighbor * N_samples+num_neighbor])
    combine = combine / torch.sum(combine, dim=1, keepdim=True)
    res_epi = F_theta1_output[:, num_neighbor * N_samples + num_neighbor:
              num_neighbor * N_samples + num_neighbor + 3 * num_neighbor * N_samples]

    # Approximated volumetric rendering (AVR)
    weights = F.softmax(weights0.view(avr_N_rays, N_samples, num_neighbor), dim=1)
    refine_rgbs = torch.sum(weights.view(avr_N_rays, N_samples, num_neighbor, 1) *
                       epi_features.view(avr_N_rays, N_samples, num_neighbor, 3), dim=1)
    mean_depth = torch.sum(depth_values[:, :, None] * weights.view(avr_N_rays, N_samples, num_neighbor), dim=-1)
    mean_depth = torch.mean(mean_depth, dim=1, keepdim=True)

    avr_rgb_map = torch.sum(combine[..., None] * refine_rgbs, 1)
    return avr_rgb_map
    
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
                  **kwargs
                  ):
    min_max_rays = min_max_ray_net(shader_plucker_pts)
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
        ref_rgbs = kwargs['images']
        ref_K = kwargs['ref_K']
        ref_poses = kwargs['poses']
        
        target_pose = kwargs['target_pose'][None].repeat(shader_N_rays,1,1)

        rel_cam_dist = torch.sum((target_pose[:,None,:, 3] - ref_poses[:, :, 3]) ** 2, 2) ** (1 / 2)
        _, rel_cam_idx = torch.sort(rel_cam_dist.detach(), dim=1)

        ref_nos = rel_cam_idx[:, 0:num_neighbor]

        ref_rgb = (ref_rgbs.permute(0, 3, 1, 2))
        ref_rgb = torch.repeat_interleave(ref_rgb, repeats=num_pts, dim=0)
        ref_pose = torch.repeat_interleave(ref_poses, repeats=num_pts, dim=0)

        ro1, rd1 = torch.transpose(shader_or_rays_o, 0, 1).unsqueeze(0), torch.transpose(shader_or_rays_d, 0, 1).unsqueeze(0)  # 1, 3, H*W
        ro1, rd1 = ro1.repeat(num_pts * k_ref, 1, 1), rd1.repeat(num_pts * k_ref, 1, 1)
        ref_K = ref_K.unsqueeze(0).repeat(num_pts * k_ref, 1, 1)
        inv_K = torch.inverse(ref_K)

        # Should we enable grad????
        # ! warp H and W will be 1, N_rays
        warp_H = 1
        warp_W = shader_N_rays
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
    epi_pts = shader_rays_o[..., None, :] + shader_rays_d[..., None, :] * depth_values[..., :, None]

    plucker_embed = kwargs['embed_rays'](epi_pts, shader_rays_d[:, None, :].repeat(1, num_pts, 1)) # nump_pts + origin
    plucker_embed = plucker_embed.view(-1, num_pts, 6).view(shader_N_rays, -1)

    refine_input = torch.cat([plucker_embed, epi_features], dim =1)
    refine_output = refine_net(refine_input)
    refine_depth_values = torch.sigmoid(refine_output[:,:N_samples])
    refine_rgb = torch.sigmoid(refine_output[:, 4*N_samples:])
    points_offset = torch.tanh(refine_output[:, N_samples:4*N_samples]).view(shader_N_rays, N_samples,3)

    mids = .5 * (depth_values[...,1:] + depth_values[...,:-1])
    upper = torch.cat([mids, 0.5*(shader_far+depth_values[...,-1:])], -1) # upper cat far
    lower = torch.cat([0.5*(shader_near+depth_values[...,:1]), mids], -1) # lower cat near
    refine_depth_values = lower + (upper - lower) * refine_depth_values
    epi_z_vals = refine_depth_values

    query_points_nerf = shader_rays_o[..., None, :] + shader_rays_d[..., None,
                                                    :] * epi_z_vals[..., :, None]  # ! this is ndc space
    query_points_nerf = query_points_nerf + (1e-2) * points_offset
    raw = network_query_fn(query_points_nerf, shader_viewdirs, network_fn)

    iter = kwargs.get('iter', 1e6)
    raw_noise_std = 0
    shader_rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, epi_z_vals, shader_rays_d, raw_noise_std, white_bkgd,
                                                                 pytest=pytest, mm_density_add=mm_density_add,
                                                                 mm_density_mul=mm_density_mul, iter=iter)
    return shader_rgb_map

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
        low_imgs = torch.Tensor(low_imgs).to(device)
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
    render_kwargs_test['i_train'] = i_train
    render_kwargs_test['images'] = low_imgs[i_ref]
    render_kwargs_test['poses'] = poses[i_ref]
    render_kwargs_test['ref_K'] = K_ten


    # if (args.render_only):
    #     # Turn on testing mode
    #     if args.render_only:
    #         testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
    #             'test' if args.render_test else 'path', start))
    #         os.makedirs(testsavedir, exist_ok=True)
    #         moviebase = os.path.join(
    #             testsavedir, '{}_spiral_{:06d}_'.format(expname, i))
    #     else:
    #         moviebase = os.path.join(
    #             basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
    #     with torch.no_grad():
    #         r_out = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=None,savedir=testsavedir)
    #         rgbs0, rgbs1, depths, depths0 = r_out[0], r_out[1], r_out[2], r_out[3]
    #     print('Done, saving', rgbs0.shape)
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

    if (args.render_test):
        if args.render_test:
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
                'test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
        else:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format('debug'))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', poses[i_test].shape)
        with torch.no_grad():
            render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
        print('Saved test set')
        if (args.render_test):
            return


        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()