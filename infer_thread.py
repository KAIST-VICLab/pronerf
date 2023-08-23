import threading
import numpy as np

import pycuda.driver as cuda
from trt_infer_v2_mask import *
import torch
import inverse_warp
import torch.nn.functional as F


class AVRThread(threading.Thread):
    def __init__(self, eventStart, eventEnd, avr_mm_input_holder, avr_refine_input_holder, basedir, expname):
        threading.Thread.__init__(self)
        
        self.eventStart = eventStart
        self.eventEnd = eventEnd

        self.cuda_ctx = None  # to be created when run
        self.avr_mm_engine = None   # to be created when run
        self.avr_refine_engine = None   # to be created when run

        self.input = None
        self.output = None
        self.shutdown = False

        self.avr_mm_input_holder = avr_mm_input_holder
        self.avr_refine_input_holder = avr_refine_input_holder

        self.basedir = basedir
        self.expname = expname

    def run(self):
        """Run until 'running' flag is set to False by main thread.

        NOTE: CUDA context is created here, i.e. inside the thread
        which calls CUDA kernels.  In other words, creating CUDA
        context in __init__() doesn't work.
        """

        print('[AVRThread]: Loading the TRT model...')
        self.cuda_ctx = cuda.Device(0).make_context()  # GPU 0

        self.avr_mm_engine = AVRMMEngine(os.path.join(self.basedir, self.expname, 'avr_minmaxrays_net_fp16.trt'))
        self.avr_refine_engine = AVRRefineEngine(os.path.join(self.basedir, self.expname, 'avr_refine_net_fp16.trt'))

        # Dummy input 
        self.avr_mm_engine.bind_input(self.avr_mm_input_holder, warmup=True)
        _ = self.avr_mm_engine.run()

        self.avr_refine_engine.bind_input(self.avr_refine_input_holder, warmup=True)
        _ = self.avr_refine_engine.run()
        self.cuda_ctx.pop()

        print('AVRThread: start running...')
        while not self.shutdown:
            self.eventStart.wait()
            self.eventStart.clear()
            if self.input is not None:
                self.output = self.process_fn(self.input)
                self.img = None

            self.eventEnd.set()

        del self.trt_model
        self.cuda_ctx.pop()
        del self.cuda_ctx
        print('AVRThread: stopped...')

    def process_fn(self, kwargs):

        avr_plucker_pts = kwargs['avr_plucker_pts']
        N_samples = kwargs['N_samples']
        avr_near = kwargs['avr_near']
        avr_far = kwargs['avr_far']
        avr_N_rays = kwargs['avr_N_rays']
        avr_or_rays_o = kwargs['avr_or_rays_o']
        avr_or_rays_d = kwargs['avr_or_rays_d']
        avr_rays_o = kwargs['avr_rays_o']
        avr_rays_d = kwargs['avr_rays_d']
        avr_viewdirs = kwargs['avr_viewdirs'] 

        self.avr_mm_engine.bind_input(avr_plucker_pts)
        F_theta0_out = self.avr_mm_engine.run()
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

        self.avr_refine_engine.bind_input(net_input)
        F_theta1_output = self.avr_refine_engine.run()
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

    def stop(self):
        self.join()