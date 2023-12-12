import torch.utils.data as data
import torch
import numpy as np
import random

# Usage?
# ray_dataset = ray_loader_dataset.RayDataset(rays_rgb, poses_train, H, W, N_rand, N_iters)
# ray_loader = torch.utils.data.DataLoader(ray_dataset, batch_size=1,
#                                             num_workers=2,
#                                             pin_memory=False, shuffle=False)

class RayDataset(data.Dataset):
    def __init__(self, rays_rgb, poses_train, H, W, N_rand, N_iters):
        self.rays_rgb = rays_rgb
        print(self.rays_rgb.shape)
        self.poses_train = poses_train
        print(self.poses_train.shape)
        self.H = H
        self.W = W
        self.N_rand = N_rand
        self.N_iters = N_iters

    def __len__(self):
        return self.N_iters

    def __getitem__(self, index):
        # with torch.no_grad():
        H, W, N_rand = self.H, self.W, self.N_rand

        # Random from one image
        img_i = random.choice(range(self.rays_rgb.shape[0]))

        # Reference data
        ref_poses = [self.poses_train[n_ref] for n_ref in range(self.rays_rgb.shape[0]) if n_ref != img_i]
        ref_poses = torch.stack(ref_poses, 0)
        ref_rgbs = [self.rays_rgb[n_ref, :, :, 2, :] for n_ref in range(self.rays_rgb.shape[0]) if n_ref != img_i]
        ref_rgbs = torch.stack(ref_rgbs, 0)

        # target data
        these_rays_rgb = self.rays_rgb[img_i]
        rays_o = these_rays_rgb[:, :, 0, :]
        rays_d = these_rays_rgb[:, :, 1, :]
        target = these_rays_rgb[:, :, 2, :]
        pose = self.poses_train[img_i, :3, :4]
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1) #(H,W,2)

        # Random selection of rays -> This takes time!
        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = torch.stack([rays_o, rays_d], 0)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        ref_rgbs = ref_rgbs.view(-1, H * W, 3)  # (N_ref, H*W, 3)

        return pose, ref_poses, ref_rgbs, select_coords, batch_rays, target_s
