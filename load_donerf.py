import json
import os
import cv2

import numpy as np
import torch

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms as T

from iopath.common.file_io import PathManager, NativePathHandler

from donerf_utils import (
    correct_poses_bounds,
    center_poses_with,
    center_poses_with_rotation_only,
    create_spiral_poses,
    create_rotating_spiral_poses,
    interpolate_poses,
)

from donerf_utils import (
    get_ndc_rays_fx_fy,
    get_ray_directions_K,
    get_rays
)

import copy
from omegaconf import OmegaConf

class BaseDataset(Dataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
    ):

        ## Settings ##

        # Path manager
        self.pmgr = PathManager()
        self.pmgr.register_handler(NativePathHandler())

        # Copy train dataset config
        if 'train_dataset' in kwargs:
            base_dataset_cfg = copy.deepcopy(kwargs['train_dataset'].cfg.dataset)
            OmegaConf.set_struct(base_dataset_cfg, False)

            for key in cfg.dataset.keys():
                base_dataset_cfg.__dict__[key] = cfg.dataset[key]
                setattr(base_dataset_cfg, key, cfg.dataset[key])

            cfg.dataset = base_dataset_cfg

        ## Dataset cfg
        self.cfg = cfg
        self.split = getattr(cfg.dataset, 'split', split)
        self.dataset_cfg = getattr(cfg.dataset, self.split, cfg.dataset)

        # Basic dataset params
        self.root_dir = os.path.expanduser(self.dataset_cfg.root_dir)
        if hasattr(self.dataset_cfg, 'img_wh')  \
            and (
                not isinstance(self.dataset_cfg.img_wh, str) \
                and self.dataset_cfg.img_wh is not None
            ):
            self._img_wh = tuple(self.dataset_cfg.img_wh)
            self.img_wh = self._img_wh
            self.aspect = (float(self.img_wh[0]) / self.img_wh[1])
            self.downsample = 1
        else:
            self.img_wh = None
            self.downsample = getattr(self.dataset_cfg, 'downsample', 1)

        self.centered_pixels = getattr(self.dataset_cfg, 'centered_pixels', False)

        # Rendering
        self.render_supersample = self.dataset_cfg.render_params['supersample']
        self.render_crop = self.dataset_cfg.render_params['crop']

        # Validation
        self.val_num = self.dataset_cfg.val_num
        self.val_skip = self.dataset_cfg.val_skip
        self.val_set = []
        self.val_crop = 1.0
        self.val_all = (self.dataset_cfg.val_all) \
            or (kwargs['val_all'] if 'val_all' in kwargs else False)

        # Crop
        self.precrop_iters =  0
        self.use_crop = self.precrop_iters > 0
        self.cur_iter = 0
        self.precrop_frac = 0.5

        # Patch loading
        self.use_patches =  False
        self.use_one_image =  False
        self.use_full_image =  self.use_one_image
        self.blur_radius =  0

        ## Set-up data ##

        self.define_transforms()
        self.prepare_data()

    def read_meta(self):
        pass

    def prepare_train_data(self):
        self.num_images = len(self.image_paths)

        ## Collect training data
        self.all_coords = []
        self.all_rgb = []

        for idx in range(len(self.image_paths)):
            # coords
            self.all_coords += [self.get_coords(idx)]

            # Color
            self.all_rgb += [self.get_rgb(idx)]

        self.update_all_data(
            torch.cat(self.all_coords, 0),
            torch.cat(self.all_rgb, 0)
        )

    def update_all_data(self, coords, rgb):
        self.all_coords = coords
        self.all_rgb = rgb
        self.all_weights = self.get_weights()

        ## Patches
        if self.use_patches or self.use_crop:
            self._all_coords = torch.clone(self.all_coords)
            self._all_rgb = torch.clone(self.all_rgb)

        ## All inputs
        self.all_inputs = torch.cat(
            [self.all_coords, self.all_rgb, self.all_weights], -1
        )

    def prepare_val_data(self):
        self.prepare_test_data()

    def prepare_test_data(self):
        pass

    def prepare_render_data(self):
        pass

    def prepare_data(self):
        self.read_meta()

        if self.split == 'train':
            # if self.cfg.params.render_only or self.cfg.params.test_only:
            #     self.all_inputs = [0]
            # else:
            self.prepare_train_data()
        elif self.split == 'val':
            self.prepare_val_data()
        elif self.split == 'test':
            self.prepare_train_data()
        elif self.split == 'render':
            self.prepare_render_data()

    def define_transforms(self):
        if self.blur_radius > 0:
            self.transform = T.Compose([
                T.ToTensor(),
                T.GaussianBlur(
                    (self.blur_radius * 2 + 1, self.blur_radius * 2 + 1),
                    self.blur_radius / 3.0
                )
            ])
        else:
            self.transform = T.ToTensor()

    def scale(self, scale):
        self.img_wh = (self._img_wh[0] // scale, self._img_wh[1] // scale)
        self.aspect = (float(self.img_wh[0]) / self.img_wh[1])

        self.define_transforms()
        self.prepare_data()

    def get_coords(self, idx):
        pass

    def get_weights(self, device='cpu'):
        return torch.ones(
            *self.all_coords[..., 0:1].shape, device=device
        )

    def get_rgb(self, idx):
        pass

    def get_closest_rgb(self, query_pose):
        pass

    def shuffle(self):
        if not self.use_patches:
            # Get permutation
            if self.use_full_image:
                self.all_coords = self.all_coords.view(-1, self.img_wh[0] * self.img_wh[1], 6)
                self.all_rgb = self.all_rgb.view(-1, self.img_wh[0] * self.img_wh[1], 3)

                perm = torch.tensor(
                    np.random.permutation(self.all_coords.shape[0])
                )
            else:
                perm = torch.tensor(
                    np.random.permutation(len(self))
                )

            # Shuffle
            self.all_coords = self.all_coords[perm].view(-1, 6)
            self.all_rgb = self.all_rgb[perm].view(-1, 3)
        else:
            self.shuffle_patches()

        # Weights and inputs
        self.all_weights = self.get_weights()
        self.all_inputs = torch.cat(
            [self.all_coords, self.all_rgb, self.all_weights], -1
        )

    def __len__(self):
        if self.split == 'train':
            if self.cfg.params.render_only or self.cfg.params.test_only:
                return 1
            else:
                return len(self.all_coords)
        elif self.split == 'val':
            return min(self.val_num, len(self.poses))
        elif self.split == 'render':
            if self.render_max_frames > 0:
                return  min(self.render_max_frames, len(self.poses))
            else:
                return len(self.poses)
        else:
            return len(self.poses)

    def get_one_image_batch(self, idx, batch_size, device='cuda'):
        return None

    def __getitem__(self, idx):
        if self.split == 'render':
            batch = {
                'coords': self.get_coords(idx),
                'pose': self.poses[idx],
                'idx': idx
            }

            batch['weight'] = torch.ones_like(batch['coords'][..., -1:])

        elif self.split == 'val' or self.split == 'test':
            batch = {
                'coords': self.get_coords(idx),
                'rgb': self.get_rgb(idx),
                'idx': idx
            }

            batch['weight'] = torch.ones_like(batch['coords'][..., -1:])
        else:
            batch = {
                'inputs': self.all_inputs[idx],
            }


        W, H, batch = self.crop_batch(batch)
        batch['W'] = W
        batch['H'] = H

        return batch

    def format_batch(self, batch):
        batch['coords'] = batch['inputs'][..., :self.all_coords.shape[-1]]
        batch['rgb'] = batch['inputs'][..., self.all_coords.shape[-1]:self.all_coords.shape[-1] + 3]
        batch['weight'] = batch['inputs'][..., -1:]
        del batch['inputs']

        return batch

    def get_batch(self, batch_idx, batch_size, jitter=None):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        return self[batch_start:batch_end]

    def crop_all_tensors(
        self,
        t,
        W,
        H,
        dW,
        dH,
    ):
        t = t.view(self.num_images, H, W, -1)

        t = t[
            :,
            (H//2 - dH):(H // 2 + dH + 1),
            (W//2 - dW):(W // 2 + dW + 1),
        ]

        return t.reshape(-1, t.shape[-1])

    def crop_one_tensor(
        self,
        t,
        W,
        H,
        dW,
        dH,
    ):
        t = t.view(1, H, W, -1)

        t = t[
            :,
            (H//2 - dH):(H // 2 + dH + 1),
            (W//2 - dW):(W // 2 + dW + 1),
        ]

        H, W = t.shape[1], t.shape[2]

        return W, H, t.reshape(-1, t.shape[-1])

    def crop(self):
        if self.use_crop and self.cur_iter < self.precrop_iters:
            W = self.img_wh[0]
            H = self.img_wh[1]
            dW = int(W//2 * self.precrop_frac)
            dH = int(H//2 * self.precrop_frac)

            self.all_coords = self.crop_all_tensors(self._all_coords, W, H, dW, dH)
            self.all_rgb = self.crop_all_tensors(self._all_rgb, W, H, dW, dH)

    def crop_batch(self, batch):
        W = self.img_wh[0]
        H = self.img_wh[1]

        if self.split == 'val' or self.split == 'test':
            crop = self.val_crop
        elif self.split == 'render':
            crop = self.render_crop
        else:
            crop = 1.0

        if crop < 1.0:
            dW = int(W//2 * crop)
            dH = int(H//2 * crop)

            for k in batch.keys():
                if torch.is_tensor(batch[k]):
                    temp_W, temp_H, batch[k] = self.crop_one_tensor(batch[k], W, H, dW, dH)

            W, H = temp_W, temp_H

        return W, H, batch

    def patchify_tensor(
        self,
        t,
        width,
        height,
        patch_offset,
        patch_width,
    ):
        c = t.shape[-1]
        t = t.view(self.num_images, height, width, c)

        # Remove boundaries
        p = self.blur_radius

        if p > 0:
            t = t[:, p:-p, p:-p]

        # Patch offset
        t = t[:, patch_offset:, patch_offset:]

        # Crop to multiple of patch width
        round_height = (t.shape[1] // patch_width) * patch_width
        round_width = (t.shape[2] // patch_width) * patch_width
        t = t[:, :round_height, :round_width]

        t = t.reshape(
            t.shape[0],
            round_height // patch_width,
            patch_width,
            round_width // patch_width,
            patch_width,
            c
        ).permute(0, 1, 3, 2, 4, 5)

        return t.reshape(-1, patch_width * patch_width, c)

    def shuffle_patches(self):
        print("Shuffle patches")

        # Patchify
        patch_width = self.dataset_cfg.patch_width
        width, height = self.img_wh[0], self.img_wh[1]
        patch_offset = int(np.random.uniform() * patch_width)

        self.all_coords = self.patchify_tensor(
            self._all_coords,
            width,
            height,
            patch_offset,
            patch_width
        )

        self.all_rgb = self.patchify_tensor(
            self._all_rgb,
            width,
            height,
            patch_offset,
            patch_width
        )

        # Shuffle
        perm = torch.tensor(
            np.random.permutation(self.all_coords.shape[0])
        )

        self.all_coords = self.all_coords[perm].reshape(-1, self.all_coords.shape[-1])
        self.all_rgb = self.all_rgb[perm].reshape(-1, self.all_rgb.shape[-1])


class Base5DDataset(BaseDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):

        # Rendering
        self.render_spherical = cfg.dataset.spherical_poses if hasattr(cfg,'spherical_poses') else False
        self.render_interpolate = cfg.dataset.render_params['interpolate']
        self.render_max_frames = cfg.dataset.render_params['max_frames'] if 'max_frames' in cfg.dataset.render_params else 0
        self.use_ndc = cfg.dataset.use_ndc if hasattr(cfg.dataset, 'use_ndc') else False
        self.include_world = cfg.dataset.include_world if hasattr(cfg.dataset, 'include_world') else False

        super().__init__(cfg, split, **kwargs)

    def prepare_render_data(self):
        if self.render_spherical:
            self.poses = create_spherical_poses(self.bounds.max())

        if not self.render_interpolate:
            close_depth, inf_depth = self.bounds.min()*.9, self.bounds.max()*5.
            dt = .75
            mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
            focus_depth = mean_dz

            radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
            self.poses = create_spiral_poses(self.poses, radii, focus_depth)
        else:
            self.poses = interpolate_poses(self.poses, self.render_supersample)

    def get_intrinsics_screen_space(self):
        K = np.copy(self.get_intrinsics())
        K[0, 2] = (K[0, 2] - self.img_wh[0] / 2)
        K[1, 2] = (K[1, 2] - self.img_wh[1] / 2)
        K[0, :] = 2 * K[0, :] / self.img_wh[0]
        K[1, :] = -2 * K[1, :] / self.img_wh[1]
        return K

    def get_intrinsics(self):
        pass

    def get_closest_rgb(self, query_pose):
        W = self.img_wh[0]
        H = self.img_wh[1]

        images = self.all_rgb.view(self.num_images, H, W, -1)
        dists = np.linalg.norm(
            self.poses[:, :3, -1] - query_pose[None, :3, -1], axis=-1
        )
        return images[list(np.argsort(dists))[0]]


    def get_coords_from_camera(
        self,
        pose,
        time,
        cam_idx,
        K,
        W,
        H,
        device='cuda'
    ):
        # Directions
        directions = get_ray_directions_K(
            H, W, K, centered_pixels=True, device=device
        )

        # Pose
        c2w = torch.FloatTensor(pose[:3, :4]).to(device)

        # Rays
        rays_o, rays_d = get_rays(directions, c2w)
        rays = torch.cat([rays_o, rays_d], dim=-1)

        # To NDC
        if self.use_ndc:
            rays = self.to_ndc(rays)

        # Add camera idx
        rays = torch.cat([rays, torch.ones_like(rays[..., :1]) * cam_idx], dim=-1)

        # Add times
        rays = torch.cat([rays, torch.ones_like(rays[..., :1]) * time], dim=-1)
        
        # Add camera idx
        return rays


class Base6DDataset(Base5DDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):

        self.render_interpolate_time = cfg.dataset.render_params['interpolate_time'] if hasattr(cfg.dataset.render_params, 'interpolate_time') else False

        super().__init__(cfg, split, **kwargs)

    def prepare_render_data(self):
        if self.render_spherical:
            self.poses = create_spherical_poses(self.bounds.max())

        if not self.render_interpolate:
            close_depth, inf_depth = self.bounds.min()*.9, self.bounds.max()*5.
            dt = .75
            mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
            focus_depth = mean_dz

            radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
            self.poses = create_spiral_poses(self.poses, radii, focus_depth)
        else:
            self.poses = interpolate_poses(self.poses, self.render_supersample)

        # Get times
        if (self.num_frames - 1) > 0:
            self.times = np.linspace(0, self.num_frames - 1, len(self.poses))

            if not self.render_interpolate_time:
                self.times = np.round(self.times)

            self.times = self.times / (self.num_frames - 1)
        else:
            self.times = [0.0 for p in self.poses]

    def get_closest_rgb(self, query_pose, query_time):
        W = self.img_wh[0]
        H = self.img_wh[1]

        # Reshape
        images = self.all_rgb.view(self.num_frames, -1, H, W, self.all_rgb.shape[-1])
        poses = self.poses.reshape(self.num_frames, -1, self.poses.shape[-2], self.poses.shape[-1])

        # Get poses at current frame
        frame_idx = int(np.round(query_time * (self.num_frames - 1)))
        images = images[frame_idx]
        poses = poses[frame_idx]

        # Distances
        dists = np.linalg.norm(
            poses[:, :3, -1] - query_pose[None, :3, -1], axis=-1
        )

        # Closest rgb
        return images[list(np.argsort(dists))[0]]

    def __getitem__(self, idx):
        if self.split == 'render':
            batch = {
                'coords': self.get_coords(idx),
                'pose': self.poses[idx],
                'time': self.times[idx],
                'idx': idx
            }

            batch['weight'] = torch.ones_like(batch['coords'][..., -1:])

        elif self.split == 'val' or self.split == 'test':
            batch = {
                'coords': self.get_coords(idx),
                'rgb': self.get_rgb(idx),
                'idx': idx
            }

            batch['weight'] = torch.ones_like(batch['coords'][..., -1:])
        else:
            batch = {
                'inputs': self.all_inputs[idx],
            }


        W, H, batch = self.crop_batch(batch)
        batch['W'] = W
        batch['H'] = H

        return batch


###############DONERF DATASET######################


class DONeRFDataset(Base5DDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):

        self.correct_poses = cfg.dataset.correct_poses if hasattr(cfg.dataset, 'correct_poses') else False
        self.center_poses = cfg.dataset.center_poses if hasattr(cfg.dataset, 'center_poses') else False
        self.use_ndc = cfg.dataset.use_ndc if hasattr(cfg.dataset, 'use_ndc') else False
        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        if self.split == 'render':
            self.read_meta_for_split('test', 'cam_path_pan.json')
        elif self.split == 'test':
            self.read_meta_for_split('test', 'transforms_test.json')
        elif self.split == 'train':
            self.read_meta_for_split('train', 'transforms_train.json')
        elif self.split == 'val':
            self.read_meta_for_split('val', 'transforms_val.json')
        else:
            self.read_meta_for_split(self.split, 'transforms_test.json')

    def load_poses_from_meta(self, meta, dataset_meta):
        origin = np.array(dataset_meta['view_cell_center'])

        # Image paths and pose
        image_paths = []
        poses = []

        for frame in meta['frames']:
            # Image path
            if 'file_path' in frame:
                image_paths += [frame['file_path']]
            else:
                image_paths += [None]

            # Pose
            pose = np.array(frame['transform_matrix'])[:3, :4]

            if self.center_poses:
                pose[:3, -1] = pose[:3, -1] - origin

            poses += [pose]

        poses = np.stack(poses, axis=0)

        return poses, image_paths

    def read_meta_for_split(self, split, split_file):
        # Load train meta
        with self.pmgr.open(
            os.path.join(self.root_dir, 'transforms_train.json'),
            'r'
        ) as f:
            self.train_meta = json.load(f)

        # Load meta
        with self.pmgr.open(
            os.path.join(self.root_dir, split_file),
            'r'
        ) as f:
            self.meta = json.load(f)

        if split == 'val':
            self.meta['frames'] = self.meta['frames'][:self.val_num]

        # Load dataset info
        with self.pmgr.open(
            os.path.join(self.root_dir, 'dataset_info.json'),
            'r'
        ) as f:
            self.dataset_meta = json.load(f)

        W, H = self.img_wh

        self.focal = 0.5 * 800 / np.tan(
            0.5 * self.dataset_meta['camera_angle_x']
        )
        self.focal *= self.img_wh[0] / 800

        self.K = np.eye(3)
        self.K[0, 0] = self.focal
        self.K[0, 2] = (W / 2.0)
        self.K[1, 1] = self.focal
        self.K[1, 2] = (H / 2.0)

        # Bounds, common for all scenes
        self.depth_range = self.dataset_meta['depth_range']
        self.near = self.dataset_meta['depth_range'][0]
        self.far = self.dataset_meta['depth_range'][1]
        #self.depth_range = np.array([self.near * 1.5, self.far])

        self.view_cell_size = np.max(np.array(self.dataset_meta['view_cell_size']))
        self.bounds = np.array([self.near, self.far])

        # Image paths and pose
        self.train_poses, _ = self.load_poses_from_meta(self.train_meta, self.dataset_meta)
        self.poses, self.image_paths = self.load_poses_from_meta(self.meta, self.dataset_meta)

        # Correct
        if self.use_ndc or self.correct_poses:
            self.poses, _ = center_poses_with_rotation_only(self.poses, self.train_poses)
        
            if self.dataset_cfg.collection in ['pavillon'] and self.split == 'render':
                self.poses[..., :3, -1] *= 0.35

        # Ray directions for all pixels, same for all images (same H, W, focal)
        self.centered_pixels = True
        self.directions = get_ray_directions_K(
            H, W, self.K, centered_pixels=self.centered_pixels
        )

    def prepare_train_data(self):
        self.num_images = len(self.image_paths)

        ## Collect training data
        self.all_coords = []
        self.all_rgb = []
        self.all_depth = []
        self.all_points = []

        for idx in range(len(self.image_paths)):
            # coords
            self.all_coords += [self.get_coords(idx)]

            # Color
            self.all_rgb += [self.get_rgb(idx)]

            # Depth
            self.all_depth += [self.get_depth(idx)]

            # Points
            self.all_points += [self.get_points(idx)]

        self.all_rgb = torch.stack(self.all_rgb, 0)
        # self.update_all_data(
        #     torch.cat(self.all_coords, 0),
        #     torch.stack(self.all_rgb, 0),
        #     torch.cat(self.all_depth, 0),
        #     torch.cat(self.all_points, 0),
        # )

        # # Calculate bounds
        # mask = (self.all_depth != 0.0)
        # self.bbox_min = self.all_points[mask.repeat(1, 3)].reshape(-1, 3).min(0)[0]
        # self.bbox_max = self.all_points[mask.repeat(1, 3)].reshape(-1, 3).max(0)[0]

        #self.near = float(self.all_depth[mask].min())
        #self.far = float(self.all_depth[mask].max())

    def update_all_data(self, coords, rgb, depth, points):
        self.all_coords = coords
        self.all_rgb = rgb
        self.all_depth = depth
        self.all_points = points
        self.all_weights = self.get_weights()

        ## Patches
        if self.use_patches or self.use_crop:
            self._all_coords = torch.clone(self.all_coords)
            self._all_rgb = torch.clone(self.all_rgb)
            self._all_depth = torch.clone(self.all_depth)

        ## All inputs
        self.all_inputs = torch.cat(
            [self.all_coords, self.all_rgb, self.all_depth, self.all_weights], -1
        )

    def format_batch(self, batch):
        batch['coords'] = batch['inputs'][..., :self.all_coords.shape[-1]]
        batch['rgb'] = batch['inputs'][..., self.all_coords.shape[-1]:self.all_coords.shape[-1] + 3]
        batch['depth'] = batch['inputs'][..., self.all_coords.shape[-1] + 3:self.all_coords.shape[-1] + 4]
        batch['weight'] = batch['inputs'][..., -1:]
        del batch['inputs']

        return batch

    def prepare_render_data(self):
        self.prepare_test_data()

    def to_ndc(self, rays):
        return get_ndc_rays_fx_fy(
            self.img_wh[1], self.img_wh[0], self.K[0, 0], self.K[1, 1], self.near, rays
        )

    def get_coords(self, idx):
        c2w = torch.FloatTensor(self.poses[idx])
        rays_o, rays_d = get_rays(self.directions, c2w)

        if self.use_ndc:
            return self.to_ndc(torch.cat([rays_o, rays_d], dim=-1))
        else:
            return torch.cat([rays_o, rays_d], dim=-1)

    def get_rgb(self, idx):
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, f'{image_path}.png'),
            'rb'
        ) as im_file:
            img = np.array(Image.open(im_file).convert('RGBA'))

        img = cv2.resize(img, self._img_wh, interpolation=cv2.INTER_AREA)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_AREA)

        img = self.transform(img)
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])
        return img.view(self.img_wh[1], self.img_wh[0],3)

    def get_depth(self, idx):
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, f'{image_path}_depth.npz'),
            'rb'
        ) as depth_file:
            with np.load(depth_file) as depth:
                img = depth['arr_0'].reshape(800, 800)

        # Resize
        img = cv2.resize(img, self._img_wh, interpolation=cv2.INTER_NEAREST)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_NEAREST)

        # Flip
        img = np.flip(img, 0)

        # Transform
        img = self.transform(np.copy(img))

        # Return
        depth = img.view(1, -1).permute(1, 0)
        directions = torch.nn.functional.normalize(self.directions, p=2.0, dim=-1).view(-1, 3)
        depth = depth / torch.abs(directions[..., 2:3])

        #depth[depth < self.near] = self.near
        #depth[depth > self.far] = self.far
        depth[depth < self.near] = 0.0
        depth[depth > self.far] = 0.0

        return depth

    def get_points(self, idx):
        rays = self.all_coords[idx][..., :6].reshape(-1, 6)
        depth = self.all_depth[idx].reshape(-1, 1)
        return rays[..., :3] + rays[..., 3:6] * depth

    def get_intrinsics(self):
        K = np.eye(3)
        K[0, 0] = self.focal
        K[0, 2] = self.img_wh[0] / 2
        K[1, 1] = self.focal
        K[1, 2] = self.img_wh[1] / 2

        return K

    def __getitem__(self, idx):
        if self.split == 'render':
            batch = {
                'coords': self.get_coords(idx),
                'pose': self.poses[idx],
                'idx': idx
            }

            batch['weight'] = torch.ones_like(batch['coords'][..., -1:])

        elif self.split == 'test':
            batch = {
                'coords': self.get_coords(idx),
                'rgb': self.get_rgb(idx),
                'idx': idx
            }

            batch['weight'] = torch.ones_like(batch['coords'][..., -1:])
        elif self.split == 'val':
            batch = {
                'coords': self.get_coords(idx),
                'rgb': self.get_rgb(idx),
                'depth': self.get_depth(idx),
                'idx': idx
            }

            batch['weight'] = torch.ones_like(batch['coords'][..., -1:])
        else:
            batch = {
                'inputs': self.all_inputs[idx],
            }


        W, H, batch = self.crop_batch(batch)
        batch['W'] = W
        batch['H'] = H

        return batch

class DatasetConfig:
    def __init__(self) -> None:
        self.name = 'donerf'
        self.collection = 'barbershop'
        self.data_subdir =  'donerf'
        self.root_dir = os.path.join('./data', self.data_subdir, self.collection)

        self.img_wh = [800, 800]
        self.spherical_poses = True
        self.use_ndc: False
        self.correct_poses = True
        self.center_poses = True

        self.val_num =  8
        self.val_skip = 1
        self.val_all = False

        self.render_params = {
            'interpolate': False,
            'supersample' : 4,
            'crop': 1.0
        }

class Config():
    def __init__(self) -> None:
        self.dataset = DatasetConfig()

def load_donerf_data():
    cfg = Config()
    train_set = DONeRFDataset(cfg, 'train')
    images_train, poses_train, near, far = train_set.all_rgb, train_set.poses, train_set.near, train_set.far
    test_set = DONeRFDataset(cfg, 'test')
    images_test, poses_test, _, _ = test_set.all_rgb, test_set.poses, test_set.near, test_set.far

    i_train = [i for i in range(poses_train.shape[0])]
    i_test = [i + poses_train.shape[0] for i in range(poses_test.shape[0])]
    hwf = np.array([train_set.img_wh[1], train_set.img_wh[0], train_set.focal])

    render_set = DONeRFDataset(cfg, 'render')
    render_poses = render_set.poses
    return torch.cat([images_train, images_test], dim =0).numpy(), np.concatenate([poses_train, poses_test], axis =0), near, far, hwf, render_poses, i_train, i_test

# if __name__ == '__main__':
#     cfg = Config()
#     train_set = DONeRFDataset(cfg, 'train')
#     images_train, poses_train, near, far = train_set.all_rgb, train_set.poses, train_set.near, train_set.far
#     test_set = DONeRFDataset(cfg, 'test')
#     images_test, poses_test, _, _ = test_set.all_rgb, test_set.poses, test_set.near, test_set.far

#     render_set = DONeRFDataset(cfg, 'render')

#     i_train = [i for i in range(poses_train.shape[0])]
#     i_test = [i + poses_train.shape[0] for i in range(poses_test.shape[0])]

#     hwf = np.array([train_set.img_wh[1], train_set.img_wh[0], train_set.focal])
#     breakpoint()