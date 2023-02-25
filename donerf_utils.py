import numpy as np
import scipy.linalg

import torch
import torch.nn.functional as F
import numpy as np

from kornia import create_meshgrid


def normalize(v):
    return v / np.linalg.norm(v)

def average_poses(poses):
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    R = np.stack([x, y, z], 1)

    center = center[..., None]
    #center = (-R @ center[..., None])

    pose_avg = np.concatenate([R, center], 1) # (3, 4)

    return pose_avg

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def center_poses(poses):
    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)


def center_poses_with(poses, train_poses, avg_pose=None):
    if avg_pose is None:
        pose_avg = average_poses(train_poses) # (3, 4)
        pose_avg_homo = np.eye(4)
        pose_avg_homo[:3] = pose_avg
        inv_pose = np.linalg.inv(pose_avg_homo)
    else:
        inv_pose = np.copy(avg_pose)

    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate
    poses_centered = inv_pose @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, inv_pose


def center_poses_with_rotation_only(poses, train_poses):
    pose_avg = average_poses(train_poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3, :3] = pose_avg[:3, :3]
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)


def center_poses_reference(poses):
    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)

    pose_avg_homo[:3] = pose_avg

    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    # Get reference
    dists = np.sum(np.square(pose_avg[:3, 3] - poses[:, :3, 3]), -1)
    reference_view_id = np.argmin(dists)
    pose_avg_homo = poses_homo[reference_view_id]

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)

def create_rotating_spiral_poses(camera_offset, poses, pose_rad, spiral_rads, focal, theta_range, N=240, rots=4, flip=False):
    # Camera offset and up
    camera_offset = np.array(camera_offset)
    up = normalize(poses[:, :3, 1].sum(0))

    # Radii in X, Y, Z
    render_poses = []
    spiral_rads = np.array(list(spiral_rads) + [1.0])

    # Pose, spiral angle
    pose_thetas = np.linspace(
        np.pi * theta_range[0],
        np.pi * theta_range[1],
        N,
        endpoint=False
    )

    spiral_thetas = np.linspace(
        0,
        2 * np.pi * rots,
        N,
        endpoint=False
    )

    # Create poses
    for pose_theta, spiral_theta in zip(pose_thetas, spiral_thetas):
        # Central cylindrical pose
        pose_x, pose_z = (
            np.sin(pose_theta) * pose_rad,
            -np.cos(pose_theta) * pose_rad,
        )
        pose_y = 0

        pose_center = np.array([pose_x, pose_y, pose_z]) + camera_offset
        pose_forward = np.array([-pose_x, -pose_y, -pose_z])
        c2w = viewmatrix(pose_forward, up, pose_center)

        # Spiral pose
        c = np.dot(c2w[:3,:4], np.array(
            [np.cos(spiral_theta), -np.sin(spiral_theta), -np.sin(spiral_theta * 0.5), 1.]
            ) * spiral_rads
        )

        z = normalize(c - np.dot(c2w[:3,:4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))

    return render_poses

def create_spiral_poses(poses, rads, focal, N=120, flip=False):
    c2w = average_poses(poses)
    up = normalize(poses[:, :3, 1].sum(0))
    rots = 2

    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array(
            [np.cos(theta), -np.sin(theta), -np.sin(theta*0.5), 1.]
            ) * rads
        )

        if flip:
            z = normalize(np.dot(c2w[:3,:4], np.array([0, 0, focal, 1.])) - c)
        else:
            z = normalize(c - np.dot(c2w[:3,:4], np.array([0, 0, -focal, 1.])))

        render_poses.append(viewmatrix(z, up, c))

    return render_poses

def create_spherical_poses(radius, n_poses=120):
    def spherical_pose(theta, phi, radius):
        def trans_t(t):
            return np.array(
                [
                    [1,0,0,0],
                    [0,1,0,-0.9*t],
                    [0,0,1,t],
                    [0,0,0,1],
                ]
            )

        def rot_phi(phi):
            return np.array(
                [
                    [1,0,0,0],
                    [0,np.cos(phi),-np.sin(phi),0],
                    [0,np.sin(phi), np.cos(phi),0],
                    [0,0,0,1],
                ]
            )

        def rot_theta(th):
            return np.array(
                [
                    [np.cos(th),0,-np.sin(th),0],
                    [0,1,0,0],
                    [np.sin(th),0, np.cos(th),0],
                    [0,0,0,1],
                ]
            )

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array(
            [[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]
        ) @ c2w
        return c2w[:3]

    spherical_poses = []

    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spherical_poses += [spherical_pose(th, -np.pi/5, radius)] # 36 degree view downwards

    return np.stack(spherical_poses, 0)

def correct_poses_bounds(poses, bounds, flip=True, use_train_pose=False, center=True, train_poses=None):
    # Original poses has rotation in form "down right back", change to "right up back"
    # See https://github.com/bmild/nerf/issues/34
    if flip:
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)

    # See https://github.com/bmild/nerf/issues/34
    if train_poses is None:
        near_original = bounds.min()
        scale_factor = near_original * 0.75 # 0.75 is the default parameter
        bounds /= scale_factor
        poses[..., :3, 3] /= scale_factor

    # Recenter
    if center:
        if use_train_pose:
            if train_poses is not None:
                poses, ref_pose = center_poses_with(poses, train_poses)
            else:
                poses, ref_pose = center_poses_reference(poses)
        else:
            poses, ref_pose = center_poses(poses)
    else:
        ref_pose = poses[0]

    return poses, ref_pose, bounds

# Assumes centered poses
def get_bounding_sphere(poses):
    dists = np.linalg.norm(poses[:, :3, -1], axis=-1)
    return dists.max()

def get_bounding_box(poses):
    min_x, max_x = poses[:, 0, -1].min(), poses[:, 0, -1].max()
    min_y, max_y = poses[:, 1, -1].min(), poses[:, 1, -1].max()
    min_z, max_z = poses[:, 2, -1].min(), poses[:, 2, -1].max()

    return [min_x, min_y, min_z, max_x, max_y, max_z]

def p34_to_44(p):
    return np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1
    )

def poses_to_twists(poses):
    twists = []

    for i in range(poses.shape[0]):
        M = scipy.linalg.logm(poses[i])
        twist = np.stack(
            [
                M[..., 2, 1],
                M[..., 0, 2],
                M[..., 1, 0],
                M[..., 0, 3],
                M[..., 1, 3],
                M[..., 2, 3],
            ],
            axis=-1
        )
        twists.append(twist)

    return np.stack(twists, 0)

def twists_to_poses(twists):
    poses = []

    for i in range(twists.shape[0]):
        twist = twists[i]
        null = np.zeros_like(twist[..., 0])

        M = np.stack(
            [
                np.stack(
                    [
                    null,
                    twist[..., 2],
                    -twist[..., 1],
                    null
                    ],
                    axis=-1
                ),
                np.stack(
                    [
                    -twist[..., 2],
                    null,
                    twist[..., 0],
                    null
                    ],
                    axis=-1
                ),
                np.stack(
                    [
                    twist[..., 1],
                    -twist[..., 0],
                    null,
                    null
                    ],
                    axis=-1
                ),
                np.stack(
                    [
                    twist[..., 3],
                    twist[..., 4],
                    twist[..., 5],
                    null
                    ],
                    axis=-1
                ),
            ],
            axis=-1
        )

        poses.append(scipy.linalg.expm(M))

    return np.stack(poses, 0)

def interpolate_poses(poses, supersample):
    t = np.linspace(0, 1, supersample, endpoint=False).reshape(1, supersample, 1)
    twists = poses_to_twists(p34_to_44(poses))

    interp_twists = twists.reshape(-1, 1, twists.shape[-1])
    interp_twists = (1 - t) * interp_twists[:-1] + t * interp_twists[1:]
    interp_twists = interp_twists.reshape(-1, twists.shape[-1])
    interp_twists = np.concatenate([interp_twists, np.tile(twists[-1:], [supersample, 1])], 0)

    return twists_to_poses(interp_twists)[:, :3, :4]


def get_lightfield_rays(
    U, V, s, t, aspect, st_scale=1.0, uv_scale=1.0, near=-1, far=0,
    use_inf=False, center_u=0.0, center_v=0.0,
    ):
    u = torch.linspace(-1, 1, U, dtype=torch.float32)
    v = torch.linspace(1, -1, V, dtype=torch.float32) / aspect

    vu = list(torch.meshgrid([v, u]))
    u = vu[1] * uv_scale
    v = vu[0] * uv_scale
    s = torch.ones_like(vu[1]) * s * st_scale
    t = torch.ones_like(vu[0]) * t * st_scale

    rays = torch.stack(
        [
            s,
            t,
            near * torch.ones_like(s),
            u - s,
            v - t,
            (far - near) * torch.ones_like(s),
        ],
        axis=-1
    ).view(-1, 6)

    return torch.cat(
        [
            rays[..., 0:3],
            torch.nn.functional.normalize(rays[..., 3:6], p=2.0, dim=-1)
        ],
        -1
    )

def get_epi_rays(
    U, v, S, t, aspect, st_scale=1.0, uv_scale=1.0, near=-1, far=0,
    use_inf=False, center_u=0.0, center_v=0.0,
    ):
    u = torch.linspace(-1, 1, U, dtype=torch.float32)
    s = torch.linspace(-1, 1, S, dtype=torch.float32) / aspect

    su = list(torch.meshgrid([s, u]))
    u = su[1] * uv_scale
    v = torch.ones_like(su[0]) * v * uv_scale
    s = su[0] * st_scale
    t = torch.ones_like(su[0]) * t * st_scale

    rays = torch.stack(
        [
            s,
            t,
            near * torch.ones_like(s),
            u - s,
            v - t,
            (far - near) * torch.ones_like(s),
        ],
        axis=-1
    ).view(-1, 6)

    return torch.cat(
        [
            rays[..., 0:3],
            torch.nn.functional.normalize(rays[..., 3:6], p=2.0, dim=-1)
        ],
        -1
    )

def get_pixels_for_image(
    H, W, device='cpu'
):
    grid = create_meshgrid(H, W, normalized_coordinates=False, device=device)[0]

    return grid

def get_random_pixels(
    n_pixels, H, W, device='cpu'
):
    grid = torch.rand(n_pixels, 2, device=device)

    i, j = grid.unbind(-1)
    grid[..., 0] = grid[..., 0] * (W - 1)
    grid[..., 1] = grid[..., 1] * (H - 1)

    return grid

def get_ray_directions_from_pixels_K(
    grid, K, centered_pixels=False, flipped=False
):
    i, j = grid.unbind(-1)

    offset_x = 0.5 if centered_pixels else 0.0
    offset_y = 0.5 if centered_pixels else 0.0

    directions = torch.stack(
        [
            (i - K[0, 2] + offset_x) / K[0, 0],
            (-(j - K[1, 2] + offset_y) / K[1, 1]) if not flipped else (j - K[1, 2] + offset_y) / K[1, 1],
            -torch.ones_like(i)
        ],
        -1
    )

    return directions

def get_ray_directions_K(H, W, K, centered_pixels=False, flipped=False, device='cpu'):
    grid = create_meshgrid(H, W, normalized_coordinates=False, device=device)[0]
    return get_ray_directions_from_pixels_K(grid, K, centered_pixels, flipped=flipped)

def get_rays(directions, c2w, normalize=True):
    # Implementation: https://github.com/kwea123/nerf_pl

    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    if normalize:
        rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)

    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def get_ndc_rays_fx_fy(H, W, fx, fy, near, rays):
    rays_o, rays_d = rays[..., 0:3], rays[..., 3:6]

    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # o_z = -near
    # (o_z / (1 - t') - o_z) / d_z

    # Store some intermediate homogeneous results
    ox_oz = rays_o[...,0] / rays_o[...,2]
    oy_oz = rays_o[...,1] / rays_o[...,2]

    # Projection
    o0 = -1./(W/(2.*fx)) * ox_oz
    o1 = -1./(H/(2.*fy)) * oy_oz
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*fx)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
    d1 = -1./(H/(2.*fy)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
    #rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)

    return torch.cat([rays_o, rays_d], -1)

def sample_images_at_xy(
    images,
    xy_grid,
    H, W,
    mode="bilinear",
    padding_mode="border"
):
    batch_size = images.shape[0]
    spatial_size = images.shape[1:-1]

    xy_grid = torch.clone(xy_grid.reshape(batch_size, -1, 1, 2))
    xy_grid[..., 0] = (xy_grid[..., 0] / (W - 1)) * 2 - 1
    xy_grid[..., 1] = (xy_grid[..., 1] / (H - 1)) * 2 - 1

    images_sampled = torch.nn.functional.grid_sample(
        images.permute(0, 3, 1, 2),
        xy_grid,
        align_corners=False,
        mode=mode,
        padding_mode=padding_mode,
    )

    return images_sampled.permute(0, 2, 3, 1).view(-1, images.shape[-1])

def dot(a, b, axis=-1, keepdim=False):
    return torch.sum(a * b, dim=axis, keepdim=keepdim)

def reflect(dirs, normal):
    dir_dot_normal = dot(-dirs, normal, keepdim=True) * normal
    return 2 * dir_dot_normal + dirs

def get_stats(rays):
    return (rays.mean(0), rays.std(0))

def get_weight_map(
    rays,
    jitter_rays,
    cfg,
    weights=None,
    softmax=True
):
    ray_dim = rays.shape[-1] // 2

    # Angles
    angles = torch.acos(
        torch.clip(
            dot(rays[..., ray_dim:], jitter_rays[..., ray_dim:]),
            -1 + 1e-8, 1 - 1e-8
        )
    ).detach()

    # Distances
    dists = torch.linalg.norm(
        rays[..., :ray_dim] - jitter_rays[..., :ray_dim],
        dim=-1
    ).detach()

    # Weights
    if weights is None:
        weights = torch.zeros_like(angles)

    if softmax:
        weights = torch.nn.functional.softmax(
            0.5 * -(torch.square(angles / cfg.angle_std) + torch.square(dists / cfg.dist_std)) + weights, dim=0
        )[..., None]
    else:
        #print("Angle:", angles.max(), angles.mean(), cfg.angle_std)
        #print("Dist:", dists.max(), dists.mean(), cfg.dist_std)

        weights = torch.exp(
            0.5 * -(torch.square(angles / cfg.angle_std) + torch.square(dists / cfg.dist_std)) + weights
        )[..., None]

    # Normalization constant
    constant = np.power(2 * np.pi * cfg.angle_std * cfg.angle_std, -1.0 / 2.0) \
        * np.power(2 * np.pi * cfg.dist_std * cfg.dist_std, -1.0 / 2.0)

    return weights / constant

def compute_sigma_angle(
    query_ray,
    rays,
    angle_std=-1
):
    # Angles
    angles = torch.acos(
        torch.clip(
            dot(rays, query_ray),
            -1 + 1e-8, 1 - 1e-8
        )
    )

    # Calculate angle std
    if angle_std < 0:
        mean_ray = torch.nn.functional.normalize(rays.mean(1).unsqueeze(1), dim=-1)
        mean_angles = torch.acos(
            torch.clip(
                dot(mean_ray, query_ray),
                -1 + 1e-8, 1 - 1e-8
            )
        )

        angle_std, _ = torch.median(torch.abs(mean_angles), dim=1, keepdim=True)
        print(angle_std[0])
        c = torch.pow(2 * np.pi * angle_std * angle_std, -1.0 / 2.0)
    else:
        c = np.power(2 * np.pi * angle_std * angle_std, -1.0 / 2.0)

    # Weights
    weights = torch.exp(
        0.5 * -(torch.square(angles / angle_std))
    )[..., None]
    weights = c * weights.mean(1)

    return weights * c

def compute_sigma_dot(
    query_ray,
    rays,
    dot_std=-1
):
    # Dots
    dots = torch.clip(
        dot(rays, query_ray),
        -1 + 1e-8,
        1 - 1e-8
    )

    # Calculate dot std
    if dot_std < 0:
        mean_ray = torch.nn.functional.normalize(rays.mean(1).unsqueeze(1), dim=-1)
        mean_dots = torch.clip(
            dot(mean_ray, query_ray),
            -1 + 1e-8, 1 - 1e-8
        )

        dot_std, _ = torch.median(torch.abs(1 - mean_dots), dim=1, keepdim=True)
        print(dot_std[0])

        c = torch.pow(2 * np.pi * dot_std * dot_std, -1.0 / 2.0)
    else:
        c = np.power(2 * np.pi * dot_std * dot_std, -1.0 / 2.0)

    # Weights
    weights = torch.exp(
        0.5 * -(torch.square((1 - dots) / dot_std))
    )[..., None]
    weights = c * weights.mean(1)

    return weights * c


def weighted_stats(rgb, weights):
    weights_sum = weights.sum(0)
    rgb_mean = ((rgb * weights).sum(0) / weights_sum)
    rgb_mean = torch.where(
        weights_sum == 0,
        torch.zeros_like(rgb_mean),
        rgb_mean
    )

    diff = rgb - rgb_mean.unsqueeze(0)
    rgb_var = (diff * diff * weights).sum(0) / weights_sum
    rgb_var = torch.where(
        weights_sum == 0,
        torch.zeros_like(rgb_var),
        rgb_var
    )

    return rgb_mean, rgb_var

def jitter_ray_origins(rays, jitter):
    ray_dim = 3

    pos_rand = torch.randn(
        (rays.shape[0], jitter.bundle_size, ray_dim), device=rays.device
    ) * jitter.pos

    rays = rays.view(rays.shape[0], -1, rays.shape[-1])

    if rays.shape[1] == 1:
        rays = rays.repeat(1, jitter.bundle_size, 1)

    rays_o = rays[..., :ray_dim] + pos_rand.type_as(rays)

    return torch.cat([rays_o, rays[..., ray_dim:]], -1)

def jitter_ray_directions(rays, jitter):
    ray_dim = 3

    dir_rand = torch.randn(
        (rays.shape[0], jitter.bundle_size, ray_dim), device=rays.device
    ) * jitter.dir

    rays = rays.view(rays.shape[0], -1, rays.shape[-1])

    if rays.shape[1] == 1:
        rays = rays.repeat(1, jitter.bundle_size, 1)

    rays_d = rays[..., ray_dim:2*ray_dim] + dir_rand.type_as(rays)
    rays_d = F.normalize(rays_d, dim=-1)

    return torch.cat([rays[..., :ray_dim], rays_d], -1)


def from_ndc(t_p, rays, near):
    t = (near / (1 - t_p) - near) / rays[..., 5, None]
    t = t + (near - rays[..., None, 2]) / rays[..., None, 5]
    return t


def get_ray_density(sigma, ease_iters, cur_iter):
    if cur_iter >= ease_iters:
        return sigma
    else:
        w = min(max(float(ease_iters) / cur_iter, 0.0), 1.0)
        return sigma * w + (1 - w)