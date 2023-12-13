from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os

torch.set_default_tensor_type("torch.cuda.FloatTensor")
pixel_coords = None
time1, time2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
    enable_timing=True
)


def bwd_warp(H, W, K, world_points, src_imgs, src_poses, patch_H, patch_W):
    homo_world_points = torch.cat(
        [world_points, torch.ones(patch_H * patch_W, 1).to(world_points.device)], 1
    )[
        ..., None
    ]  # H*W, 4,1
    src_imgs = src_imgs.permute(0, 3, 1, 2)  # B,3,H,W
    homo_T = torch.cat(
        [src_poses, torch.zeros(src_imgs.shape[0], 1, 4)], dim=1
    )  # B,4,4
    homo_T[:, -1, -1] = 1
    inv_T = torch.inverse(homo_T)
    rect_points = torch.matmul(inv_T[:, None, :3, :], homo_world_points)
    # Rotate to cam coord
    rect_points[:, :, 1:] *= -1

    cam_points = torch.matmul(
        torch.from_numpy(K).to(rect_points.device)[None, None].float(), rect_points
    )
    cam_points[:, :, :2, :] /= cam_points[:, :, 2:, :] + 1e-7
    pix_coords = cam_points[:, :, :2, :].view(-1, patch_H, patch_W, 2)  # B,H,W,2
    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2

    warped_imgs = F.grid_sample(
        src_imgs.to(pix_coords.device),
        pix_coords,
        align_corners=False,
        padding_mode="border",
    )
    return warped_imgs


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    # i_range = Variable(torch.arange(h-1, -1, -1).view(1, h, 1).expand(1,h,w))#.type_as(depth)  # [1, H, W]
    i_range = Variable(
        torch.arange(0, h).view(1, h, 1).expand(1, h, w)
    )  # .type_as(depth)  # [1, H, W]
    j_range = Variable(
        torch.arange(0, w).view(1, 1, w).expand(1, h, w)
    )  # .type_as(depth)  # [1, H, W]
    ones = Variable(torch.ones(1, h, w)).type_as(i_range)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert all(condition), "wrong size for {}, expected {}, got  {}".format(
        input_name, "x".join(expected), list(input.size())
    )


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h or pixel_coords.size(3) < w:
        set_id_grid(depth)

    # Convert pixel locations to camera locations
    current_pixel_coords = (
        pixel_coords[:, :, :h, :w]
        .type_as(depth)
        .expand(b, 3, h, w)
        .contiguous()
        .view(b, 3, -1)
    )  # [B, 3, H*W]
    cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)

    # Weight these locations by the normalized depth? this means min depth of 1m max depth 1/beta = 100m?
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot.bmm(cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2]  # .clamp(min=1e-3)

    X_norm = (
        2 * (X / torch.abs(Z)) / (w - 1) - 1
    )  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]
    # X_norm = -X_norm # invert for image coordinates
    # Y_norm = -Y_norm # invert for image coordinates
    if padding_mode == "zeros":
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[
            X_mask
        ] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.view(b, h, w, 2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack(
        [cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], dim=1
    ).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack(
        [cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=1
    ).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack(
        [ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], dim=1
    ).view(B, 3, 3)

    rotMat = xmat.bmm(ymat).bmm(zmat)
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach() * 0 + 1, quat], dim=1)
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode="euler"):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == "euler":
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == "quat":
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def inverse_warp(
    img,
    depth,
    pose,
    intrinsics,
    intrinsics_inv,
    rotation_mode="euler",
    padding_mode="zeros",
):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    check_sizes(img, "img", "B3HW")
    check_sizes(depth, "depth", "BHW")
    check_sizes(pose, "pose", "B6")
    check_sizes(intrinsics, "intrinsics", "B33")
    check_sizes(intrinsics_inv, "intrinsics", "B33")

    assert intrinsics_inv.size() == intrinsics.size()

    batch_size, _, img_height, img_width = img.size()

    # Get world coordinates
    cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]

    # Get projection matrix for tgt camera frame to source pixel frame
    # In other words, get (R|t)
    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]

    src_pixel_coords = cam2pixel(
        cam_coords,
        proj_cam_to_src_pixel[:, :, :3],
        proj_cam_to_src_pixel[:, :, -1:],
        padding_mode,
    )  # [B,H,W,2]
    projected_img = torch.nn.functional.grid_sample(
        img, src_pixel_coords, padding_mode=padding_mode, align_corners=True
    )

    return projected_img


def inverse_warp_rt(img, depth, pose, intrinsics, intrinsics_inv, padding_mode="zeros"):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    check_sizes(img, "img", "B3HW")
    check_sizes(depth, "depth", "BHW")
    check_sizes(pose, "pose", "B34")
    check_sizes(intrinsics, "intrinsics", "B33")
    check_sizes(intrinsics_inv, "intrinsics", "B33")

    assert intrinsics_inv.size() == intrinsics.size()

    batch_size, _, img_height, img_width = img.size()

    # Get world coordinates
    cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]
    cam_coords[:, 1::, :, :] = -1 * cam_coords[:, 1::, :, :]  # make yz negative

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.bmm(pose)  # [B, 3, 4]

    src_pixel_coords = cam2pixel(
        cam_coords,
        proj_cam_to_src_pixel[:, :, :3],
        proj_cam_to_src_pixel[:, :, -1:],
        padding_mode,
    )  # [B,H,W,2]
    projected_img = torch.nn.functional.grid_sample(
        img, src_pixel_coords, padding_mode=padding_mode, align_corners=True
    )

    return projected_img


def inverse_warp_rt1_rt2(
    img, depth, c2w1, c2w2, intrinsics, intrinsics_inv, padding_mode="zeros"
):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    B, H, W = depth.shape

    R1 = c2w1[:, :, 0:3]
    t1 = c2w1[:, :, 3, None]
    R2 = c2w2[:, :, 0:3]
    t2 = c2w2[:, :, 3, None]
    R2_ = torch.transpose(R2, 2, 1)
    t2_ = -torch.bmm(R2_, t2)

    # 1. Lift into 3D cam coordinates, pixel coordinates is p=[u,v,1]: c1 = D1 * K_invp1
    c1 = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]
    c1[:, 1::, :, :] *= -1  # make z negative

    # 2. Get world coordinates from c1: w = R1c1 + t1
    w = torch.bmm(R1, c1.view(B, 3, -1)) + t1

    # 3. Get camera coordinates in c2: c2 = R2'w + (-R2't2)
    c2 = torch.bmm(R2_, w) + t2_

    # 4. Get pixel coordinates in c2: p2 = Kc2 / c2[z]
    z = torch.abs(c2[:, 2, None, :])
    c2_ = c2 / z
    c2_[:, 2, :] = 1
    c2_[:, 1, :] *= -1
    p2 = torch.bmm(intrinsics, c2_)

    X = p2[:, 0]
    Y = p2[:, 1]
    X_norm = (
        2 * X / (W - 1) - 1
    )  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * Y / (H - 1) - 1  # Idem [B, H*W]

    if padding_mode == "zeros":
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[
            X_mask
        ] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    src_pixel_coords = pixel_coords.view(B, H, W, 2)
    projected_img = torch.nn.functional.grid_sample(
        img, src_pixel_coords, padding_mode=padding_mode, align_corners=True
    )

    return projected_img


def inverse_warp_rod1_rt2(
    img, depth, ro1, rd1, c2w2, intrinsics, intrinsics_inv, padding_mode="zeros"
):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    B, H, W = depth.shape

    R2 = c2w2[:, :, 0:3]
    t2 = c2w2[:, :, 3, None]
    R2_ = torch.transpose(R2, 2, 1)
    t2_ = -torch.bmm(R2_, t2)

    # 1. Lift directly into 3D world coordinates [B, 3, H*W]
    w = ro1 + rd1 * depth.view(B, 1, -1)

    # 3. Get camera coordinates in c2: c2 = R2'w + (-R2't2)
    c2 = torch.bmm(R2_, w) + t2_

    # 4. Get pixel coordinates in c2: p2 = Kc2 / c2[z]
    z = torch.abs(c2[:, 2, None, :])
    c2_ = c2 / z
    c2_[:, 2, :] = 1
    c2_[:, 1, :] *= -1
    p2 = torch.bmm(intrinsics, c2_)

    X = p2[:, 0]
    Y = p2[:, 1]
    X_norm = (
        2 * X / (W - 1) - 1
    )  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * Y / (H - 1) - 1  # Idem [B, H*W]

    if padding_mode == "zeros":
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[
            X_mask
        ] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    src_pixel_coords = pixel_coords.view(B, H, W, 2)
    projected_img = torch.nn.functional.grid_sample(
        img, src_pixel_coords, padding_mode=padding_mode, align_corners=True
    )

    return projected_img


def inverse_warp_rod1_rt2_v2(
    img, depth, points, c2w2, intrinsics, padding_mode="zeros"
):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    B, H, W = depth.shape

    R2 = c2w2[:, :, 0:3]
    t2 = c2w2[:, :, 3, None]
    R2_ = torch.transpose(R2, 2, 1)
    t2_ = -torch.bmm(R2_, t2)

    # 3. Get camera coordinates in c2: c2 = R2'w + (-R2't2)
    c2 = torch.bmm(R2_, points) + t2_

    # 4. Get pixel coordinates in c2: p2 = Kc2 / c2[z]
    z = torch.abs(c2[:, 2, None, :])
    c2_ = c2 / z
    c2_[:, 2, :] = 1
    c2_[:, 1, :] *= -1
    p2 = torch.bmm(intrinsics, c2_)

    X = p2[:, 0]
    Y = p2[:, 1]
    X_norm = (
        2 * X / (W - 1) - 1
    )  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * Y / (H - 1) - 1  # Idem [B, H*W]

    if padding_mode == "zeros":
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[
            X_mask
        ] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    src_pixel_coords = pixel_coords.view(B, H, W, 2)
    projected_img = torch.nn.functional.grid_sample(
        img, src_pixel_coords, padding_mode=padding_mode, align_corners=True
    )

    return projected_img


def inverse_warp_rod1_rt2_coords_patch(
    img,
    depth,
    ro1,
    rd1,
    c2w2,
    intrinsics,
    intrinsics_inv,
    scale=1.0,
    padding_mode="zeros",
):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    B, H, W = depth.shape
    _, C, Hfull, Wfull = img.shape

    R2 = c2w2[:, :, 0:3]
    t2 = c2w2[:, :, 3, None]
    R2_ = torch.transpose(R2, 2, 1)
    t2_ = -torch.bmm(R2_, t2)

    # 1. Lift directly into 3D world coordinates [B, 3, H*W]
    w = ro1 + rd1 * depth.view(B, 1, -1)

    # 3. Get camera coordinates in c2: c2 = R2'w + (-R2't2)
    c2 = torch.bmm(R2_, w) + t2_

    # 4. Get pixel coordinates in c2: p2 = Kc2 / c2[z]
    z = torch.abs(c2[:, 2, None, :])
    c2_ = c2 / (z + 1e-6)
    c2_[:, 2, :] = 1
    c2_[:, 1, :] *= -1
    p2 = torch.bmm(intrinsics, c2_)

    X = p2[:, 0][..., None].repeat(1, 1, 3)
    Y = p2[:, 1][..., None].repeat(1, 1, 3)

    # convolve window
    X[:, :, 0] = X[:, :, 0] - 1
    X[:, :, 2] = X[:, :, 2] + 1

    Y[:, :, 0] = Y[:, :, 0] - 1
    Y[:, :, 2] = Y[:, :, 2] + 1

    X = X[..., None].expand(-1, -1, -1, 3)
    Y = Y[:, :, None, :].expand(-1, -1, 3, -1)

    X_norm = (
        2 * X / (Wfull - 1) - 1
    )  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * Y / (Hfull - 1) - 1  # Idem [B, H*W]

    # unnorm_pixel_coords = torch.stack([X, Y], dim=2)
    # valid_mask = inbound(unnorm_pixel_coords, h=Hfull, w = Wfull).view(B,H,W)

    # if padding_mode == 'zeros':
    #     X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
    #     X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
    #     Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
    #     Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=-1)  # [B, H*W, 2]
    src_pixel_coords = (
        pixel_coords.view(B, H, W, 9, 2).permute(3, 0, 1, 2, 4).reshape(9 * B, H, W, 2)
    )

    if scale != 1:
        src_pixel_coords = torch.nn.functional.interpolate(
            torch.permute(src_pixel_coords, (0, 3, 1, 2)),
            size=(int(scale * H), int(scale * W)),
            mode="bilinear",
            align_corners=True,
        )
        src_pixel_coords = torch.permute(src_pixel_coords, (0, 2, 3, 1))
        img = torch.nn.functional.interpolate(
            img,
            size=(int(scale * H), int(scale * W)),
            mode="bilinear",
            align_corners=True,
        )
        projected_img = torch.nn.functional.grid_sample(
            img, src_pixel_coords, padding_mode=padding_mode, align_corners=True
        )
        projected_img = torch.nn.functional.interpolate(
            projected_img, size=(H, W), mode="bilinear", align_corners=True
        )
    else:
        projected_img = torch.nn.functional.grid_sample(
            img[None].expand(9, -1, -1, -1, -1).reshape(9 * B, 3, Hfull, Wfull),
            src_pixel_coords,
            padding_mode=padding_mode,
            align_corners=True,
        )
        projected_img = (
            projected_img.view(9, B, 3, H, W)
            .permute(1, 0, 2, 3, 4)
            .reshape(B, 27, H, W)
        )

    return projected_img, None


def inverse_warp_rod1_rt2_coords(
    img,
    depth,
    ro1,
    rd1,
    c2w2,
    intrinsics,
    intrinsics_inv,
    scale=1.0,
    padding_mode="zeros",
):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    B, H, W = depth.shape
    _, C, Hfull, Wfull = img.shape

    R2 = c2w2[:, :, 0:3]
    t2 = c2w2[:, :, 3, None]
    R2_ = torch.transpose(R2, 2, 1)
    t2_ = -torch.bmm(R2_, t2)

    # 1. Lift directly into 3D world coordinates [B, 3, H*W]
    w = ro1 + rd1 * depth.view(B, 1, -1)

    # 3. Get camera coordinates in c2: c2 = R2'w + (-R2't2)
    c2 = torch.bmm(R2_, w) + t2_

    # 4. Get pixel coordinates in c2: p2 = Kc2 / c2[z]
    z = torch.abs(c2[:, 2, None, :])
    c2_ = c2 / z
    c2_[:, 2, :] = 1
    c2_[:, 1, :] *= -1
    p2 = torch.bmm(intrinsics, c2_)

    X = p2[:, 0]
    Y = p2[:, 1]

    X_norm = (
        2 * X / (Wfull - 1) - 1
    )  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * Y / (Hfull - 1) - 1  # Idem [B, H*W]

    # unnorm_pixel_coords = torch.stack([X, Y], dim=2)
    # valid_mask = inbound(unnorm_pixel_coords, h=Hfull, w = Wfull).view(B,H,W)

    if padding_mode == "zeros":
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[
            X_mask
        ] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    src_pixel_coords = pixel_coords.view(B, H, W, 2)

    if scale != 1:
        src_pixel_coords = torch.nn.functional.interpolate(
            torch.permute(src_pixel_coords, (0, 3, 1, 2)),
            size=(int(scale * H), int(scale * W)),
            mode="bilinear",
            align_corners=True,
        )
        src_pixel_coords = torch.permute(src_pixel_coords, (0, 2, 3, 1))
        img = torch.nn.functional.interpolate(
            img,
            size=(int(scale * H), int(scale * W)),
            mode="bilinear",
            align_corners=True,
        )
        projected_img = torch.nn.functional.grid_sample(
            img, src_pixel_coords, padding_mode=padding_mode, align_corners=True
        )
        projected_img = torch.nn.functional.interpolate(
            projected_img, size=(H, W), mode="bilinear", align_corners=True
        )
    else:
        projected_img = torch.nn.functional.grid_sample(
            img, src_pixel_coords, padding_mode=padding_mode, align_corners=True
        )

    return projected_img, None


# @profile
def inverse_warp_rod1_rt2_coords_trt(
    img, depth, ro1, rd1, w2c, scale=1.0, padding_mode="zeros"
):
    """
    Inverse warp a source image to the target image plane. Fast versiion: we compute K*I'*[R|t] from the beginning

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    B, H, W = depth.shape
    _, C, Hfull, Wfull = img.shape

    w = ro1 + rd1 * depth.view(B, 1, -1)
    p2 = torch.bmm(w2c, w)

    p2[:, :2, :] /= p2[:, 2:, :]
    X = p2[:, 0]
    Y = p2[:, 1]

    X_norm = (
        2 * X / (Wfull - 1) - 1
    )  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * Y / (Hfull - 1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    src_pixel_coords = pixel_coords.view(B, H, W, 2)

    # time1.record()
    projected_img = torch.nn.functional.grid_sample(
        img, src_pixel_coords, padding_mode=padding_mode, align_corners=True
    )
    # time2.record()
    # torch.cuda.synchronize()
    # print('warp 3c:', time1.elapsed_time(time2))
    return projected_img, None


# @profile
def inverse_warp_rod1_rt2_coords_trt_1c(
    img, depth, ro1, rd1, w2c, scale=1.0, padding_mode="zeros"
):
    """
    Inverse warp a source image to the target image plane. Fast versiion: we compute K*I'*[R|t] from the beginning

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    B, H, W = depth.shape
    _, C, Hfull, Wfull = img.shape
    # 1. Lift directly into 3D world coordinates [B, 3, H*W]

    w = ro1 + rd1 * depth.view(B, 1, -1)

    p2 = torch.bmm(w2c, w)

    p2[:, :2, :] /= p2[:, 2:, :]
    X = p2[:, 0]
    Y = p2[:, 1]

    X_norm = (
        2 * X / (Wfull - 1) - 1
    )  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * Y / (Hfull - 1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    src_pixel_coords = pixel_coords.view(B, H, W, 2)
    # time1.record()
    projected_img = torch.nn.functional.grid_sample(
        img,
        src_pixel_coords,
        padding_mode=padding_mode,
        mode="nearest",
        align_corners=True,
    )
    # time2.record()
    # torch.cuda.synchronize()
    # print('warp 1c:', time1.elapsed_time(time2))

    imgB = projected_img // (2**16)
    resd = projected_img % (2**16)
    imgG = resd // (2**8)
    imgR = resd % (2**8)
    projected_img = torch.cat([imgR, imgG, imgB], 1) / 255.0

    return projected_img, None


def inverse_warp_rod1_rt2_coords_feat(
    img,
    feat,
    depth,
    ro1,
    rd1,
    c2w2,
    intrinsics,
    intrinsics_inv,
    scale=1.0,
    padding_mode="zeros",
):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    B, H, W = depth.shape
    _, C, Hfull, Wfull = img.shape

    R2 = c2w2[:, :, 0:3]
    t2 = c2w2[:, :, 3, None]
    R2_ = torch.transpose(R2, 2, 1)
    t2_ = -torch.bmm(R2_, t2)

    # 1. Lift directly into 3D world coordinates [B, 3, H*W]
    w = ro1 + rd1 * depth.view(B, 1, -1)

    # 3. Get camera coordinates in c2: c2 = R2'w + (-R2't2)
    c2 = torch.bmm(R2_, w) + t2_

    # 4. Get pixel coordinates in c2: p2 = Kc2 / c2[z]
    z = torch.abs(c2[:, 2, None, :])
    c2_ = c2 / z
    c2_[:, 2, :] = 1
    c2_[:, 1, :] *= -1
    p2 = torch.bmm(intrinsics, c2_)

    X = p2[:, 0]
    Y = p2[:, 1]
    X_norm = (
        2 * X / (Wfull - 1) - 1
    )  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * Y / (Hfull - 1) - 1  # Idem [B, H*W]

    unnorm_pixel_coords = torch.stack([X, Y], dim=2)
    valid_mask = inbound(unnorm_pixel_coords, h=Hfull, w=Wfull).view(B, H, W)

    if padding_mode == "zeros":
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[
            X_mask
        ] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    src_pixel_coords = pixel_coords.view(B, H, W, 2)

    if scale != 1:
        src_pixel_coords = torch.nn.functional.interpolate(
            torch.permute(src_pixel_coords, (0, 3, 1, 2)),
            size=(int(scale * H), int(scale * W)),
            mode="bilinear",
            align_corners=True,
        )
        src_pixel_coords = torch.permute(src_pixel_coords, (0, 2, 3, 1))
        img = torch.nn.functional.interpolate(
            img,
            size=(int(scale * H), int(scale * W)),
            mode="bilinear",
            align_corners=True,
        )
        projected_img = torch.nn.functional.grid_sample(
            img, src_pixel_coords, padding_mode=padding_mode, align_corners=True
        )
        projected_img = torch.nn.functional.interpolate(
            projected_img, size=(H, W), mode="bilinear", align_corners=True
        )
    else:
        projected_img = torch.nn.functional.grid_sample(
            img, src_pixel_coords, padding_mode=padding_mode, align_corners=True
        )
        projected_feat = torch.nn.functional.grid_sample(
            feat, src_pixel_coords, padding_mode=padding_mode, align_corners=True
        )

    return projected_img, projected_feat, valid_mask


def inbound(pixel_locations, h, w):
    """
    check if the pixel locations are in valid range
    :param pixel_locations: [..., 2]
    :param h: height
    :param w: weight
    :return: mask, bool, [...]
    """
    return (
        (pixel_locations[..., 0] <= w - 1.0)
        & (pixel_locations[..., 0] >= 0)
        & (pixel_locations[..., 1] <= h - 1.0)
        & (pixel_locations[..., 1] >= 0)
    )
