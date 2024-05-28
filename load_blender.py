import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

# from sklearn.cluster import KMeans


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    render_poses = []
    for phi in np.linspace(-17,-85,5):
        for theta in np.linspace(-180,180,40+1)[:-1]:
            render_poses.append(pose_spherical(theta, phi, 4.0))
    render_poses = torch.stack(render_poses, 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split

def load_blender_data_infer(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    # render_poses = []
    # for phi in np.linspace(-17,-85,5):
    #     for theta in np.linspace(-180,180,40+1)[:-1]:
    #         render_poses.append(pose_spherical(theta, phi, 4.0))
    # render_poses = torch.stack(render_poses, 0)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()


    # # init hemicube
    # cube_size = 6.0
    # cube_center = np.array([0,0,0]).astype(np.float32)
    # cube_resolution = 20
    # cube_points = []

    # # top face
    # x, y = np.meshgrid(np.linspace(0,1,cube_resolution + 1, dtype=np.float32), np.linspace(0,1,cube_resolution + 1, dtype=np.float32), indexing='xy')
    # x = x*cube_size - cube_size / 2
    # y = y*cube_size - cube_size / 2
    # z = np.ones_like(x)*cube_size / 2
    # corner_points = np.stack([x,y,z], axis = -1)
    # mid_points = (corner_points[1:,...] + corner_points[:-1,...])/2
    # mid_points = (mid_points[:,1:,:] + mid_points[:,:-1,:])/2
    # cube_points.append(mid_points.reshape(-1,3))

    # # side face
    # z, y = np.meshgrid(np.linspace(0,1,cube_resolution + 1, dtype=np.float32), np.linspace(0,1,cube_resolution + 1, dtype=np.float32), indexing='xy')
    # z = z*cube_size - cube_size / 2
    # y = y*cube_size - cube_size / 2
    # x = np.ones_like(z)*cube_size / 2 # x > 0
    # corner_points = np.stack([x,y,z], axis = -1)
    # mid_points = (corner_points[1:,...] + corner_points[:-1,...])/2
    # mid_points = (mid_points[:,1:,:] + mid_points[:,:-1,:])/2
    # cube_points.append(mid_points[mid_points[:,:,-1] >= 0])

    # z, y = np.meshgrid(np.linspace(0,1,cube_resolution + 1, dtype=np.float32), np.linspace(0,1,cube_resolution + 1, dtype=np.float32), indexing='xy')
    # z = z*cube_size - cube_size / 2
    # y = y*cube_size - cube_size / 2
    # x = -np.ones_like(z)*cube_size / 2 # x > 0
    # corner_points = np.stack([x,y,z], axis = -1)
    # mid_points = (corner_points[1:,...] + corner_points[:-1,...])/2
    # mid_points = (mid_points[:,1:,:] + mid_points[:,:-1,:])/2
    # cube_points.append(mid_points[mid_points[:,:,-1] >= 0])

    # z, x = np.meshgrid(np.linspace(0,1,cube_resolution + 1, dtype=np.float32), np.linspace(0,1,cube_resolution + 1, dtype=np.float32), indexing='xy')
    # z = z*cube_size - cube_size / 2
    # x = x*cube_size - cube_size / 2
    # y = np.ones_like(z)*cube_size / 2 # x > 0
    # corner_points = np.stack([x,y,z], axis = -1)
    # mid_points = (corner_points[1:,...] + corner_points[:-1,...])/2
    # mid_points = (mid_points[:,1:,:] + mid_points[:,:-1,:])/2
    # cube_points.append(mid_points[mid_points[:,:,-1] >= 0])

    # z, x = np.meshgrid(np.linspace(0,1,cube_resolution + 1, dtype=np.float32), np.linspace(0,1,cube_resolution + 1, dtype=np.float32), indexing='xy')
    # z = z*cube_size - cube_size / 2
    # x = x*cube_size - cube_size / 2
    # y = -np.ones_like(z)*cube_size / 2 # x > 0
    # corner_points = np.stack([x,y,z], axis = -1)
    # mid_points = (corner_points[1:,...] + corner_points[:-1,...])/2
    # mid_points = (mid_points[:,1:,:] + mid_points[:,:-1,:])/2
    # cube_points.append(mid_points[mid_points[:,:,-1] >= 0])

    # cube_points = np.concatenate(cube_points, axis =0)
    # visibilities = np.zeros((len(i_split[0]), cube_points.shape[0]))

    # K = np.array([
    #     [focal, 0, 0.5 * W],
    #     [0, focal, 0.5 * H],
    #     [0, 0, 1]
    # ])
    # i_train = i_split[0]

    # # hemi cube has 5 planes: z=2, x=-2, x=2, y=-2, y=2
    # occ_matrix = np.ones((5, cube_points.shape[0]))*100

    # # compute visbible matrix
    # for cam_i in i_split[0]:
    #     c2w2 = poses[cam_i]

    #     # occlusion test with 5 plane
    #     cam_origin = c2w2[:3,3] 
    #     c2p_dir = cube_points - cam_origin[None]

    #     # test z = 2
    #     t = (2.0 - cam_origin[-1]) / c2p_dir[:,-1]
    #     intersect_pts = cam_origin[None] + t[:,None]*c2p_dir
    #     intersect_id = (intersect_pts[:,0] >= -2) & (intersect_pts[:,0] <= 2) & (intersect_pts[:,1] >= -2) & (intersect_pts[:,1] <= 2)
    #     occ_matrix[0][intersect_id] = t[intersect_id]

    #     # test x = 2
    #     t = (2.0 - cam_origin[0]) / c2p_dir[:,0]
    #     intersect_pts = cam_origin[None] + t[:,None]*c2p_dir
    #     intersect_id = (intersect_pts[:,-1] >= 0) & (intersect_pts[:,-1] <= 2) & (intersect_pts[:,1] >= -2) & (intersect_pts[:,1] <= 2)
    #     occ_matrix[1][intersect_id] = t[intersect_id]

    #     # test x = -2
    #     t = (-2.0 - cam_origin[0]) / c2p_dir[:,0]
    #     intersect_pts = cam_origin[None] + t[:,None]*c2p_dir
    #     intersect_id = (intersect_pts[:,-1] >= 0) & (intersect_pts[:,-1] <= 2) & (intersect_pts[:,1] >= -2) & (intersect_pts[:,1] <= 2)
    #     occ_matrix[2][intersect_id] = t[intersect_id]

    #     # test y = 2
    #     t = (2.0 - cam_origin[1]) / c2p_dir[:,1]
    #     intersect_pts = cam_origin[None] + t[:,None]*c2p_dir
    #     intersect_id = (intersect_pts[:,-1] >= 0) & (intersect_pts[:,-1] <= 2) & (intersect_pts[:,0] >= -2) & (intersect_pts[:,0] <= 2)
    #     occ_matrix[3][intersect_id] = t[intersect_id]

    #     # test y = -2
    #     t = (-2.0 - cam_origin[1]) / c2p_dir[:,1]
    #     intersect_pts = cam_origin[None] + t[:,None]*c2p_dir
    #     intersect_id = (intersect_pts[:,-1] >= 0) & (intersect_pts[:,-1] <= 2) & (intersect_pts[:,0] >= -2) & (intersect_pts[:,0] <= 2)
    #     occ_matrix[4][intersect_id] = t[intersect_id]

    #     occ_mask = occ_matrix.min(axis = 0) < 1.0

    #     # projects cube points onto camera
    #     R_w2c = c2w2[:3, 0:3].T

    #     t_c2w = c2w2[:3, 3, None]
    #     t_w2c = -np.matmul(R_w2c, t_c2w)
    #     cam_points = np.matmul(R_w2c[None], cube_points[...,None]) + t_w2c[None]
    #     cam_points_z = np.abs(cam_points[:, 2, None, :])
    #     cam_points_ = cam_points / cam_points_z
    #     cam_points_[:, 2, :] = 1
    #     cam_points_[:, 1, :] *= -1
    #     p2 = np.matmul(K[None], cam_points_)

    #     X = p2[:, 0]
    #     Y = p2[:, 1]

    #     X_norm = X / (W - 1)
    #     Y_norm = Y / (H - 1)

    #     mask = (X_norm >= 0) & (X_norm <= 1) & (Y_norm >= 0) & (Y_norm <= 1)

    #     mask = mask.squeeze() & (~occ_mask) # visible and not occluded
    #     visibilities[cam_i][mask] = 1
        
    #     # reset occ matrix
    #     occ_matrix = np.ones((5, cube_points.shape[0]))*100

    num_neighbor = 40
    # raw_i_ref = []


    # for _ in range(num_neighbor):
    #     total_visible_points = visibilities.sum(-1)
    #     most_visible = np.argmax(total_visible_points)
        
    #     raw_i_ref.append(most_visible)
    #     num_points = total_visible_points[most_visible]
    #     if total_visible_points[most_visible] <= 0:
    #         print('warnings 0 point found !')
    #         breakpoint()
    #     print('Choose img {} with {} points'.format(i_train[most_visible], total_visible_points[most_visible]))

    #     # update visible list
    #     visibilities = visibilities - visibilities[most_visible][None]
    #     visibilities[visibilities < 0] = 0
    # print('Total ref views: {}/{}'.format(len(raw_i_ref), len(i_train)))


    i_train = i_split[0]
    cam_origins = poses[i_train,:3,3]
    kmeans = KMeans(n_clusters=num_neighbor).fit(cam_origins)
    centers = kmeans.cluster_centers_

    cluster_distance = ((cam_origins[:,None,:] - centers[None])**2).mean(-1)
    cam_index = np.argmin(cluster_distance, axis =0)
    i_ref = cam_index.tolist()
    return imgs, poses, render_poses, [H, W, focal], i_split, i_ref


