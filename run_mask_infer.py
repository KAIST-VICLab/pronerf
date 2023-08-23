import os
import sys

gpu_n = '6'
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
import math
import lpips

import matplotlib.pyplot as plt
from run_nerf_helpers import *
import glob

scene = "flower"
iter = "490000"
iter_ibr = "500000"

nerf_path = "logs_epi_RR/{}_refine_8samples_v2/renderonly_test_{}".format(scene, iter)
ibr_path = "logs_epi_RR/{}_Nn4_Ns8_Savr_eSavr/renderonly_test_{}".format(scene, iter_ibr)
mask_path = "logs_epi_RR/{}_sampler_e2e_donerf_8samples_mask_50/testset_100000".format(scene)

nerf_psnrs, ibr_psnrs, hybrid_psnrs = [],[],[]
compress_ratios = []

num_imgs = len(glob.glob(os.path.join(ibr_path, 'gt_*.png')))
for i in range(num_imgs):
    img_id = str(i).zfill(3)
    nerf_img = imageio.imread(os.path.join(nerf_path, '{}.png'.format(img_id)))/255.0
    ibr_img = imageio.imread(os.path.join(ibr_path, '{}.png'.format(img_id)))/255.0
    mask_img = imageio.imread(os.path.join(mask_path, '{}.png'.format(img_id)))/255.0
    gt_img = imageio.imread(os.path.join(ibr_path, 'gt_{}.png'.format(img_id)))/255.0

    mask_img = mask_img[...,None]
    mask_img[mask_img > 0.6] = 1.0
    mask_img[mask_img <= 0.6] = 0.0

    hybrid_img = ibr_img*mask_img + nerf_img*(1-mask_img)

    rgb8 = to8b(hybrid_img)
    filename = os.path.join(mask_path, 'hybrid_{:03d}.png'.format(i))
    imageio.imwrite(filename, rgb8)
    
    p = mse2psnr(img2mse(torch.from_numpy(nerf_img), torch.from_numpy(gt_img)))
    nerf_psnrs.append(p)

    p = mse2psnr(img2mse(torch.from_numpy(ibr_img), torch.from_numpy(gt_img)))
    ibr_psnrs.append(p)

    p = mse2psnr(img2mse(torch.from_numpy(hybrid_img), torch.from_numpy(gt_img)))
    hybrid_psnrs.append(p)

    compress_ratios.append(mask_img.sum()/(756*1008))

print(np.array(nerf_psnrs).mean(), np.array(ibr_psnrs).mean(), np.array(hybrid_psnrs).mean())
print(np.array(compress_ratios).mean())
