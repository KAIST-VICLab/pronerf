import numpy as np
import os
import torch
from imageio.v2 import imread

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2mse_np = lambda x, y : np.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
mse2psnr_np = lambda x : -10. * np.log(x) / np.log([10.])
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

dir = 'E:\juan\code\MinMaxNeRF\logs_minmax\dumb'
img1 = '000.png'
img2 = '001.png'
img3 = '002.png'

img1 = imread(os.path.join(dir, img1)) / 255
img2 = imread(os.path.join(dir, img2)) / 255
img3 = imread(os.path.join(dir, img3)) / 255
# print(img1)

tgt_imgs = np.load(os.path.join(dir, 'test_imgs.npy'))
print(tgt_imgs.shape)

mean_psnr = mse2psnr_np(img2mse_np(img1, tgt_imgs[0])) + mse2psnr_np(img2mse_np(img2, tgt_imgs[1])) + \
            mse2psnr_np(img2mse_np(img3, tgt_imgs[2]))

print(mse2psnr_np(img2mse_np(img1, tgt_imgs[0])))
print(mse2psnr_np(img2mse_np(img2, tgt_imgs[1])))
print(mse2psnr_np(img2mse_np(img3, tgt_imgs[2])))

print(mean_psnr / 3)