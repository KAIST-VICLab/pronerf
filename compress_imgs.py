import cv2
import os
import glob
from tqdm import tqdm

scenes = ["fern", "fortress","flower","horns","leaves","orchids","room", "trex"]
root_dir = "./data/nerf_llff_data"
factor = 8
quality_rate = 75

for scene in tqdm(scenes):
    img_dir = os.path.join(root_dir, scene, 'images_{}'.format(str(factor)))
    img_paths = glob.glob(os.path.join(img_dir, '*.png'))
    compress_path = os.path.join(root_dir, scene, 'c{}_images_{}'.format(str(quality_rate),str(factor)))
    if not os.path.exists(compress_path):
        os.makedirs(compress_path)
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img_name = img_path.split('/')[-1]
        cv2.imwrite(os.path.join(compress_path,img_name.replace('png', 'jpg')), img, [cv2.IMWRITE_JPEG_QUALITY, quality_rate])