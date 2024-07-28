import os
import sys
import cv2
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageOps


# python preprocess/data_aug.py
INDIR = './data/twitter/images/face'
OUTDIR = './data/twitter_aug4/images/face'


# 1. 回転 (Rotation)
def rotation(image):
    angle = random.uniform(-15, 15)
    rotated_image = image.rotate(angle)
    return rotated_image

# 2. 平行移動 (Translation)
def translation(image):
    width, height = image.size
    delta = min(width, height) // 4
    tx, ty = random.randint(-delta, delta), random.randint(-delta, delta)
    M = (1, 0, tx, 0, 1, ty)
    translated_image = image.transform((width, height), Image.AFFINE, M)
    return translated_image

# 3. スケーリング (Scaling)
def scaling(image, min_size=20):
    min_scale = min_size / min(image.width, image.height)
    scale_factor = random.uniform(min_scale, 1.0)
    scaled_image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)))
    return scaled_image

# 4. フリップ (Flip)
def flip(image):
    flipped_image = ImageOps.mirror(image)
    return flipped_image

# 5. ノイズの追加 (Adding Noise)
def adding_noise(image):
    image_array = np.array(image)
    noise = np.random.normal(0, 25, image_array.shape).astype(np.uint8)
    noisy_image_array = image_array + noise
    noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_image_array)
    return noisy_image

# 6. カラージッタリング (Color Jittering)
def color_jittering(image):
    color_enhancer = ImageEnhance.Color(image)
    color_jittered_image = color_enhancer.enhance(random.uniform(0.7, 1.3))
    return color_jittered_image

# 7. カットアウト (Cutout)
def cutout(image):
    cutout_size = min(image.width, image.height) // 4 # (30, 30)
    x = random.randint(0, image.width - cutout_size)
    y = random.randint(0, image.height - cutout_size)
    cutout_image = image.copy()
    cutout_area = Image.new("RGB", (cutout_size, cutout_size), (0, 0, 0))
    cutout_image.paste(cutout_area, (x, y))
    return cutout_image


# num -> aug_function
aug_functions = {
    0: rotation,
    1: translation,
    2: scaling,
    3: flip,
    4: adding_noise,
    5: color_jittering,
    6: cutout
}
# データ拡張1つ
def augment_image1(image_path, auglen=7):
    aug_images = []
    for i in range(auglen):
        image = Image.open(image_path)
        aug_images.append(aug_functions[i](image))
    return aug_images
# データ拡張2つ
def augment_image2(image_path, auglen=7):
    aug_images = []
    for i in range(auglen):
        for j in range(i+1, auglen):
            list_num = [i, j]
            image = Image.open(image_path)
            for num in list_num:
                image = aug_functions[num](image)
            aug_images.append(image)
    return aug_images
# データ拡張複数回
def augment_image3(image_path, auglen=7, times=5):
    aug_images = []
    for i in range(auglen):
        for j in range(times):
            image = Image.open(image_path)
            image = aug_functions[i](image)
            aug_images.append(image)
            if i == 3: break
    return aug_images

def main():
    
    # member
    member_names = os.listdir(INDIR)
    member_names = [member_name for member_name in member_names if member_name != '.DS_Store']
    
    # data aug
    for member_name in tqdm(member_names):
        in_member_dir = os.path.join(INDIR, member_name)
        out_member_dir = os.path.join(OUTDIR, member_name)
        os.makedirs(out_member_dir, exist_ok=True)
        img_names = os.listdir(in_member_dir)
        img_names = [img_name for img_name in img_names if img_name != '.DS_Store']
        for img_name in img_names:
            img_path = os.path.join(in_member_dir, img_name)
            inflated_images = augment_image3(img_path)
            for i, im in enumerate(inflated_images):
                im.save(os.path.join(out_member_dir, img_name.replace('.jpg', f'_{i}.jpg')))


if __name__ == '__main__':
    main()