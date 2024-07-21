import os
import cv2
import numpy as np
from tqdm import tqdm


# python preprocess/data_augmentation.py
INDIR = './data/twitter/images/face'
OUTDIR = './data/twitter_reshape/images/face'


def inflated_image(img, flip=True, thr=True, filt=True, resize=True, erode=True):
    methods = [flip, thr, filt, resize, erode]
    img_size = img.shape
    mat = cv2.getRotationMatrix2D(tuple(np.array([img_size[1], img_size[0]]) / 2 ), 45, 1.0)
    filter1 = np.ones((3, 3))
    images = [img]
    scratch = np.array([
        lambda x: cv2.flip(x, 1),                                                                           # 左右反転
        lambda x: cv2.threshold(x, 100, 255, cv2.THRESH_TOZERO)[1],                                         # 閾値処理
        lambda x: cv2.GaussianBlur(x, (5, 5), 0),                                                           # ぼかし
        lambda x: cv2.resize(cv2.resize(x, (img_size[1]//6, img_size[0]//6)), (img_size[1], img_size[0])),  # モザイク処理
        lambda x: cv2.erode(x, filter1)                                                                     # 収縮
    ])
    doubling_images = lambda f, imag: (imag + [f(i) for i in imag])
    for func in scratch[methods]:
        images = doubling_images(func, images)
    return images


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
            img = cv2.imread(img_path)
            inflated_images = inflated_image(img)
            for i, im in enumerate(inflated_images):
                cv2.imwrite(os.path.join(out_member_dir, img_name.replace('.jpg', f'_{i}.jpg')), im)


if __name__ == '__main__':
    main()