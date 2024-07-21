import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# DataAugmentation
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


class IdolData(Dataset):
    def __init__(self, data_dir, member_list, transform, split, aug_dir=None):
        self.data_dir = data_dir
        self.transform = transform
        with open(member_list, 'r', encoding='utf-8') as f:
            self.classes = f.read().splitlines()
        self.num_classes = len(self.classes)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.images = []
        self.labels = []
        for class_name in self.classes:
            img_names = os.listdir(os.path.join(data_dir, class_name))
            img_names.sort()
            img_paths = [os.path.join(data_dir, class_name, img_name) for img_name in img_names if img_name != '.DS_Store']
            N = len(img_names)
            train_size = int(N * 0.8)
            val_size = int(N * 0.1)
            test_size = N - train_size - val_size
            if split == 'train':
                if aug_dir == None:
                    self.images.extend(img_paths[:train_size])
                    self.labels.extend([self.class_to_idx[class_name]]*train_size)
                else:
                    aug_img_paths = [os.path.join(aug_dir, class_name, img_name) for img_name in img_names if img_name != '.DS_Store']
                    for aug_img_path in aug_img_paths[:train_size]:
                        for i in range(32):
                            self.images.append(aug_img_path.replace('.jpg', f'_{i}.jpg'))
                        self.labels.extend([self.class_to_idx[class_name]]*32)
                        
            elif split == 'val':
                self.images.extend(img_paths[train_size:train_size+val_size])
                self.labels.extend([self.class_to_idx[class_name]]*val_size)
            else:
                self.images.extend(img_paths[train_size+val_size:])
                self.labels.extend([self.class_to_idx[class_name]]*test_size)
        print(f'{split}: {len(self.images)}')
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        label = self.labels[idx]
        return image, label