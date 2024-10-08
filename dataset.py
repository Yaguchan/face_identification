import os
import cv2
import torch
import numpy as np
from PIL import Image
from utils import pil_img_resize
from torch.utils.data import Dataset


class IdolData(Dataset):
    def __init__(self, data_dir, member_list, transform, split, aug_dir=None, aug_size=None):
        self.data_dir = data_dir
        self.transform = transform
        with open(member_list, 'r', encoding='utf-8') as f:
            self.classes = f.read().splitlines()
        self.num_classes = len(self.classes)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.images = []
        self.labels = []
        list_len = []
        # ダウンサンプリング
        for class_name in self.classes:
            img_names = os.listdir(os.path.join(data_dir, class_name))
            img_names = [img_name for img_name in img_names if img_name != '.DS_Store']
            list_len.append(len(img_names))
        N = min(list_len)
        for class_name in self.classes:
            img_names = os.listdir(os.path.join(data_dir, class_name))
            img_names.sort()
            img_paths = [os.path.join(data_dir, class_name, img_name) for img_name in img_names if img_name != '.DS_Store']
            train_size = int(N * 0.8)
            val_size = int(N * 0.1)
            test_size = len(img_paths) - train_size - val_size
            if split == 'train':
                print(f'{class_name}: {len(img_names)}')
                if aug_dir == None:
                    self.images.extend(img_paths[:train_size])
                    self.labels.extend([self.class_to_idx[class_name]]*train_size)
                else:
                    aug_img_paths = [os.path.join(aug_dir, class_name, img_name) for img_name in img_names if img_name != '.DS_Store']
                    for aug_img_path in aug_img_paths[:train_size]:
                        for i in range(aug_size):
                            self.images.append(aug_img_path.replace('.jpg', f'_{i}.jpg'))
                        self.labels.extend([self.class_to_idx[class_name]]*aug_size)
            elif split == 'val':
                self.images.extend(img_paths[train_size:train_size+val_size])
                self.labels.extend([self.class_to_idx[class_name]]*val_size)
            else:
                self.images.extend(img_paths[train_size+val_size:train_size+val_size+test_size])
                self.labels.extend([self.class_to_idx[class_name]]*test_size)
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        # image = pil_img_resize(image)
        if self.transform: image = self.transform(image)
        label = self.labels[idx]
        return image, label