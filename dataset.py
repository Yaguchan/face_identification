import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


sizes = ['base', '32', '64', '128']


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
                        for size in sizes:
                            self.images.append(aug_img_path.replace('.jpg', f'_{size}.jpg'))
                        """
                        for i in range(32):
                            self.images.append(aug_img_path.replace('.jpg', f'_{i}.jpg'))
                        self.labels.extend([self.class_to_idx[class_name]]*32)
                        """
                        self.labels.extend([self.class_to_idx[class_name]]*len(sizes))
                        
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