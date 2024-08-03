import os
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from model import FaceNet
from utils import seed_everything
from torchvision import transforms


# python inference/faceimg2name.py
FACENET_MODEL = './weights/facenet/resize_all.pt'
IMG_PATH = './data/icrawler/images/face/MizukiYamauchi/6.jpg'
MEMBER_LIST = './member_list/all.txt'
DEVICE = 'cpu'


# facenet
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def main():
    
    # seed
    seed_everything(42)
    
    # class
    with open(MEMBER_LIST, 'r', encoding='utf-8') as f:
        classes = f.read().splitlines()
    num_classes = len(classes)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # model
    device = torch.device(DEVICE)
    model = FaceNet(num_classes, device)
    model.load_state_dict(torch.load(FACENET_MODEL, map_location=device))
    model.to(device)
    model.eval()
    
    # inference
    image = Image.open(IMG_PATH).convert('RGB')
    image = transform(image)
    prob, pred_idx = model.inference(image.unsqueeze(0))
    print(f'idx: {pred_idx.item()}, prob: {prob.item()}')
    print(f'output: {idx_to_class[pred_idx.item()]}')
        

if __name__ == '__main__':
    main()