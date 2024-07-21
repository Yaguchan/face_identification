import os
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from model import CNNModel
from utils import seed_everything
from torchvision import transforms


# python inference_sample.py
MODELDIR = './model/mobilenetv2/twitter_aug_all'
IMG_PATH = './data/icrawler/images/face/MizukiYamauchi/6.jpg'
DEVICE = 'cuda:1'
MEMBER_LIST = './complete_member.txt'
# MEMBER_LIST = './member.txt'


# resnet
"""
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),                                  
    transforms.ToTensor(),                                                      
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
"""
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
    model = CNNModel(num_classes, device)
    model.load_state_dict(torch.load(os.path.join(MODELDIR, 'best_val_loss_model.pth')))
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