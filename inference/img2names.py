import os
import cv2
import sys
sys.path.append(os.pardir)
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import FaceNet
from facenet_pytorch import MTCNN
from utils import seed_everything
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont


# python inference/img2names.py
MODELPATH = './weights/facenet/63rdsingle.pt'
IMG_PATH = './data/sample/pair1.jpg'
MEMBER_LIST = './member_list/63rdsingle.txt'
MEMBER_ENJP = './member_list/member.csv'
FONT_PATH = './data/font/NotoSansJP-Black.ttf'
FONT_SIZE = 'large' # normal/large
DEVICE = 'cpu'
JP = True


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
    colors = np.random.randint(0, 255, size=(num_classes, 3))
    member_to_color = {member: color for member, color in zip(classes, colors)}
    
    # font size
    if FONT_SIZE == 'large':
        font_list = [60, -3, -30, 15, 5, -32]
    else:
        font_list = [20, -1, -10, 5, 5, -8]
    
    # class_jp
    if JP:
        df = pd.read_csv(MEMBER_ENJP)
        en2jp = df.set_index('en')['jp'].to_dict()
        font = ImageFont.truetype(FONT_PATH, font_list[0])
    
    # model
    device = torch.device(DEVICE)
    # facenet
    model = FaceNet(num_classes, device)
    model.load_state_dict(torch.load(MODELPATH, map_location=torch.device(DEVICE)))
    model.to(device)
    model.eval()
    # mtcnn
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    
    # inference
    xyxy_list = []
    idx_list = []
    image = cv2.imread(IMG_PATH)
    pil_image = Image.open(IMG_PATH)
    boxes, _ = mtcnn.detect(pil_image)
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(b) for b in box]
            face_image = transform(pil_image.crop((x1, y1, x2, y2)))
            prob, idx = model.inference(face_image.unsqueeze(0))
            idx_list.append(idx.item())
            xyxy_list.append((x1, y1, x2, y2))
    # print(idx_list, xyxy_list)
    
    # plot
    draw = ImageDraw.Draw(pil_image)
    colors = np.random.randint(0, 255, size=(len(xyxy_list), 3))
    for xyxy, idx in zip(xyxy_list, idx_list):
        x1, y1, x2, y2 = xyxy
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        member = idx_to_class[idx]
        B, G, R = map(int, member_to_color[member])
        # B, G, R = 0, 0, 255
        if JP:
            member = en2jp[member]
            draw.rectangle([x1, y1, x2, y2], outline=(R,G,B), width=3)
            bbox = draw.textbbox((x1, y1), member, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.rectangle([x1 + font_list[1], y1 - text_height + font_list[2], x1 + text_width + font_list[3], y1], fill=(R,G,B))
            draw.text((x1 + font_list[4], y1 - text_height + font_list[5]), member, font=font)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (B, G, R), 3)
            cv2.rectangle(image, (x1 - 1, y1 - 20), (x1 + len(member) * 10, y1), (B, G, R), -1)
            cv2.putText(image, member, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if JP:
        pil_image.save(IMG_PATH.replace('.jpg', '_jp.jpg'))
    else:
        cv2.imwrite(IMG_PATH.replace('.jpg', '_en.jpg'), image)
        

if __name__ == '__main__':
    main()