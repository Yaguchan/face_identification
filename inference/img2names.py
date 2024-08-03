import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import FaceNet
from ultralytics import YOLO
from utils import seed_everything
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont


# python inference/img2names.py
YOLO_MODEL = './weights/yolo/yolov8l-face.pt'
FACENET_MODEL = './weights/facenet/aug2_all.pt'
# FACENET_MODEL = './weights/facenet/resize_64thsingle.pt'
IMG_PATH = './data/sample/sample3.jpg'
MEMBER_LIST = './member_list/all.txt'
# MEMBER_LIST = './member_list/64thsingle.txt'
MEMBER_ENJP = './member_list/member.csv'
FONT_PATH = './data/font/NotoSansJP-Black.ttf'
JP = False
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
    colors = np.random.randint(0, 255, size=(num_classes, 3))
    member_to_color = {member: color for member, color in zip(classes, colors)}
    
    # class_jp
    if JP:
        df = pd.read_csv(MEMBER_ENJP)
        en2jp = df.set_index('en')['jp'].to_dict()
        font = ImageFont.truetype(FONT_PATH, 20)
    
    # model
    device = torch.device(DEVICE)
    yolo = YOLO(YOLO_MODEL)
    facenet = FaceNet(num_classes, device)
    facenet.load_state_dict(torch.load(FACENET_MODEL, map_location=device))
    facenet.to(device)
    facenet.eval()
    
    # inference
    image = cv2.imread(IMG_PATH)
    pil_image = Image.open(IMG_PATH)
    boxes = yolo.predict(image)[0].boxes
    xyxy_list = boxes.xyxy.tolist()
    conf_list = boxes.conf.tolist()
    idx_list = []
    for xyxy in xyxy_list:
        face_image = transform(pil_image.crop(xyxy))
        prob, idx = facenet.inference(face_image.unsqueeze(0))
        idx_list.append(idx.item())
    
    # plot
    draw = ImageDraw.Draw(pil_image)
    colors = np.random.randint(0, 255, size=(len(xyxy_list), 3))
    for xyxy, idx in zip(xyxy_list, idx_list):
        x1, y1, x2, y2 = xyxy
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        member = idx_to_class[idx]
        B, G, R = map(int, member_to_color[member])
        if JP:
            member = en2jp[member]
            draw.rectangle([x1, y1, x2, y2], outline=(R,G,B), width=2)
            bbox = draw.textbbox((x1, y1), member, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.rectangle([x1 - 1, y1 - text_height - 10, x1 + text_width + 5, y1], fill=(R,G,B))
            draw.text((x1 + 5, y1 - text_height - 8), member, font=font)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(image, (x1 - 1, y1 - 20), (x1 + len(member) * 10, y1), (B, G, R), -1)
            cv2.putText(image, member, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if JP:
        pil_image.save(IMG_PATH.replace('.jpg', '_jp.jpg'))
    else:
        cv2.imwrite(IMG_PATH.replace('.jpg', '_en.jpg'), image)
    

if __name__ == '__main__':
    main()