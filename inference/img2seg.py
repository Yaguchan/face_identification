import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import FaceNet
from ultralytics import YOLO
from facenet_pytorch import MTCNN
from utils import seed_everything
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont


# python inference/img2seg.py
YOLO_MODEL = './weights/yolo/yolov8x-seg.pt'
MODELDIR = './weights/facenet/kamiseven_twitter_mtcnn_aug10'
IMG_PATH = './data/sample/kami3.jpg'
MEMBER_LIST = './member_list/kamiseven.txt'
MEMBER_ENJP = './member_list/member_2012.csv'
FONT_PATH = './data/font/NotoSansJP-Black.ttf'
JP = True
DEVICE = 'cpu'
MINSIZE = 20


# facenet
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def cut_img(image, seg_mask):
    seg_mask = np.array(seg_mask, dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [seg_mask], (255, 255, 255))
    masked_image = cv2.bitwise_and(image, mask)
    x, y, w, h = cv2.boundingRect(seg_mask)
    cropped_image = masked_image[y:y+h, x:x+w]
    # cv2.imwrite(f'{x}_mask.jpg', cropped_image)
    return cropped_image


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
    # yolo
    yolo = YOLO(YOLO_MODEL)
    # facenet
    facenet = FaceNet(num_classes, device)
    facenet.load_state_dict(torch.load(os.path.join(MODELDIR, 'best_val_loss_model.pt')))
    facenet.to(device)
    facenet.eval()
    # mtcnn
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=MINSIZE,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    
    # inference
    seg_list = []
    idx_list = []
    image = cv2.imread(IMG_PATH)
    pil_image = Image.open(IMG_PATH)
    results = yolo(image, classes=0, verbose=False)
    annotated_image = results[0].plot()
    cv2.imwrite('sample.jpg', annotated_image)
    if results[0].masks is not None:
        seg_masks = results[0].masks.xy
        for seg_mask in seg_masks:
            human_image = cut_img(image, seg_mask)
            if human_image.shape[0] < MINSIZE or human_image.shape[1] < MINSIZE: continue
            boxes, _ = mtcnn.detect(human_image)
            if boxes is None: continue
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
            face_image = transform(Image.fromarray(human_image).crop((x1, y1, x2, y2)))
            prob, idx = facenet.inference(face_image.unsqueeze(0))
            seg_list.append(seg_mask.astype(np.int32))
            idx_list.append(idx.item())
    
    # plot
    draw = ImageDraw.Draw(pil_image)
    colors = np.random.randint(0, 255, size=(len(seg_list), 3))
    for seg_mask, idx in zip(seg_list, idx_list):
        member = idx_to_class[idx]
        x1, y1, w, h = cv2.boundingRect(seg_mask)
        B, G, R = map(int, member_to_color[member])
        if JP:
            member = en2jp[member]
            bbox = draw.textbbox((x1, y1), member, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            seg_mask = seg_mask.tolist()
            seg_mask = [(x, y) for x, y in seg_mask]
            seg_mask.append(seg_mask[0])
            draw.line(seg_mask, fill=(R,G,B), width=2)
            draw.rectangle([x1-1, y1-text_height-10, x1+text_width+5, y1], fill=(R,G,B))
            draw.text((x1+5, y1-text_height-8), member, font=font)
        else:
            cv2.polylines(image, [seg_mask], isClosed=True, color=(R,G,B), thickness=2)
            cv2.rectangle(image, (x1-1, y1-20), (x1+len(member)*10, y1), (R,G,B), -1)
            cv2.putText(image, member, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    if JP:
        pil_image.save(IMG_PATH.replace('.jpg', '_seg_jp.jpg'))
    else:
        cv2.imwrite(IMG_PATH.replace('.jpg', '_seg_en.jpg'), image)
    

if __name__ == '__main__':
    main()