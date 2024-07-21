import cv2
import torch
import numpy as np
from PIL import Image
from model import CNNModel
from ultralytics import YOLO
from utils import seed_everything
from torchvision import transforms


# python inference/img2names.py
YOLO_MODEL = './weights/yolo/yolov8l-face.pt'
FACENET_MODEL = './weights/facenet/all.pth'
IMG_PATH = './data/sample/sample2.jpg'
DEVICE = 'cpu'
MEMBER_LIST = './member_list/all.txt'


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
    
    # model
    device = torch.device(DEVICE)
    yolo = YOLO(YOLO_MODEL)
    facenet = CNNModel(num_classes, device)
    facenet.load_state_dict(torch.load(FACENET_MODEL, map_location=device))
    facenet.to(device)
    facenet.eval()
    
    # inference
    image = cv2.imread(IMG_PATH)
    boxes = yolo.predict(image)[0].boxes
    xyxy_list = boxes.xyxy.tolist()
    conf_list = boxes.conf.tolist()
    idx_list = []
    for xyxy in xyxy_list:
        pil_image = Image.fromarray(image).crop(xyxy)
        face_image = transform(pil_image)
        prob, idx = facenet.inference(face_image.unsqueeze(0))
        idx_list.append(idx.item())
    
    # plot
    colors = np.random.randint(0, 255, size=(len(xyxy_list), 3))
    for xyxy, idx in zip(xyxy_list, idx_list):
        x1, y1, x2, y2 = xyxy
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        member = idx_to_class[idx]
        B, G, R = map(int, member_to_color[member])
        cv2.rectangle(image, (x1, y1), (x2, y2), (B, G, R), 2)
        cv2.rectangle(image, (x1 - 1, y1 - 20), (x1 + len(member) * 12, y1), (B, G, R), -1)
        cv2.putText(image, member, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imwrite(IMG_PATH.replace('.jpg', '_plot.jpg'), image)
        

if __name__ == '__main__':
    main()