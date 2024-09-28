import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN


# python preprocess/cut_face.py
INDIR = 'data/twitter/2024/original'
OUTDIR = 'data/twitter/2024/mtcnn_face'
DEVICE = 'cpu'


def revised_box(boxes, img):
    max_h, max_w, _ = img.shape
    if boxes[0][0] < 0: boxes[0][0] = 0
    if boxes[0][1] < 0: boxes[0][1] = 0
    if boxes[0][2] >= max_w: boxes[0][2] = max_w-1
    if boxes[0][3] >= max_h: boxes[0][3] = max_h-1
    return boxes


def main():
    
    member_names = os.listdir(INDIR)
    member_names = [member_name for member_name in member_names if member_name != '.DS_Store']
    # member_names = ['ErinaOda']
    
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=torch.device(DEVICE)
    )
    
    for member_name in tqdm(member_names):
        in_member_dir = os.path.join(INDIR, member_name)
        out_member_dir = os.path.join(OUTDIR, member_name)
        os.makedirs(out_member_dir, exist_ok=True)
        img_names = os.listdir(in_member_dir)
        img_names = [img_name for img_name in img_names if img_name != '.DS_Store']
        for img_name in img_names:
            in_img_path = os.path.join(in_member_dir, img_name)
            out_img_path = os.path.join(out_member_dir, img_name)
            img = Image.open(in_img_path)
            boxes, probs = mtcnn.detect(img)
            if boxes is not None and len(boxes) == 1:
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                boxes = revised_box(boxes, img)
                face_img = img[int(boxes[0][1]):int(boxes[0][3]), int(boxes[0][0]):int(boxes[0][2])]
                face_img = cv2.resize(face_img, (160, 160))
                cv2.imwrite(out_img_path.replace('.jpeg', '.jpg'), face_img)
    

if __name__ == '__main__':
    main()