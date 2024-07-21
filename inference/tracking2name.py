import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from model import CNNModel
from collections import Counter
from utils import seed_everything
from torchvision import transforms


# python inference/tracking2name.py
MODELDIR = './model/resnet18/twitter_all'
DATADIR = './data/yolo-face/64thSingle_short2'
DEVICE = 'cuda:1'
MEMBER_LIST = './member_list/all.txt'
TOPN = 10

# resnet
"""
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),                                  
    transforms.ToTensor(),                                                      
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
"""
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
    
    # track
    with open(os.path.join(MODELDIR, 'inference_tracking.txt'), 'a') as f:
        track_idxs = os.listdir(DATADIR)
        track_idxs = [int(track_idx) for track_idx in track_idxs if track_idx != '.DS_Store' and '.txt' not in track_idx]
        track_idxs.sort()
        for track_idx in tqdm(track_idxs):
            print(f'track_name: {track_name}')
            track_path = os.path.join(DATADIR, str(track_idx))
            # img_names = [os.path.join(class_dir, img_name) for img_name in img_names if img_name != '.DS_Store']
            clips = os.listdir(track_path)
            pred_list = [0 for _ in range(num_classes)]
            max_prob = -1e10
            max_idx = -1
            list_class = []
            list_conf = []
            for clip in clips:
                img_path = os.path.join(track_path, clip)
                image = Image.open(img_path).convert('RGB')
                image = transform(image)
                prob, pred_idx = model.inference(image.unsqueeze(0))
                pred_list[pred_idx] += 1
                list_class.append(pred_idx.item())
                list_conf.append(prob.item())
                if prob > max_prob:
                    max_prob = prob
                    max_idx = pred_idx
            # topN conf
            if len(list_class) > TOPN:
                list_class = list_class[:TOPN]
                list_conf = list_class[:TOPN]
            paired_list = list(zip(list_conf, list_class))
            paired_list.sort()
            list_prob, list_class = zip(*paired_list)
            counter = Counter(list_class)
            most_common_class, count = counter.most_common(1)[0]
            print(f'pred_list: {pred_list}')
            print(f'max_prob_member: {idx_to_class[max_idx.item()]}, max_prob: {max_prob.item()}, ')
            print(f'top{TOPN}_member: {idx_to_class[most_common_class]}, count: {count}')
            #
            track_class = np.argmax(pred_list)
            f.write(f'{track_idx}, {idx_to_class[track_class]}\n')
        

if __name__ == '__main__':
    main()