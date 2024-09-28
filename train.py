import os
import torch
import torch.nn as nn
from model import FaceNet
from dataset import IdolData
from utils import trainer, tester, seed_everything
from torchvision import transforms
from torch.utils.data import DataLoader


# python train.py
DATANAME = 'twitter'
# LISTNAME = 'kamiseven'
LISTNAME = '61stsingle'
DATADIR = f'./data/{DATANAME}/2024/mtcnn_face'
# DATADIR = f'./data/{DATANAME}_resize/32/images/mtcnn_face'
AUG_DIR = None #f'./data/{DATANAME}_aug10/2024/mtcnn_face'
# AUG_DIR = f'./data/{DATANAME}_scale2/images/yolo_face'
AUG_SIZE = 62 # 2+6*N / 2+10*N
# MODELDIR = f'./weights/facenet/{LISTNAME}_{DATANAME}_aug10'
MODELDIR = f'./weights/facenet/{LISTNAME}_{DATANAME}_mtcnn_aug10'
MEMBER_LIST = f'./member_list/{LISTNAME}.txt'
DEVICE = 'cuda:1'
LR = 1e-5
EPOCH = 10
BATCHSIZE = 32
TRAIN = False


transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def main():
    
    # seed
    seed_everything(42)
    
    # dir
    os.makedirs(MODELDIR, exist_ok=True)
    
    # cuda
    device = torch.device(DEVICE)
    
    # dataset
    # データの呼び出し
    print('---------------------------')
    train_dataset = IdolData(data_dir=DATADIR, member_list=MEMBER_LIST, transform=transform, split='train', aug_dir=AUG_DIR, aug_size=AUG_SIZE)
    val_dataset = IdolData(data_dir=DATADIR, member_list=MEMBER_LIST, transform=transform, split='val')
    test_dataset = IdolData(data_dir=DATADIR, member_list=MEMBER_LIST, transform=transform, split='test')
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=True)
    loader_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    print('---------------------------')
    print(f'train: {len(train_dataset)}')
    print(f'val  : {len(val_dataset)}')
    print(f'test : {len(test_dataset)}')
    print('---------------------------')
    
    # model     
    model = FaceNet(train_dataset.num_classes, device)
    model.to(device)
    
    # optimizer
    parameters = model.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=LR)
    
    # 学習
    if TRAIN:
        trainer(
            num_epochs=EPOCH,
            model=model,
            loader_dict=loader_dict,
            optimizer=optimizer,
            outdir=MODELDIR
        )
	# テスト
    tester(
		model=model,
        loader_dict=loader_dict,
        modeldir=MODELDIR,
        device=device
	)


if __name__ == '__main__':
    main()