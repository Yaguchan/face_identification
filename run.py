import os
import torch
import torch.nn as nn
from model import CNNModel
from dataset import IdolData
from utils import trainer, tester, seed_everything
from torchvision import transforms
from torch.utils.data import DataLoader


# python run.py
NAME = 'twitter'
DATADIR = f'./data/{NAME}/images/face'
AUG_DIR = None #'./data/twitter/images/face'
MODELDIR = f'./model/facenet/{NAME}_all'
MEMBER_LIST = './complete_member.txt'
# MEMBER_LIST = './member.txt'
DEVICE = 'cuda:1'
LR = 1e-4
EPOCH = 15
TRAIN = True


# resnet
"""
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),                                                   # 画像サイズをリサイズ
    transforms.ToTensor(),                                                        # Tensorに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 標準化
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
    
    # dir
    os.makedirs(MODELDIR, exist_ok=True)
    
    # cuda
    device = torch.device(DEVICE)
    
    # dataset
    # データの呼び出し
    train_dataset = IdolData(data_dir=DATADIR, member_list=MEMBER_LIST, transform=transform, split='train', aug_dir=AUG_DIR)
    val_dataset = IdolData(data_dir=DATADIR, member_list=MEMBER_LIST, transform=transform, split='val')
    test_dataset = IdolData(data_dir=DATADIR, member_list=MEMBER_LIST, transform=transform, split='test')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    loader_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    # model     
    model = CNNModel(train_dataset.num_classes, device)
    model.to(device)
    
    # freeze
    """
    for name, param in model.named_parameters():
        if name != 'model.fc.weight' and name != 'model.fc.bias':
        # if name != 'model.fc.weight' and name != 'model.fc.bias' and 'layer4' in name:
            param.requires_grad = False
    """
    
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