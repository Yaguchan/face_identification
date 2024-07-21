import os
import torch
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


# seed
def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_cm(preds, labels, class_names=None, savedir='./'):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    if class_names == None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(savedir, 'cm.png'))
    plt.close()


def train(model, optimizer, data_loader):
    model.train()
    total_loss = 0
    total_acc = 0
    for batch in tqdm(data_loader):
        optimizer.zero_grad() 
        output = model(batch)
        loss = output['loss']
        acc = output['acc']
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)
        total_acc += acc
    total_loss = total_loss / len(data_loader.dataset)
    total_acc = total_acc / len(data_loader.dataset)
    return total_loss, total_acc


def test(model, data_loader, split, outdir=None):
    model.eval()
    total_loss = 0
    total_acc = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs, labels = batch
            output = model(batch)
            loss = output['loss']
            acc = output['acc']
            total_loss += loss.item() * len(batch)
            total_acc += acc
            all_preds.extend(output['pred'].detach().cpu().numpy())
            all_targets.extend(labels)
    if split == 'test': 
        save_cm(all_preds, all_targets, savedir=outdir)
    total_loss = total_loss / len(data_loader.dataset)
    total_acc = total_acc / len(data_loader.dataset)
    return total_loss, total_acc


def trainer(num_epochs, model, loader_dict, optimizer, outdir):
    best_val_loss = 1e10
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch+1}')
        train_loss, train_acc = train(model, optimizer, loader_dict['train'])
        val_loss, val_acc = test(model, loader_dict['val'], 'val')
        print(f'Train loss : {train_loss}, Train Acc :{train_acc}')
        print(f'Val loss   : {val_loss}, Val Acc   :{val_acc}')
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(outdir, 'best_val_loss_model.pt'))


def tester(model, loader_dict, modeldir, device):
    model.load_state_dict(torch.load(os.path.join(modeldir, 'best_val_loss_model.pt')))
    model.to(device)
    test_loss, test_acc = test(model, loader_dict['test'], 'test', modeldir)
    print(f'Test loss  : {test_loss}, Test Acc  :{test_acc}')