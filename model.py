import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from facenet_pytorch import InceptionResnetV1


class CNNModel(nn.Module):
    
    def __init__(self, output_dim, device):
        super().__init__()
        self.device = device
        # resnet
        # model = models.resnet18(weights="DEFAULT")
        # model.fc = nn.Linear(model.fc.in_features, out_features=output_dim)
        # mobilenet
        # model = models.mobilenet_v2(weights='DEFAULT')
        # model.classifier[1] = nn.Linear(model.classifier[1].in_features, output_dim)
        # efficientnet
        # model = models.efficientnet_b0(weights='DEFAULT')
        # model.classifier[1] = nn.Linear(model.classifier[1].in_features, output_dim)
        # vgg16
        # model = models.vgg16(pretrained=True)
        # model.classifier[6] = nn.Linear(model.classifier[6].in_features, out_features=output_dim) 
        # facenet
        model = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=output_dim)
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, batch):
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = self.criterion(outputs, labels)
        acc = torch.sum(preds == labels)
        results = {
            'pred': preds,
            'loss': loss,
            'acc': acc
        }
        return results
    
    def inference(self, inputs):
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        prob, pred = torch.max(outputs, 1)
        return prob, pred
        
    