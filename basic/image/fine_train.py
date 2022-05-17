# -*- coding: utf-8 -*-
'''
# Created on 12月-02-20 16:17
# @filename: fine_train.py
# @author: tcxia
'''

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.utils.data as tud

from loguru import logger

os.environ['TORCH_HOME'] = './resnet'
root_dir = '/data/nlp_dataset/hymenoptera_data'

bacth_size = 8
num_classes = 2
feature_extract = True
learning_rate = 0.01
momentum = 0.9

epochs = 100



device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def set_param_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

def init_model(num_classes, feature_extract, use_pretrained=True):
    input_size = 224
    if use_pretrained:
        model_ft = models.resnet18(pretrained=False)
        # print(model_ft.layer4[0].conv1.weight) # 答应某一层权重
        set_param_requires_grad(model_ft, feature_extract)
        # print(model_ft.layer4[0].conv1.weight.requires_grad)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    else:
        model_ft = models.resnet18(pretrained=False)

    return model_ft, input_size


def Trainer(epoch, model, train_dataloder, optimizer, crit, device):
    model.train()

    for i, (inputs, labels) in enumerate(train_dataloder):
        inputs = inputs.to(device)

        labels = labels.to(device)

        outputs = model(inputs)
        loss = crit(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            logger.info(
                "Epoch: [{}/{}] | Train-Loss: {}".format(
                    epoch, epochs, loss.item()))

        # _, preds = torch.max(outputs, 1)



if __name__ == "__main__":

    
    model_ft, input_size = init_model(num_classes, feature_extract)
    train_set = datasets.ImageFolder(
        root=os.path.join(root_dir, 'train'),
        transform=transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
    train_loader = tud.DataLoader(train_set, batch_size=bacth_size, num_workers=0, shuffle=True)

    # img = next(iter(train_loader))
    # print(img) # [input, labels]
    # print(img[0].shape) # [8, 3, 224, 224]

    
    model_ft.to(device)

    params_to_update = model_ft.parameters()
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print('\t', name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    optimizer = torch.optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)

    crit = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        Trainer(epoch, model_ft, train_loader, optimizer, crit, device)
