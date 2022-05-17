# -*- coding: utf-8 -*-
'''
# Created on 12月-02-20 15:04
# @filename: train.py
# @author: tcxia
'''

import torch
import torch.utils.data as tud
from torchvision import datasets, transforms
import torch.nn.functional as F

from loguru import logger

from model.cnn import CNNNet


torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)

epochs = 10
batch_size = 4
learning_rate = 0.01
momentum = 0.5


def Trainer(epoch, model, train_dataloader, optimizer, device):
    model.train()

    for i, batch in enumerate(train_dataloader):
        data, target = batch
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = F.nll_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            logger.info("Epoch: [{}/{}] | Step: [{}/{}] | Loss: {}".format(
                epoch, epochs, i * len(data), len(train_dataloader.dataset), loss.item()))


def Tester(epoch, model, test_dataloader, device):
    model.eval()

    test_loss = 0.
    correct = 0.
    for i, batch in enumerate(test_dataloader):
        data, target = batch
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    logger.info("Epoch: [{}/{}] | Avg Loss: {:.4f} | Acc: {:.4f}".format(epoch, epochs, test_loss, 100*correct / len(test_dataloader.dataset)))






if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # mnist_data
    # fashion_mnist_data
    train_loader = tud.DataLoader(datasets.MNIST('./mnist_data',
                                                 train=True,
                                                 download=True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(
                                                         (0.1307, ),
                                                         (0.3081, ))
                                                 ])),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=0)

    test_loader = tud.DataLoader(datasets.MNIST('./mnist_data',
                                                train=False,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        (0.1307, ), (0.3081, ))
                                                ])),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=0)


    model = CNNNet()
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(epochs):
        Trainer(epoch, model, train_loader, optimizer, device)
        Tester(epoch, model, test_loader, device)
