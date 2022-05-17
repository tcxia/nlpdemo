# -*- coding: utf-8 -*-
'''
# Created on 2021/01/26 15:08:03
# @filename: train.py
# @author: tcxia
'''



import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from models.model import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Train(model, optimizer, train_loader, dev_loader, epoches, device):

    model.train()
    for epoch in range(epoches):
        train_loader = train_loader.to(device)
        optimizer.zero_grad()
        out = model(train_loader)
        loss = F.nll_loss(out[train_loader.train_mask], train_loader.y[train_loader.train_mask])
        loss.backward()
        optimizer.step()

        acc = Eval(model, dev_loader, device)
        print("Epoch: {} | Loss: {:.4f} | Eval-acc: {:.4f}".format(epoch, loss.item(), acc))

def Eval(model, dev_loader, device):
    model.eval()
    dev_loader = dev_loader.to(device)
    _, pred = model(dev_loader).max(dim=1)
    correct = float(pred[dev_loader.test_mask].eq(dev_loader.y[dev_loader.test_mask]).sum().item())
    acc = correct / dev_loader.test_mask.sum().item()
    return acc

if __name__ == "__main__":
    dataset = Planetoid(root='/data/nlp_dataset/Cora', name='Cora')
    train_loader = dataset[0]
    # print(dir(train_loader))
    model = Net(dataset).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    Train(model, optimizer, train_loader, train_loader, 100, device)