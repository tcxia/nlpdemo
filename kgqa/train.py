# -*- coding: utf-8 -*-
'''
# Created on 2021/01/25 13:21:05
# @filename: train.py
# @author: tcxia
'''

import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, nargs='?', help='Bacth Size')

args = parser.parse_args()


def Trainer(model, optimizer, train_loader, epoches, device):
    for epoch in range(epoches):
        model.train()

        for i, batch in enumerate(train_loader):
            data_input, data_target = batch
            optimizer.zero_grad()

            e1_idx = torch.tensor(data_input[:, 0])
            r_idx = torch.tensor(data_input[:, 1])

            e1_idx = e1_idx.to(device)
            r_idx = r_idx.to(device)

            pred = model(e1_idx, r_idx)
            loss = model.loss(pred, data_target)

            loss.backward()
            optimizer.step()

            print("Epoch: {} | Iteration: {} | Loss: {}".format(epoch, i, loss.item()))

