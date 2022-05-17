# -*- coding: utf-8 -*-
'''
# Created on 2021/01/15 11:10:04
# @filename: train.py
# @author: tcxia
'''

import numpy as np
import scipy.io as sio
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as tud

from data.dataset import DatasetMIMI
from models.baseline import AutoEncoder
from models.beta_vae import BetaVAE
from models.vanilla_vae import VanillaVAE
from models.base_vae import VAE
from models.conditional_vae import ConditionalVAE

if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
else:
    torch.manual_seed(1)

NUM_FEEDBACK_BITS = 512
CHANNEL_SHAPE_DIM1 = 24
CHANNEL_SHAPE_DIM2 = 16
CHANNEL_SHAPE_DIM3 = 2

EPOCHES = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def Trainer(model, train_loader, dev_loader, optimizer, criterion, epoches, device):

    best_loss = 10.
    for epoch in range(epoches):
        model.train()

        for i, batch in enumerate(train_loader):
            batch = batch.to(device)

            # out = model(batch)


            # m_n = BATCH_SIZE / len(train_loader)

            x_hat, kld = model(batch)
            loss = criterion(x_hat, batch)

            if kld is not None:
                elbo = -loss - kld
                loss = -elbo

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("Epoch: {} | Iteration: {} | Train_Loss:{:.3f} | KLD_loss: {:.3f}".format(epoch, i, loss.item(), kld.item()))

        if epoch % 2 == 0:
            val_loss = Evaler(model, dev_loader, criterion, device)
            print("Epoch: {} |  Val_Loss: {}".format(epoch, val_loss))
            if val_loss < best_loss:
                best_loss = val_loss
                print("====== Save Model =====")
                torch.save({
                    'state_dict': model.encoder.state_dict(),
                }, './checkpoint/encoder.pth.tar')

                torch.save({
                    'state_dict': model.decoder.state_dict(),
                }, './checkpoint/decoder.pth.tar')


def Evaler(model, dev_loader, criterion, device):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dev_loader, desc='[Evaluating]')):
            batch = batch.to(device)
            # out = model(batch)
            # loss = model.loss_func(out[0], batch)
            x_hat, kld = model(batch)
            loss = model.NMSE(batch, x_hat)

            if kld is not None:
                elbo = -loss - kld
                loss = -elbo

            total_loss += loss.item()
        avg_loss = total_loss / len(dev_loader)
    return avg_loss


if __name__ == "__main__":
    mat_path = '/data/nlp_dataset/H_4T4R.mat'
    mat_data = sio.loadmat(mat_path)

    data = mat_data['H_4T4R']
    data = data.astype('float32')
    data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2,
                             CHANNEL_SHAPE_DIM3))

    split_data = int(data.shape[0] * 0.7)
    data_train, data_dev = data[:split_data], data[split_data:]

    train_set = DatasetMIMI(data_train)
    train_loader = tud.DataLoader(train_set,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True)

    dev_set = DatasetMIMI(data_dev)
    dev_loader = tud.DataLoader(dev_set,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True)


    # model = AutoEncoder(NUM_FEEDBACK_BITS)
    model = ConditionalVAE(in_channels=3, num_classes=10, latent_dim=10)
    model.to(device)

    criterion = nn.MSELoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    Trainer(model, train_loader, dev_loader, optimizer, criterion, EPOCHES, device)
