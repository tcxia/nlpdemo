# -*- coding: utf-8 -*-
'''
# Created on 2021/01/18 13:21:02
# @filename: base_vae.py
# @author: tcxia
'''

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 768),
            nn.Sigmoid()
        )

    def forward(self, x):

        bs = x.size(0)
        x = x.view(bs, -1)

        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)

        z = mu + logvar * torch.randn_like(logvar)

        kld = 0.5 * torch.mean(torch.pow(mu, 2) + torch.pow(logvar, 2) - torch.log(1e-8 + torch.pow(logvar, 2)) - 1)

        x_hat = self.decoder(h)
        x_hat = x_hat.view(bs, 24, 16, 2)

        return x_hat, kld

    def NMSE(self, x, x_hat):
        x = x.detach().cpu().numpy()
        x_hat = x_hat.detach().cpu().numpy()

        x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
        x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
        x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
        x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
        x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
        x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
        power = np.sum(abs(x_C)**2, axis=1)
        mse = np.sum(abs(x_C - x_hat_C)**2, axis=1)
        nmse = np.mean(mse / power)

        return Variable(torch.tensor(nmse), requires_grad=True)
