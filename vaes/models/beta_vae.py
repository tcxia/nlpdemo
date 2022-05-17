# -*- coding: utf-8 -*-
'''
# Created on 2021/01/14 13:19:12
# @filename: beta_vae.py
# @author: tcxia
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from torch.autograd import Variable

# 在传统的VAE上loss中关于KL散度，添加一个超参数，使得KL更小

class BetaVAE(nn.Module):

    num_iter = 0 # keep track of iterations

    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dim,
                 beta=4,
                 gamma=1000.,
                 max_capacity=25,
                 capacity_max_iter=1e5,
                 loss_type='B') -> None:
        super().__init__()

        self.latent_dim = latent_dim

        self.beta = beta
        self.gamma = gamma


        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = capacity_max_iter


        # 构建 Encoder
        encoder_modules = []
        if hidden_dim is None:
            hidden_dim = [32, 64, 128, 256, 512]

        for h_dim in hidden_dim:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              h_dim,
                              kernel_size=3,
                              stride=2,
                              padding=1),
                    nn.LeakyReLU()))
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_modules)

        self.fc_mu = nn.Linear(hidden_dim[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dim[-1] * 4, latent_dim)


        # 构建解码器
        decoder_modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dim[-1] * 4)

        hidden_dim.reverse()
        for i in range(len(hidden_dim) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dim[i], hidden_dim[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dim[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*decoder_modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[-1], hidden_dim[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh()
        )


    def encode(self, inputs):
        # inputs: [batch_size, c, h, w]
        result = self.encoder(inputs)
        # 按照batch_size拉直
        # [batch_size, c * h * w]
        result = torch.flatten(result, start_dim=1) # result = result.view(bs, -1)


        # split the result into mu and var components of the latent Guassian Distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        # z: [B x D]
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        # [B, C, H, W]
        return result

    def reparameterize(self, mu, logvar):
        """
            Reparameterization trick to sample from N(mu, var) from N(0, 1)
            mu: Mean of the latent Gaussian
            logvar: Standard deviation of the latent Guassian
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # 构建sample
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var, z]


    # KL散度
    def loss_func(self, output, m_n):
        self.num_iter += 1

        recons = output[0] # 重建
        inputs = output[1]
        mu = output[2]
        log_var = output[3]

        kld_weight = m_n

        recons_loss = F.mse_loss(recons, inputs) # 重构损失

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':
            self.C_max = self.C_max.to(inputs.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type')

        return {'loss': loss, 'Recons_Loss': recons_loss, 'KLD': kld_loss}


    def sample(self, num_sample):
        z = torch.rand(num_sample, self.latent_dim)
        z.to(self.device)

        samples = self.decode(z)
        return samples


    def generate(self, x):
        return self.forward(x)[0]

