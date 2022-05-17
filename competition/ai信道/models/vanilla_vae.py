# -*- coding: utf-8 -*-
'''
# Created on 2021/01/15 18:50:36
# @filename: vanilla_vae.py
# @author: tcxia
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaVAE(nn.Module):
    def __init__(self, inc, latent_dim, hidden_dim=None) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        if hidden_dim is None:
            hidden_dim = [32, 64, 128, 256, 512]

        encoder_modules = []
        for h_dim in hidden_dim:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(inc, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            inc = h_dim

        self.encoder = nn.Sequential(*encoder_modules)

        self.fc_mu = nn.Linear(hidden_dim[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dim[-1], latent_dim)

        decoder_modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dim[-1])

        hidden_dim.reverse()

        for i in range(len(hidden_dim) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dim[i],
                                       hidden_dim[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dim[i + 1]), nn.LeakyReLU()))

        self.decoder = nn.Sequential(*decoder_modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[-1],
                               hidden_dim[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dim[-1]), nn.LeakyReLU(),
            nn.Conv2d(hidden_dim[-1], out_channels=2, kernel_size=3,
                      padding=1), nn.Tanh())

    def encode(self, inputs):
        out = self.encoder(inputs)
        out = torch.flatten(out, start_dim=1)

        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        return [mu, log_var]

    def decode(self, z):
        out = self.decoder_input(z)
        out = out.view(-1, 512, 1, 1)
        out = self.decoder(out)
        out = self.final_layer(out)
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        # x: [batch, 2, 24, 16]
        x = x.permute(0, 3, 1, 2)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        
        return [out, x, mu, log_var]

    def loss_func(self, output, m_n):
        recons = output[0]
        x = output[1]
        mu = output[2]
        log_var = output[3]

        kld_weight = m_n

        recons_loss = F.mse_loss(recons, x)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
            dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {
            'loss': loss,
            'recons_loss': recons_loss,
            'kld_loss': -kld_loss
        }
