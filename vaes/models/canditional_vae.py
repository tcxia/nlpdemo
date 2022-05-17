# -*- coding: utf-8 -*-
'''
# Created on 2021/01/18 15:08:41
# @filename: canditional_vae.py
# @author: tcxia
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    def __init__(self, in_channels, num_classes, latent_dim, hidden_dim=None, img_size=64) -> None:
        super().__init__()

        self.inc = in_channels
        self.img_size = img_size

        self.embed_size = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        if hidden_dim is None:
            hidden_dim = [32, 64, 128, 256, 512]

        in_channels += 1 # 增加一维通道 标签占位符

        encoder_modules = []
        for h_dim  in hidden_dim:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*encoder_modules)

        self.fc_mu = nn.Linear(hidden_dim[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dim[-1] * 4, latent_dim)


        decoder_modules = []

        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dim[-1] * 4)

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

        self.fc = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[-1], hidden_dim[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh()
        )



    def forward(self, x, classes):
        y = classes['labels'].float()
        embed_cls = self.embed_size(y)
        embed_cls = embed_cls.view(-1, self.img_size, self.img_size).unsqueeze(1)

        embed_x = self.embed_data(x)
    
        x = torch.cat([embed_x, embed_cls], dim=1)
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        out = self.decode(torch.cat([z, y], dim=1))
        return [out, x, mu, logvar]


    def encode(self, x):

        bs = x.shape[0]

        out = self.encoder(x)
        out = out.view(bs, -1)

        mu = self.fc_mu(out)
        logvar = self.fc_var(out)
        return [mu, logvar]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = eps * std + mu
        return sample

    def decode(self, z):
        out = self.decoder_input(z)
        out = out.view(-1, 512, 2, 2)
        out = self.decoder(out)
        out = self.fc(out)
        return out


    def loss_fc(self, out, m_n):
        recons = out[0]
        x = out[1]
        mu = out[2]
        logvar = out[3]

        kld_weight = m_n
        
        recons_loss = F.mse_loss(recons, x)
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - logvar.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        
        return {'loss': loss, 'recons-loss': recons_loss, 'kld-loss': -kld_loss}

