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
                 hidden_dim=None,
                 anneal_steps=200,
                 alpha=1.,
                 beta=6.,
                 gamma=1.,
                 max_capacity=25,
                 capacity_max_iter=1e5,
                 loss_type='B',
                 device='cpu') -> None:
        super().__init__()

        self.latent_dim = latent_dim

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = capacity_max_iter

        self.anneal_steps = anneal_steps

        self.device = device

        # 构建 Encoder
        encoder_modules = []
        if hidden_dim is None:
            hidden_dim = [32, 32, 32, 32, 32]

        for h_dim in hidden_dim:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              h_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    nn.LeakyReLU()))
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_modules)

        self.fc = nn.Linear(hidden_dim[-1] * 16 * 24, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)


        # 构建解码器
        decoder_modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dim[-1] * 16 * 24)

        hidden_dim.reverse()
        for i in range(len(hidden_dim) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dim[i], hidden_dim[i+1], kernel_size=3, stride=1, padding=1, output_padding=0),
                    # nn.BatchNorm2d(hidden_dim[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*decoder_modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[-1], hidden_dim[-1], kernel_size=3, stride=1, padding=1, output_padding=0),
            # nn.BatchNorm2d(hidden_dim[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim[-1], out_channels=2, kernel_size=3, padding=1),
            nn.Tanh()
        )


    def encode(self, inputs):
        # inputs: [batch_size, c, h, w]
        inputs = inputs.permute(0, 3, 1, 2)
        result = self.encoder(inputs)
        # print(result.shape)
        # 按照batch_size拉直
        # [batch_size, c * h * w]
        result = torch.flatten(result, start_dim=1)
        # print(result.shape) # [100, 32]
        result = self.fc(result)
        # split the result into mu and var components of the latent Guassian Distribution

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        # z: [B x D]
        result = self.decoder_input(z)
        # print(result.shape)
        result = result.view(-1, 32, 24, 16)
        result = self.decoder(result)
        # print(result.shape)
        result = self.final_layer(result)
        result = result.permute(0, 2, 3, 1)
        # [B, C, H, W]
        return result

    def reparameterize(self, mu, logvar):
        """
            Reparameterization trick to sample from N(mu, var) from N(0, 1)
            mu: Mean of the latent Gaussian
            logvar: Standard deviation of the latent Guassian
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var, z]


    # KL散度
    def loss_func(self, output, **kwargs):
        # self.num_iter += 1
        recons = output[0] # 重建
        inputs = output[1]
        mu = output[2]
        log_var = output[3]

        kld_weight = kwargs['M_N']


        recons_loss = F.mse_loss(recons, inputs)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':
            self.C_max = self.C_max.to(inputs.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type')

        return {'loss': loss, 'reconstruction loss': recons_loss, 'KLD': kld_loss}


    def log_density_guassian(self, x, mu, logvar):
        norm = -0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    def loss_func_tc(self, *args, **kwargs):
        recons = args[0]
        inputs = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]

        weight = 1
        recons_loss = F.mse_loss(recons, inputs, reduction='sum')

        log_q_zx = self.log_density_guassian(z, mu, log_var).sum(dim=1)

        zeros = torch.zeros_like(z)
        log_p_z = self.log_density_guassian(z, zeros, zeros).sum(dim=1)

    
        bs, latent_dim = z.shape
        mat_log_q_z = self.log_density_guassian(z.view(bs, 1, latent_dim), mu.view(1, bs, latent_dim), log_var.view(1, bs, latent_dim))

        # dataset_size = (1 / m_n) * bs
        dataset_size = int((1 / kwargs['M_N']) * bs)
        start_weight = (dataset_size - bs + 1) / (dataset_size * (bs - 1))
        importance_weight = torch.Tensor(bs, bs).fill_(1 / (bs - 1)).to(inputs.device)
        importance_weight.view(-1)[::bs] = 1 / dataset_size
        importance_weight.view(-1)[1::bs] = start_weight
        importance_weight[bs - 2, 0] = start_weight
        log_importance_weight = importance_weight.log()

        mat_log_q_z += log_importance_weight.view(bs, bs, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prob_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

        mi_loss = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prob_q_z).mean()
        kld_loss = (log_prob_q_z - log_p_z).mean()

        if self.training:
            self.num_iter += 1
            anneal_rate = min(1 * self.num_iter / self.anneal_steps, 1)
        else:
            anneal_rate = 1.

        loss =  recons_loss / bs + self.alpha * mi_loss + weight * (self.beta * tc_loss + anneal_rate * self.gamma * kld_loss)

        return {'loss': loss, 'Reconstruction_loss': recons_loss, 'KLD': kld_loss, 'TC_loss': tc_loss, 'MI_loss': mi_loss}


    def sample(self, num_sample):
        z = torch.rand(num_sample, self.latent_dim)
        z.to(self.device)

        samples = self.decode(z)
        return samples


    def generate(self, x):
        return self.forward(x)[0]

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


if __name__ == "__main__":

    data = torch.randn(100, 2, 24, 16)
    model = BetaVAE(in_channels=2, latent_dim=10, anneal_steps=10000)
    # print(model)

    out = model(data)
    print(out[0].shape)
    print(out[1].shape)