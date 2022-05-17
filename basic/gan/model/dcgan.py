# -*- coding: utf-8 -*-
'''
# Created on 12月-03-20 16:27
# @filename: dcgan.py
# @author: tcxia
'''

import torch
import torch.nn as nn



class GenNet(nn.Module):
    def __init__(self, nz, ngf, nc) -> None:
        super(GenNet, self).__init__()

        self.main = nn.Sequential(
            # torch.nn.ConvTranspose2d(in_channels, out_channels,
            # kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)

            # Input: (N, C_{in}, H_{in}, W_{in})
            # Output: (N, C_{out}, H_{out}, W_{out})
            # H_out=(H_in−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
            # W_out=(W_in−1)×stride[1]−2×padding[1]+dilation[1]×(kernel_size[1]−1)+output_padding[1]+1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class DisNet(nn.Module):
    def __init__(self, nc, ndf) -> None:
        super(DisNet, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


if __name__ == "__main__":
    nz = 100 # latent vector的大小
    ngf = 64 # generator feature map size
    ndf = 64 # discriminator feature map size
    nc = 3 # color channels

    netG = GenNet(nz, ngf, nc)
    print(netG.main[0].weight)