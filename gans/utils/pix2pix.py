# -*- coding: utf-8 -*-
'''
# Created on 2021/01/19 13:20:28
# @filename: pix2pix.py
# @author: tcxia
'''

import torch
import torch.nn as nn

import functools




class PixDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d) -> None:
        super().__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
        ]

        self.net = nn.Sequential(*net)
    
    def forward(self, x):
        return self.net(x)