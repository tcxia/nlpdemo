# -*- coding: utf-8 -*-
'''
# Created on 12月-02-20 19:50
# @filename: vgg.py
# @author: tcxia
'''

import os
import torch
import torch.nn as nn

from torchvision import models

# os.environ['TORCH_HOME'] = '../vgg'


class VGGNet(nn.Module):
    def __init__(self) -> None:
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)

        return features
        
        


# vgg_model = models.vgg19(pretrained=True)
# print(vgg_model.features._modules.items())
