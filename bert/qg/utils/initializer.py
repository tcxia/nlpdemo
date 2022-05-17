# -*- coding: utf-8 -*-
'''
# Created on 11 月-18-20 15:18
# initializer.py
# @author: tcxia
'''
import torch
import torch.nn as nn


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)