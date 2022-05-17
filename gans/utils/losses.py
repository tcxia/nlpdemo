# -*- coding: utf-8 -*-
'''
# Created on 2021/01/19 13:45:18
# @filename: losses.py
# @author: tcxia
'''
import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1., target_fake_label=0.) -> None:
        super().__init__()

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError('gan mode [%s] not implemented' % gan_mode)

        # 注册buffer
        # 在内存中定义一个常量， 同时模型保存和加载的时候可以写入和读出
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

    
    def get_target_tensor(self, pred, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(pred)

    def __call__(self, pred, target_is_real):

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(pred, target_is_real)
            loss = self.loss(pred, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -pred.mean()
            else:
                loss = pred.mean()
        else:
            loss = None
        return loss

