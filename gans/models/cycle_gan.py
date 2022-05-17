# -*- coding: utf-8 -*-
'''
# Created on 2021/01/18 16:28:24
# @filename: cycle_gan.py
# @author: tcxia
'''

import itertools

import torch
import torch.nn as nn

from models.networks import define_G, define_D
from utils.losses import GANLoss
from utils.img_pool import ImagePool


class CycleGAN(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ndf,
                 lr=0.01,
                 beta=0.5,
                 pool_size=128,
                 direction='AtoB',
                 gan_mode='lsgan',
                 netG='resnet_9',
                 netD='basic',
                 n_layers_D=3,
                 norm='batch',
                 use_dropout=False,
                 init_type='normal',
                 init_gain=0.02,
                 gpus_ids=[],
                 device='cpu') -> None:
        super().__init__()
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        self.netG_A = define_G(input_nc, output_nc, ndf, netG, norm, use_dropout, init_type, init_gain, gpus_ids)
        self.netG_B = define_G(input_nc, output_nc, ndf, netG, norm, use_dropout, init_type, init_gain, gpus_ids)

        if self.training:
            self.netD_A = define_D(input_nc, ndf, netD, n_layers_D, norm, init_type, init_gain, gpus_ids)
            self.netD_B = define_D(input_nc, ndf, netD, n_layers_D, norm, init_type, init_gain, gpus_ids)

        self.criterionGAN = GANLoss(gan_mode).to(device)
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()

        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=lr, betas=(beta, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=lr, betas=(beta, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        self.direction = direction
        self.device = device

        self.fake_A_pool = ImagePool(pool_size)
        self.fake_B_pool = ImagePool(pool_size)

        self.lambda_A = 10. # A->B->A的cycle loss 权重
        self.lambda_B = 10. # B->A->B的cycle loss 权重
        self.lambda_idt = 0.5


    def set_input(self, x):
        AtoB = self.direction == 'AtoB'
        self.real_A = x['A' if AtoB else 'B'].to(self.device)
        self.real_B = x['B' if AtoB else 'A'].to(self.device)
        self.img_paths = x['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A) # G_A(A)
        self.recy_A = self.netG_B(self.fake_B) # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)
        self.recy_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        pred_fake = netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D =  (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        if self.lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.lambda_B * self.lambda_idt

            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.lambda_A * self.lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss ||G_B(G_A(A)) - A ||
        self.loss_cycle_A = self.criterionCycle(self.recy_A, self.real_A) * self.lambda_A

        # Backward cycle loss ||G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.recy_B, self.real_B) * self.lambda_B

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

