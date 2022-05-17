# -*- coding: utf-8 -*-
'''
# Created on 2021/01/19 16:21:29
# @filename: train.py
# @author: tcxia
'''
import torch



real_label = 1
fake_label = 0
nz = 100 # 随机噪声的维度

def Train(netD, netG, optimizer_D, optimizer_G, criteion, train_loader, epoches, device):

    for epoch in range(epoches):
        for i, data in enumerate(train_loader):


            ##
            # (1) 更新判别器网络: maximize log(D(x)) + log(1 - D(G(z)))
            ##
            netD.zero_grad()

            # 先用全部为真实图像的批次训练
            real = data[0].to(device)
            batch_size = real.size(0)
            label = torch.full((batch_size, ), real_label, device=device)
            # 得到D的预测结果
            output = netD(real).view(-1)
            # 计算交叉熵损失
            loss_real = criteion(output, label)
            # 反向传播
            loss_real.backward()

            # 构造全部为生成图像的批次训练
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            # 得到D的预测结果
            output = netD(fake.detach()).view(-1)
            # 计算交叉熵损失
            loss_fake = criteion(output, label)
            # 反向传播
            loss_fake.backward()

            # 总得D的损失
            loss = loss_real + loss_fake
            # 更新判别器参数
            optimizer_D.step()

            ##
            # (2) 更新生成器网络: maximize log(D(G(z)))
            ##
            netG.zero_grad() # 梯度清零
            label.fill_(real_label) # 对于判别器来说，生成的假的图片的label是真实的
            # 得到G的预测结果
            output = netD(fake).view(-1)
            # 计算交叉熵损失
            loss_G = criteion(output, label)
            # 反向传播
            loss_G.backward()
            # 更新生成器参数
            optimizer_G.step()
