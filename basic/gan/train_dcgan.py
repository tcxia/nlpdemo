# -*- coding: utf-8 -*-
'''
# Created on 12月-03-20 16:19
# @filename: train_dcgan.py
# @author: tcxia
'''




from loguru import logger
import torch
import torch.nn as nn
import torch.utils.data as tud
from torchvision import datasets, transforms


from model.dcgan import GenNet, DisNet

root_data = '/data/nlp_dataset/celeba'
image_size = 64
bacth_size = 16
nz = 100
ngf = 64
ndf = 64
nc = 3
epochs = 10

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def Trainer(epoch, train_dataloader, d_model, g_model, d_optim, g_optim, loss_fn, device):

    for i, data in enumerate(train_dataloader):

        d_model.zero_grad()

        real_images = data[0].to(device)
        b_size = real_images.size(0)
        label = torch.ones(b_size).to(device)
        output = d_model(real_images).view(-1)

        real_loss = loss_fn(output, label)
        real_loss.backward()
        D_x = output.mean().item()


        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = g_model(noise)
        label.fill_(0)
        output = d_model(fake_images.detach()).view(-1)
        fake_loss = loss_fn(output, label)
        fake_loss.backward()
        D_gz1 = output.mean().item()
        loss_D = real_loss + fake_loss
        d_optim.step()

        g_model.zero_grad()
        label.fill_(1)
        output = d_model(fake_images).view(-1)
        loss_G = loss_fn(output, label)
        loss_G.backward()
        D_gz2 = output.mean().item()
        g_optim.step()

        if i % 50 == 0:
            logger.info(
                "Epoch: [{}/{}] | Step: [{}/{}] | d-Loss: {:.4f} | g-Loss: {:.4f} | D_x: {:.4f} | D_gz: {}/{}"
                .format(epoch, epochs, i, len(train_dataloader), loss_D.item(), loss_G.item(), D_x, D_gz1, D_gz2))


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    data = datasets.ImageFolder(root=root_data, transform=transform)

    train_data = tud.DataLoader(data, batch_size=bacth_size, shuffle=True, num_workers=0)


    netG = GenNet(nz, ngf, nc).to(device)
    netG.apply(weight_init)

    netD = DisNet(nc, ndf).to(device)
    netD.apply(weight_init)

    loss_fn = nn.BCELoss()
    d_optim = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.99))
    g_optim = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.99))


    for epoch in range(epochs):
        Trainer(epoch, train_data, netD, netG, d_optim, g_optim, loss_fn, device)