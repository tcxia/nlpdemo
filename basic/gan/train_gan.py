# -*- coding: utf-8 -*-
'''
# Created on 12月-03-20 13:33
# @filename: train_gan.py
# @author: tcxia
'''


from loguru import logger
import torch
import torch.nn as nn
import torch.utils.data as tud

from torchvision import datasets, transforms

epochs = 10
batch_size = 8
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

img_size = 784
hidden_size = 256

D = nn.Sequential(
    nn.Linear(img_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
)


latent_size = 64
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, img_size),
    nn.Tanh()
)


dNet = D.to(device)
gNet = G.to(device)

loss_fn = nn.BCELoss()
d_optimizer = torch.optim.Adam(dNet.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(gNet.parameters(), lr=0.0002)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


def Trainer(epoch, train_dataloader, d_model, g_model, d_optimizer, g_optimizer, device):

    for step, batch in enumerate(train_dataloader):
        images, _ = batch
        images = images.reshape(batch_size, img_size).to(device)

        # 生成 正负样本标签
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        outputs = d_model(images)
        d_loss_real = loss_fn(outputs, real_labels)
        real_score = outputs


        # 随机生成fake images
        z = torch.rand(batch_size, latent_size).to(device)
        fake_images = g_model(z)
        outputs = d_model(fake_images)
        d_loss_fake = loss_fn(outputs, fake_labels)
        fake_score = outputs

        # 优化D
        d_loss = d_loss_fake + d_loss_real

        reset_grad()
        
        d_loss.backward()
        d_optimizer.step()


        # 优化G
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = d_model(fake_images)
        g_loss = loss_fn(outputs, fake_labels)

        reset_grad()

        g_loss.backward()
        g_optimizer.step()

        if step % 1000 == 0:
            logger.info(
                "Epoch: [{}/{}] | Step: [{}/{}] | d_loss: {:.4f} | g_loss: {:.4f} | D(x): {:.4f} | D(G(z)): {:.4f}"
                .format(epoch, epochs, step, len(train_dataloader), d_loss.item(), g_loss.item(), real_score.mean().item(), fake_score.mean().item()))



if __name__ == "__main__":

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    mnist_data = datasets.MNIST('../img_cls/mnist_data', train=True, transform=transform)

    train_loader = tud.DataLoader(mnist_data, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        Trainer(epoch, train_loader, dNet, gNet, d_optimizer, g_optimizer, device)
