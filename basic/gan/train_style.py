# -*- coding: utf-8 -*-
'''
# Created on 12月-02-20 19:26
# @filename: train.py
# @author: tcxia
'''
import os
import torch
from torchvision import transforms

from loguru import logger

from data.dataset import load_image
from model.vgg import VGGNet

learning_rate = 0.01
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

os.environ['TORCH_HOME'] = './vgg'

total_steps = 2000
style_weight = 100.


def Trainer(step, content, style, target, model, optimizer, device):

    model.eval()


    content = content.to(device)
    style = style.to(device)
    target = target.to(device)

    target_features = model(target)
    content_features = model(content)
    style_features = model(style)

    content_loss = 0.
    style_loss = 0.
    for tf, cf, sf in zip(target_features, content_features, style_features):

        content_loss += torch.mean((tf - cf)**2)

        _, c, h, w = tf.size()
        tf = tf.view(c, h*w)
        sf = sf.view(c, h*w)

        tf = torch.mm(tf, tf.t())
        sf = torch.mm(sf, sf.t())

        style_loss += torch.mean((tf - sf)**2)/(c*h*w)

    loss = content_loss + style_loss * style_weight

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        logger.info(
            "Step: [{}/{}] | Content-Loss: {:.4f} | Style-Loss: {:.4f}".format(
                step, total_steps, content_loss.item(), style_loss.item()))

def pred(target, denorm):
    img = target.clone().squeeze()
    img = denorm(img).clamp_(0, 1)
    return img


if __name__ == "__main__":
    content_path = './data/content.jpeg'
    style_path = './data/style.jpg'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    content = load_image(content_path, transform=transform, max_size=400)
    style = load_image(style_path, transform=transform, shape=[content.shape[2], content.shape[3]])


    model = VGGNet()
    model = model.to(device)

    # target自身的像素参数需要更新
    target = content.clone().requires_grad_(True)

    # betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
    # betas用于计算梯度及其平方的运行平均值的系数
    optimizer = torch.optim.Adam([target], lr=learning_rate, betas=[0.5, 0.99])

    # 训练
    for step in range(total_steps):
        Trainer(step, content, style, target, model, optimizer, device)

    # 展示迁移图片
    Denorm = transforms.Normalize([-2.12, -2.04, -1.80], [4.37, 4.46, 4.44])
    img = pred(target, Denorm)
    print(img.shape)