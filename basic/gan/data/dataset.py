# -*- coding: utf-8 -*-
'''
# Created on 12月-02-20 19:27
# @filename: dataset.py
# @author: tcxia
'''

from PIL import Image
import numpy as np

from torchvision import transforms

def load_image(image_path, transform=None, max_size=None, shape=None):
    img = Image.open(image_path)
    if max_size:
        scale = max_size / max(img.size)
        size = np.array(img.size) * scale
        img = img.resize(size.astype(int), Image.ANTIALIAS)
    
    if shape:
        img = img.resize(shape, Image.LANCZOS)

    if transform:
        img = transform(img).unsqueeze(0)
    
    return img

    
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    content = load_image('content.jpeg', transform=transform, max_size=400)
    print(content.shape) # [1, 3, 228, 400]

    style = load_image('style.jpg', transform=transform, shape=[content.shape[2], content.shape[3]])
    print(style.shape)

