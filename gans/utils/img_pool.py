# -*- coding: utf-8 -*-
'''
# Created on 2021/01/19 14:26:40
# @filename: img_pool.py
# @author: tcxia
'''

import torch
import random

class ImagePool():
    def __init__(self, pool_size) -> None:
        super().__init__()

        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.imgs = []
    
    def query(self, imgs):
        if self.pool_size == 0:
            return imgs
        
        ret_imgs = []
        for img in imgs:
            img = torch.unsqueeze(img.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.imgs.append(img)
                ret_imgs.append(img)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.imgs[random_id].clone()
                    self.img[random_id] = img
                    ret_imgs.append(tmp)
                else:
                    ret_imgs.append(img)
        ret_imgs = torch.cat(ret_imgs, 0)
        return ret_imgs