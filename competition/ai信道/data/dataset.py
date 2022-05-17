# -*- coding: utf-8 -*-
'''
# Created on 2021/01/15 13:00:28
# @filename: dataset.py
# @author: tcxia
'''

import torch.utils.data as tud
import scipy.io as sio
import numpy as np

CHANNEL_SHAPE_DIM1 = 24
CHANNEL_SHAPE_DIM2 = 16
CHANNEL_SHAPE_DIM3 = 2
#=======================================================================================================================
#=======================================================================================================================
# Data Loader Class Defining
class DatasetMIMI(tud.Dataset):
    def __init__(self, matData):
        self.matdata = matData

    def __getitem__(self, index):
        return self.matdata[index]

    def __len__(self):
        return self.matdata.shape[0]


if __name__ == "__main__":
    mat_path = '/data/nlp_dataset/H_4T4R.mat'
    mat_data = sio.loadmat(mat_path)
    data = mat_data['H_4T4R']
    data = data.astype('float32')

    data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
    print(data.shape)