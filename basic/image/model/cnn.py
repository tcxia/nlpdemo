# -*- coding: utf-8 -*-
'''
# Created on 12月-02-20 14:36
# @filename: cnn.py
# @author: tcxia
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()

        # torch.nn.Conv2d(in_channels,
        #                 out_channels,
        #                 kernel_size,
        #                 stride=1,
        #                 padding=0,
        #                 dilation=1,
        #                 groups=1,
        #                 bias=True,
        #                 padding_mode='zeros')

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)


    def forward(self, x):
        # input: [N, C_in, H_in, W_in]
        # output: [N, C_out, H_out, W_out]

        # H_out = (H_in+2×padding[0]−dilation[0]×(kernel_size[0]−1)−1) / stride[0] + 1
        # W_out = (W_in+2×padding[1]−dilation[1]×(kernel_size[1]−1)−1) / stride[1] + 1


        # [bacth_size, channel_in, H, W]
        # x : [10, 1, 28, 28]

        x = F.relu(self.conv1(x)) # [10, 20, 24, 24]

        x = F.max_pool2d(x, 2, 2) # [10 , 20, 12, 12]

        x = F.relu(self.conv2(x)) # [10, 50, 8, 8]

        x = F.max_pool2d(x, 2, 2) # [10, 50, 4, 4]

        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # [10, 10]


if __name__ == "__main__":
    model = CNNNet()
    x = torch.rand((10, 1, 28, 28))
    # print(x.shape)

    output = model(x)
    print(output.shape)
