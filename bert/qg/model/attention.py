# -*- coding: utf-8 -*-
'''
# Created on 11 月-18-20 10:31
# attention.py
# @author: tcxia
'''

import torch
import torch.nn as nn
import torch.nn.functional as f


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, attention_hidden_size) -> None:
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, attention_hidden_size)
        self.v = nn.Parameter(torch.rand(attention_hidden_size), requires_grad=True)

    def forward(self, key, queries):
        # queries: [bs, len]
        batch_size = queries.shape[0]
        src_len = queries.shape[1]

        key = key.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(self.attn(torch.cat((key, queries), dim=2)))

        v = self.v.repeat(batch_size, 1).unsqueeze(2)

        attention = torch.bmm(energy, v).squeeze(2)

        return f.softmax(attention, dim=1)

if __name__ == "__main__":
    attn = Attention(100, 90, 150)
    print(attn)





