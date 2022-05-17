# -*- coding: utf-8 -*-
'''
# Created on 11 月-18-20 10:49
# decoder.py
# @author: tcxia
'''

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout, attention, device) -> None:
        super().__init__()
        self.device = device
        
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, dec_hid_dim, batch_first=True, num_layers=num_layers, dropout=dropout)

        self.out = nn.Linear(enc_hid_dim + dec_hid_dim, output_dim)

    def forward(self, src, queries, hidden):
        src = src.unsqueeze(1)

        embedded = self.embedding(src)
        embedded = self.dropout(embedded) # [bacth_size, src_len, emb_dim]


        output, hidden = self.rnn(embedded, hidden)
        output = output.squeeze()

        # print(output.shape)
        # print(queries.shape)
        a = self.attention(output, queries)
        a = a.unsqueeze(1)

        weighted = torch.bmm(a, queries)

        weighted = weighted.squeeze(1)

        output = self.out(torch.cat((output, weighted), dim=1))

        return output, hidden
