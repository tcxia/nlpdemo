# -*- coding: utf-8 -*-
'''
# Created on 2021/03/30 14:15:39
# @filename: Siamese.py
# @author: tcxia
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size=300, num_layer=2, device="cpu") -> None:
        super().__init__()

        self.device = device

        # self.embed_dim = embeddings.shape[1]
        self.embed_dim = embed_dim

        self.word_emb = nn.Embedding(vocab_size, embed_dim)

        # self.word_emb.weight = nn.Parameter(torch.from_numpy(embeddings))
        # self.word_emb.float()
        # self.word_emb.weight.requires_grad = False
        # self.word_emb.to(device)

        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.gru = nn.LSTM(self.embed_dim, self.hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.h0 = self.init_hidden((2 * self.num_layer, 1, self.hidden_size))
        self.h0.to(device)
        self.fc = nn.Linear(50, 2)


    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal_(h0)
        return h0

    def dropout(self, v):
        return F.dropout(v, p=0.2, training=self.training)
    
    def forward_once(self, x):
        output, hidden = self.gru(x)
        return output

    def forward(self, sent1, sent2):
        p_encode = self.word_emb(sent1)
        # print(p_encode[0, 0, :])
        # print(p_encode.shape) #[128, 50, 100]
        h_encode = self.word_emb(sent2)
        p_encode = self.dropout(p_encode)
        h_encode = self.dropout(h_encode)

        encoding1 = self.forward_once(p_encode)
        encoding2 = self.forward_once(h_encode)

        sim = torch.exp(-torch.norm(encoding1 - encoding2, p=2, dim=-1, keepdim=True))
        x = self.fc(sim.squeeze(dim=-1))
        prob = F.softmax(x, dim=-1)
        return x, prob
