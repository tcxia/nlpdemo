# -*- coding: utf-8 -*-
'''
# Created on 12月-07-20 12:44
# @filename: layers.py
# @author: tcxia
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import math



class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6) -> None:
        super().__init__()
        self.d_model = d_model

        # 可训练参数
        self.alpha = nn.Parameter(torch.ones(self.d_model)) # 全1
        self.bias = nn.Parameter(torch.zeros(self.d_model)) # 全0

        self.eps = eps

    def forward(self, x):
        # x: [batch_size, max_len, d_model]
        # x.mean(dim=-1, keepdim) / x.std: [batch_size, max_len, 1]
        return self.alpha * ( x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout_rate) -> None:
        super().__init__()

        self.d_model = d_model # 512
        self.d_k = d_model // heads # 512 / 8 = 64
        self.heads = heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0) #[bacth_size, seqlen]

        # [batch_size, seqlen, d_model]
        k = self.k(k).view(batch_size, -1, self.heads, self.d_k) # [batch_size, seq_len, d_model] -> [batch_size, seq_len, heads, d_k]
        q = self.q(q).view(batch_size, -1, self.heads, self.d_k)
        v = self.v(v).view(batch_size, -1, self.heads, self.d_k)

        # [batch_size, heads, seq_len, d_k]
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # [batch_size, heads, seq_len, d_k]
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # [batch_size, seq_len, d_model]
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # [batch_sizem, seq_ken, d_model]
        output = self.out(concat)
        return output


    def attention(self, q, k, v, d_k, mask=None, dropout=None):

        # q, k, v: [batch_size, heads, seq_len, d_k]
        # k.T: [batch_size, heads, d_k, seq_len] 
        # socres: [batch_size, heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask==0, -1e9)

        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        # [batch_size, heads, seq_len, d_k]
        output = torch.matmul(scores, v)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout_rate=0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout_rate, d_ff=2048) -> None:
        super().__init__()

        self.norm = Norm(d_model)
        self.attn = MultiHeadAttention(d_model, heads, dropout_rate)
        self.ffnn = FeedForward(d_model,d_ff, dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x, mask):
        # x: [batch_size, max_len, d_model]
        new_x = self.norm(x)  # [batch_size, max_len, d_model]

        # new_x直接拆分成q, k, v
        x = x + self.dropout(self.attn(new_x, new_x, new_x, mask))   # ADD

        new_x = self.norm(x) # layerNorm

        x = x + self.dropout(self.ffnn(new_x)) # ADD
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout_rate) -> None:
        super().__init__()

        self.norm = Norm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

        self.attn = MultiHeadAttention(d_model, heads, dropout_rate)
        self.ffnn = FeedForward(d_model)

    def forward(self, x, encoder_outputs, src_mask, trg_mask):
        # x: [batch_size, seq_len, d_model]
        new_x = self.norm(x)
        
        x = x + self.dropout(self.attn(new_x, new_x, new_x, trg_mask)) # ADD

        new_x = self.norm(x) # layerNorm

        # q -- > x
        # k --> encoder_outputs
        # v --> encoder_outputs
        x = x + self.dropout(self.attn(x, encoder_outputs, encoder_outputs, src_mask))

        new_x = self.norm(x) # layerNorm
        x = x + self.dropout(self.ffnn(new_x)) # 
        return x
