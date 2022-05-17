# -*- coding: utf-8 -*-
'''
# Created on 12月-07-20 11:09
# @filename: models.py
# @author: tcxia
'''
import torch
from torch.autograd import Variable
import torch.nn as nn

import math
import copy


from model.layers import EncoderLayer, DecoderLayer, Norm


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 词向量编码
class EmbedEncoder(nn.Module):
    def __init__(self, vocab_size, d_model) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

# 位置编码
class PosEncoder(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len=200, device='cpu') -> None:
        super().__init__()
        self.d_model = d_model # 512
        self.dropout = nn.Dropout(dropout_rate)
        self.device = device

        # [200, 512]
        self.posEmbed = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                self.posEmbed[pos, i] = math.sin(pos / (10000 ** ((2*i) / d_model)))
                self.posEmbed[pos, i + 1] = math.cos(pos / (10000 ** ((2*(i+1)) / d_model)))
        # [1, 200, 512]
        self.posEmbed = self.posEmbed.unsqueeze(0)

    def forward(self, x):
        # make embedding relatively larger

        # x: [batch_size, seq_len, d_model]
        x = x * math.sqrt(self.d_model) # ******
        seq_len = x.size(1)

        posEmbed = Variable(self.posEmbed[:, :seq_len], requires_grad=False)

        x = x.to(self.device)
        posEmbed = posEmbed.to(self.device)

        x = x + posEmbed
        return self.dropout(x)



class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, dropout_rate, max_len, heads, N, device='cpu') -> None:
        super().__init__()

        self.N = N # 编码堆叠层数
        # self.embed = nn.Embedding(vocab_size, d_model)
        self.embed = EmbedEncoder(vocab_size, d_model)
        self.posEmbed = PosEncoder(d_model, dropout_rate, max_len, device)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout_rate), N)
        self.norm = Norm(d_model)


    def forward(self, src, mask):
        # src: [batch_size, max_len]
        x = self.embed(src) # [batch_size, max_len, d_model]
        x = self.posEmbed(x) # [batch_size, max_len, d_model]
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, N, heads, vocab_size, d_model, dropout_rate, max_len, device='cpu') -> None:
        super().__init__()
        self.N = N
        # self.embed = nn.Embedding(vocab_size, d_model)
        self.embed = EmbedEncoder(vocab_size, d_model)
        self.posEmbed = PosEncoder(d_model, dropout_rate, max_len, device)

        self.layers = get_clones(DecoderLayer(d_model, heads, dropout_rate), N)

        self.norm = Norm(d_model)

        self.device = device

    def forward(self, trg, encoder_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.posEmbed(x) # [batch_size, seq_len, d_model]
        for i in range(self.N):
            x = self.layers[i](x, encoder_outputs, src_mask, trg_mask)
        return self.norm(x)


class TransformModel(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout_rate, max_len, device='cpu') -> None:
        super().__init__()

        self.encoder = Encoder(src_vocab, d_model, dropout_rate, max_len, heads, N, device)
        self.decoder = Decoder(N, heads, trg_vocab, d_model, dropout_rate, max_len, device)
        self.out = nn.Linear(d_model, trg_vocab)


    def forward(self, src, trg, src_mask, trg_mask):

        # [batch_size, seq_len, d_model]
        encoder_outputs = self.encoder(src, src_mask)

        decoder_outputs = self.decoder(trg, encoder_outputs, src_mask, trg_mask)
        output = self.out(decoder_outputs)
        return output
