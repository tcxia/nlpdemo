# -*- coding: utf-8 -*-
'''
# Created on 12月-04-20 13:29
# @filename: plain.py
# @author: tcxia
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class PlainEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout_rate) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

        # batch_first: If True, then the input and output tensors are provided as (batch, seq, feature).
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, x_lengths):
        # 降序排列
        # dim=0, 按列排列， 1: 按行排列
        sorted_len, sorted_idx = x_lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx.long()]

        embedded = self.embed(x_sorted)
        embedded = self.dropout(embedded)

        packed_embed = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(), batch_first=True, enforce_sorted=True)
        packed_out, hid = self.rnn(packed_embed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, origin_idx = sorted_idx.sort(0, descending=True)
        out = out[origin_idx.long()].contiguous()
        hid = hid[:, origin_idx.long()].contiguous()

        return out, hid


class PlainDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout_rate) -> None:
        super().__init__()

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, y, y_lengths, hid):
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]

        hid = hid[:, sorted_idx.long()]

        y_sorted = self.dropout(self.embed(y_sorted))

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(packed_seq)

        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        _, origin_idx = sorted_idx.sort(0, descending=True)
        output_seq = unpacked[origin_idx.long()].contiguous()
        hid = hid[:, origin_idx.long()].contiguous()

        output = F.log_softmax(self.out(output_seq), -1)
        return output, hid


class PlainSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        output, hid = self.decoder(y, y_lengths, hid)
        return output, None
