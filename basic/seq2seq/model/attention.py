# -*- coding: utf-8 -*-
'''
# Created on 12月-08-20 10:23
# @filename: attention.py
# @author: tcxia
'''



import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout_rate, word_vec=None) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        # 词向量编码
        self.embed = nn.Embedding(vocab_size, embed_size)

        if word_vec is not None:
            self.init_embed_weight(word_vec)
        # GRU模型
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout_rate)

        # 全连接层
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, x, x_lengths):
        # x: [batch_size, max_len]
        # x_length: [batch_size, seq_len]
        sorted_len, sorted_idx = x_lengths.sort(0, descending=True)

        x_sorted = x[sorted_idx.long()] #
        embedded = self.dropout(self.embed(x_sorted)) # [batch_size, max_len, embed_size]

        # [batch_size, max_len, embed_size]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(), batch_first=True)

        # num_directions:2
        # patch_out: [batch_size, max_len, 2*enc_hidden_size]
        # hid: [1 * 2, batch_size, enc_hidden_size]
        packed_out, hid = self.rnn(packed_embedded)

        # [batch_size, max_len, enc_hidden_size]
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        #
        _, origin_idx = sorted_idx.sort(0, descending=True)


        # out: [batch_size, max_len, enc_hidden_size]
        out = out[origin_idx.long()].contiguous()
        # hid: [1 * 2, batch_size, enc_hidden_size]
        hid = hid[:, origin_idx.long()].contiguous()

        # concat 倒数最后一层和倒数第二层
        # [batch_size, enc_hidden_size]
        # [batch_size, 2 * enc_hidden_size]
        hid = torch.cat([hid[-2], hid[-1]], dim=1) #

        # [1, batch_size, dec_hidden_size]
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)

        return out, hid


    def init_embed_weight(self, word_vec):
        # word_vec: numpy

        word2vec = torch.from_numpy(word_vec)
        # unk = torch.randn(1, self.embed_size) / math.sqrt(self.embed_size)
        # pad = torch.zeros(1, self.embed_size)
        # return self.embed.weight.data.copy_(torch.cat([word2vec, unk, pad], 0))
        return self.embed.weight.data.copy_(word2vec)


class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size) -> None:
        super().__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size * 2, dec_hidden_size, bias=False)
        self.linear_out = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size)

    def forward(self, output, context, mask):
        # output: [batch_size, output_len, dec_hidden_size]
        # context: [batch_size, context_len, 2 * enc_hidden_size]

        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1) # context_len

        # [batch_size, input_len, dec_hidden_size]
        context_in = self.linear_in(context.view(batch_size * input_len, -1)).view(batch_size, input_len, -1)

        # context_in.transpose(1,2): [batch_size, dec_hidden_size, input_len]

        # [batch_size, output_len, input_len]
        attn = torch.bmm(output, context_in.transpose(1,2))

        attn.data.masked_fill_(mask, -1e6)

        # [batch_size, output_len, input_len]     *****
        attn = F.softmax(attn, dim=2)

        # [batch_size, output_len, 2*enc_hidden_size]
        context = torch.bmm(attn, context)

        # [batch_size, output_len, 2 * enc_hidden_size +  dec_hidden_size]
        output = torch.cat((context, output), dim=2)

        output = output.view(batch_size*output_len, -1)

        # [batch_size, output_len, dec_hidden_size] ******
        output = torch.tanh(self.linear_out(output))


        output = output.view(batch_size, output_len, -1)
        return output, attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout_rate) -> None:
        super().__init__()

        self.vocab_size = vocab_size

        # 词向量
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.attention =  Attention(enc_hidden_size, dec_hidden_size)

        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True)

        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ctx, ctx_lengths, y, y_lengths, hid):
        # ctx: [batch_size, ]
        # ctx_length: [batch_size, ]
        # y: [batch_size, max_len - 1]
        # y_lengths: [batch_size, seq_len - 1]
        # hid = [1, batch_size, dec_hidden_size]

        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]

        hid = hid[:, sorted_idx.long()]

        # [batch_size, max_len - 1, embed_size]
        y_sorted = self.dropout(self.embed(y_sorted))

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)


        # out: [batch_size, max_len - 1, enc_hidden_size]
        # hid: [1, batch_size, enc_hidden_size]
        out, hid = self.rnn(packed_seq, hid)

        unpacked_seq, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        _, origin_idx = sorted_idx.sort(0, descending=True)

        output_seq = unpacked_seq[origin_idx.long()].contiguous()
        hid = hid[:, origin_idx.long()].contiguous()

        mask = self.create_mask(y_lengths, ctx_lengths)

        # output: [batch_size, output_len, dec_hidden_size]
        # attn: [batch_size, output_len, input_len]
        output, attn = self.attention(output_seq, ctx, mask)

        # [batch_size, output_len, vocab_size]
        output = F.log_softmax(self.out(output), -1)

        return output, hid, attn

    def create_mask(self, x_len, y_len):
        device = x_len.device

        max_x_len = x_len.max()
        max_y_len = y_len.max()

        x_mask = torch.arange(max_x_len, device=device)[None, :] < x_len[:, None]
        y_mask = torch.arange(max_y_len, device=device)[None, :] < y_len[:, None]

        # mask = (1 - x_mask[:, :, None] * y_mask[:, None, :]).byte()
        mask = ~(x_mask[:, :, None] * y_mask[:, None, :]).bool()

        return mask

    # 词向量初始化
    def init_embed_weight(self, word_vec):
        # word_vec: numpy
        word2vec = torch.from_numpy(word_vec)
        unk = torch.randn(1, self.vocab_size) / math.sqrt(self.vocab_size)
        pad = torch.zeros(1, self.vocab_size)
        return self.embed.weight.data.copy_(torch.cat([word2vec, unk, pad], 0))

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        output, hid, attn = self.decoder(ctx=encoder_out, ctx_lengths=x_lengths, y=y, y_lengths=y_lengths, hid=hid)

        return output, attn


    def translate(self, x, x_lengths, y, max_length=100):
        encoder_out, hid = self.encoder(x, x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            output, hid, attn = self.decoder(
                ctx=encoder_out,
                ctx_lengths=x_lengths,
                y=y,
                y_lengths=torch.ones(batch_size).long().to(y.device),
                hid=hid)
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)
            attns.append(attn)

        return torch.cat(preds, 1), torch.cat(attns, 1)
