# -*- coding: utf-8 -*-
'''
# Created on 2021/04/02 10:36:17
# @filename: siameseGRU.py
# @author: tcxia
'''

import torch
import torch.nn as nn

class SiameseGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, out_dim, pretrained_weight, padding_idx, dropout_rate) -> None:
        super().__init__()
        # self.embed = nn.Embedding.from_pretrained(pretrained_weight,
        #                                           freeze=False,
        #                                           padding_idx=padding_idx)
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.gru = Encoder(input_size=embed_size, hidden_size=hidden_size, dropout_rate=dropout_rate)
        
        self.proj = nn.Linear(8 * hidden_size, out_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.padding_idx = padding_idx
        self.hidden_size = hidden_size

        self.init_params()

    def init_params(self):
        nn.init.xavier_normal_(self.proj)
        nn.init.constant_(self.proj.bias, 0)
    
    def forward(self, texta, textb):
        batch_size, max_len_a = texta.shape
        _, max_len_b = textb.shape
        lens_a = torch.sum(texta != self.padding_idx, dim=-1)
        lens_b = torch.sum(textb != self.padding_idx, dim=-1)

        embed_a = self.dropout(self.embed(texta))
        embed_b = self.dropout(self.embed(textb))

        output_a = self.gru(embed_a.float(), lens_a)
        output_a = torch.max(output_a, dim=1)[0]

        output_b = self.gru(embed_b.float(), lens_b)
        output_b = torch.max(output_b, dim=1)[0]

        sim = torch.cat([output_a, output_a * output_b, torch.abs(output_a - output_b), output_b], dim=1)
        return self.proj(sim)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=2,
                              dropout=dropout_rate,
                              bidirectional=True,
                              batch_first=True)
    
    def forward(self, sequence_batch, sequence_lengths):
        sorted_batch, sorted_seq_lens, _, restoration_index = self.sort_by_seq_lens(sequence_batch, sequence_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_seq_lens, batch_first=True)
        output, _ = self.encoder(packed_batch)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output.index_select(0, restoration_index)


    def sort_by_seq_lens(self, batch, sequence_lengths, descending=True):
        sorted_seq_lens, sorting_index = sequence_lengths.sort(0, descending=descending) # 根据序列长度排序，返回序列长度以及对应的索引
        sorted_batch = batch.index_select(0, sorting_index)
        idx_range = torch.arange(0, len(sequence_lengths)).type_as(sequence_lengths)
        _, reverse_mapping = sorting_index.sort(0, descending=True)
        restoration_index = idx_range.index_select(0, reverse_mapping)
        return sorted_batch, sorted_seq_lens, sorting_index, restoration_index


