# -*- coding: utf-8 -*-
'''
# Created on 2021/01/20 10:42:17
# @filename: preprocess.py
# @author: tcxia
'''

import torch
import numpy as np


class Preprocessing():
    def __init__(self, max_position, hidden_dim, word2idx) -> None:
        super().__init__()

        self.max_position = max_position + 2
        self.hidden_dim = hidden_dim
        self.word2idx = word2idx

        self.pad_index = 0
        self.unk_index = 1
        self.cls_index = 2
        self.sep_index = 3
        self.mask_index = 4
        self.num_index = 5
        self.position_encoding = self.init_pos_encoding()

    def init_pos_encoding(self):
        pos_enc = np.array(
            [pos / np.power(10000, 2 * i / self.hidden_dim) 
            for i in range(self.hidden_dim)] 
            if pos != 0 else np.zeros(self.hidden_dim) 
            for pos in range(self.max_position)
        )
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])
        denominator = np.sqrt(np.sum(pos_enc**2, axis=1, keepdims=True))
        pos_enc = pos_enc / (denominator + 1e-8)
        return pos_enc

    def tokenize(self, text, dicts):
        return [dicts.get(i, self.unk_index) for i in text]
    
    def add_cls_seq(self, tokens):
        return [self.cls_index] + tokens + [self.sep_index]
    
    def add_cls_seq_padding(self, tokens):
        return [self.pad_index] + tokens + [self.pad_index]

    def __call__(self, text_list, max_seq_len):
        text_list_len = [len(i) for i in text_list]
        if max(text_list_len) > self.max_position - 2:
            raise AssertionError("max_seq_len exceeds the maximum length of pos encoding")
        
        batch_max_seq_len = max_seq_len + 2
        text_tokens = [self.tokenize(i, self.word2idx) for i in text_list]
        text_tokens = [self.add_cls_seq(i) for i in text_tokens]
        text_tokens = [torch.tensor(i) for i in text_tokens]
        text_tokens = torch.nn.utils.rnn.pad_sequence(text_tokens, batch_first=True)
        pos_enc = torch.from_numpy(self.pos_enc[:batch_max_seq_len]).astype(torch.FloatTensor)
        return text_tokens, pos_enc

