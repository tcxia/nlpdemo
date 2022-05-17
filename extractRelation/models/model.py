# -*- coding: utf-8 -*-
'''
# Created on 2021/01/11 14:20:53
# @filename: model.py
# @author: tcxia
'''

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel


class SentenceRE(nn.Module):
    def __init__(self, pretrained_model_path, embedding_dim, tag_size, dropout) -> None:
        super().__init__()

        self.pretained_model_path = pretrained_model_path
        self.embedding_dim = embedding_dim
        self.tag_size = tag_size

        self.bert_model = BertModel.from_pretrained(self.pretained_model_path)

        self.embed = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.Tanh()
        self.norm = nn.LayerNorm(self.embedding_dim * 3)
        self.pred = nn.Linear(self.embedding_dim * 3, self.tag_size)

    def forward(self, token_ids, token_type_ids, attention_mask, e1_mask, e2_mask):
        sequence_out, pooled_output = self.bert_model(
            input_ids=token_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False)

        e1_h = self.entity_avg(sequence_out, e1_mask)
        e2_h = self.entity_avg(sequence_out, e2_mask)
        e1_h = self.activation(self.embed(e1_h))
        e2_h = self.activation(self.embed(e2_h))

        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        concat_h = self.norm(concat_h)
        logits = self.pred(self.drop(concat_h))
        return logits

    @staticmethod
    def entity_avg(hid_out, e_mask):
        e_mask_unsqueeze = e_mask.unsqueeze(1)
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)

        sum_vec = torch.bmm(e_mask_unsqueeze.float(), hid_out).squeeze(1)
        avg_vec = sum_vec.float() / length_tensor.float()
        return avg_vec
