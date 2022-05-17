# -*- coding: utf-8 -*-
'''
# Created on 2021/01/13 15:51:09
# @filename: model.py
# @author: tcxia
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertForSequenceClassification, XLNetForSequenceClassification
from transformers import XLNetConfig, BertConfig

class TextMatch_Bert(nn.Module):
    def __init__(self, pretrained_path) -> None:
        super().__init__()
        self.bert_pretrained = BertForSequenceClassification.from_pretrained(pretrained_path, num_labels=2)
        for param in self.bert_pretrained.parameters():
            param.requires_grad = True

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert_pretrained(input_ids=batch_seqs, attention_mask=batch_seq_masks, token_type_ids=batch_seq_segments, labels=labels)[:2]
        prob = F.softmax(logits, dim=-1)
        return loss, logits, prob


class TextMatch_Bert_test(nn.Module):
    def __init__(self, json_path) -> None:
        super().__init__()
        config = BertConfig.from_pretrained(json_path)
        self.bert = BertForSequenceClassification(config)

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments):
        output = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks, token_type_ids=batch_seq_segments)
        return output


