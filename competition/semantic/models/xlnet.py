# -*- coding: utf-8 -*-
'''
# Created on 2021/03/30 10:10:45
# @filename: xlnet.py
# @author: tcxia
'''

import torch
import torch.nn as nn
from transformers import XLNetConfig, XLNetForSequenceClassification
import torch.nn.functional as F

class TextMatch_XLNet(nn.Module):
    def __init__(self, pretrained_path) -> None:
        super().__init__()
        self.xlnet = XLNetForSequenceClassification.from_pretrained(pretrained_path, num_labels=2)
        for param in self.xlnet.parameters():
            param.requires_grad = True

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.xlnet(input_ids=batch_seqs,
                                  attention_mask=batch_seq_masks,
                                  token_type_ids=batch_seq_segments,
                                  labels=labels)[:2]
        prob = F.softmax(logits, dim=-1)
        return loss, logits, prob


class TextMatch_XLNet_test(nn.Module):
    def __init__(self, pretrained_path) -> None:
        super().__init__()
        config = XLNetConfig.from_pretrained(pretrained_path)
        self.xlnet = XLNetForSequenceClassification(config)

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments):
        output = self.xlnet(input_ids=batch_seqs,
                                  attention_mask=batch_seq_masks,
                                  token_type_ids=batch_seq_segments)
        return output
