# -*- coding: utf-8 -*-
'''
# Created on 2021/04/14 13:11:57
# @filename: xlnet.py
# @author: tcxia
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import XLNetForMultipleChoice


class XLNetQA(nn.Module):
    def __init__(self, pretrained_path) -> None:
        super().__init__()
        self.xlnet = XLNetForMultipleChoice.from_pretrained(pretrained_path)
        for param in self.xlnet.parameters():
            param.requires_grad = True

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        output = self.xlnet(
            input_ids=batch_seqs,
            attention_mask=batch_seq_masks,
            token_type_ids=batch_seq_segments,
            labels=labels)
        return output