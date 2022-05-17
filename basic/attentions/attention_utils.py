# -*- coding: utf-8 -*-
"""
# Created on 12月-17-20 15:19
# @filename: attention_utils.py
# @author: tcxia
"""
import torch
import torch.nn.functional as F
import numpy as np


# attention 操作函数
# 长度掩码，主要对于句子生成对应的mask
def create_src_lengths_mask(batch_size, src_lengths, max_src_len=None):
    if max_src_len is None:
        max_src_len = int(src_lengths.max())

    src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_src_len)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)
    # [batch_size, max_src_len]
    return (src_indices < src_lengths).int().detach()


# 对于被mask的句子的softmax操作
def masked_softmax(scores, src_lengths, src_length_masking=True):
    if src_length_masking:
        bsz, max_src_len = scores.size()
        src_mask = create_src_lengths_mask(bsz, src_lengths)
        scores = scores.masked_fill(src_mask == 0, -np.inf)
    return F.softmax(scores.float(), dim=-1).type_as(scores)
