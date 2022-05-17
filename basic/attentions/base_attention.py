# -*- coding: utf-8 -*-
"""
# Created on 12月-17-20 14:48
# @filename: base_attention.py
# @author: tcxia
"""

import torch.nn as nn


# 基本attention操作，主要通过decoder和context生成
class BaseAttention(nn.Module):
    def __init__(self, decoder_hidden_state_dim, context_dim) -> None:
        super().__init__()
        self.decoder_hidden_state_dim = decoder_hidden_state_dim
        self.context_dim = context_dim

    def forward(self, decoder_state, source_hids, src_lengths):
        # decoder_state: [bsz, decoder_hidden_state_dim]
        # source_hids: [srclen, bsz, context_dim]
        # src_lengths: [bsz, 1]
        pass
