# -*- coding: utf-8 -*-
"""
# Created on 12月-17-20 14:56
# @filename: dot_attention.py
# @author: tcxia
"""

import torch
import torch.nn as nn

from base_attention import BaseAttention
from attention_utils import masked_softmax


# dot attention, 主要通过dot操作完成attention功能
class DotAttention(BaseAttention):
    def __init__(
        self,
        decoder_hidden_state_dim,
        context_dim,
        force_proj=False,
        src_length_masking=True,
    ) -> None:
        super().__init__()

        self.input_proj = None
        if force_proj or decoder_hidden_state_dim != context_dim:
            self.input_proj = nn.Linear(
                decoder_hidden_state_dim, context_dim, bias=True
            )

        self.src_length_masking = src_length_masking

    def forward(self, decoder_state, source_hids, src_lengths):

        source_hids = source_hids.transpose(0, 1)  # [bsz, srclen, context_dim]

        if self.input_proj is not None:
            decoder_state = self.input_proj(decoder_state)  # [bsz, context_dim]

        # [bsz, srclen]
        attn_scores = torch.bmm(source_hids, decoder_state.unsqueeze(2)).squeeze(2)

        # [bsz, maxlen]
        normalized_masked_attn_scores = masked_softmax(
            attn_scores, src_lengths, self.src_length_masking
        )

        attn_weighted_context = (
            (source_hids * normalized_masked_attn_scores.unsqueeze(2))
            .contiguous()
            .sum(1)
        )

        return attn_weighted_context, normalized_masked_attn_scores
