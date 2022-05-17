# -*- coding: utf-8 -*-
"""
# Created on 12月-17-20 15:41
# @filename: mlp_attention.py
# @author: tcxia
"""

import torch.nn as nn
import torch.nn.functional as F
from base_attention import BaseAttention
from attention_utils import masked_softmax


class MLPAttention(BaseAttention):
    def __init__(
        self, decoder_hidden_state_dim, context_dim, src_length_masking=True
    ) -> None:
        super().__init__()

        self.context_dim = context_dim
        self.attention_dim = context_dim

        self.encoder = nn.Linear(context_dim, self.attention_dim, bias=True)
        self.decoder = nn.Linear(
            decoder_hidden_state_dim, self.attention_dim, bias=False
        )
        self.scores = nn.Linear(self.attention_dim, 1, bias=False)

        self.src_length_masking = src_length_masking

    def forward(self, decoder_state, source_hids, src_lengths):
        src_len, bsz, _ = source_hids.size()
        flat_source_hids = source_hids.view(
            -1, self.context_dim
        )  # [src_len * bsz, context_dim]

        encoder_component = self.encoder(
            flat_source_hids
        )  # [src_len*bsz, attention_dim]
        encoder_component = encoder_component.view(src_len, bsz, -1)  #

        decoder_component = self.decoder(decoder_state)  # [bsz, attention_dim]

        hidden_att = F.tanh(
            (decoder_component.unsqueeze(0) + encoder_component).view(
                -1, self.attention_dim
            )
        )

        attn_scores = self.scores(hidden_att).view(src_len, bsz).t()

        normalized_masked_attn_scores = masked_softmax(
            attn_scores, src_lengths, self.src_length_masking
        ).t()

        attn_weight_context = (
            source_hids * normalized_masked_attn_scores.unsqueeze(2)
        ).sum(0)

        return attn_weight_context, normalized_masked_attn_scores
