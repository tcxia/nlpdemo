# -*- coding: utf-8 -*-
"""
# Created on 12月-17-20 17:38
# @filename: pooling_attention.py
# @author: tcxia
"""

import torch
from torch.autograd import Variable
from base_attention import BaseAttention
from attention_utils import _, create_src_lengths_mask


class PoolingAttention(BaseAttention):
    def __init__(self, decoder_hidden_state_dim, context_dim, pool_type="mean") -> None:
        super().__init__()

        self.pool_type = pool_type

    def forward(self, decoder_state, source_hids, src_lengths):
        assert self.decoder_hidden_state_dim == self.context_dim

        max_src_len = source_hids.size()[0]
        assert max_src_len == src_lengths.data.max()

        bsz = source_hids.size()[1]

        src_mask = (
            create_src_lengths_mask(bsz, src_lengths)
            .type_as(source_hids)
            .t()
            .unsqueeze(2)
        )

        if self.pool_type == "mean":
            denom = src_lengths.view(1, bsz, 1).type_as(source_hids)  # [1, bsz, 1]
            masked_hiddens = source_hids * src_mask  # [srclen, bsz, context_dim]
            context = (masked_hiddens / denom).sum(dim=0)
        elif self.pool_type == "max":
            masked_hiddens = source_hids - 10e6 * (1 - src_mask)
            context = masked_hiddens.max(dim=0)[0]
        else:
            raise ValueError("pooling type is not supported!")

        attn_scores = Variable(
            torch.ones(src_mask.shape[1], src_mask.shape[0]).type_as(source_hids.data),
            requires_grad=False,
        ).t()
        return context, attn_scores


class MaxPoolingAttention(PoolingAttention):
    def __init__(self, decoder_hidden_state_dim, context_dim, pool_type) -> None:
        super().__init__(decoder_hidden_state_dim, context_dim, pool_type="max")


class MeanPoolingAttention(PoolingAttention):
    def __init__(self, decoder_hidden_state_dim, context_dim, pool_type) -> None:
        super().__init__(decoder_hidden_state_dim, context_dim, pool_type="mean")
