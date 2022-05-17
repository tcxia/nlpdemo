# -*- coding: utf-8 -*-
"""
# Created on 12月-17-20 16:02
# @filename: multihead_attention.py
# @author: tcxia
"""

from base_attention import BaseAttention
from attention_utils import _, create_src_lengths_mask


class MultiHeadAttention(BaseAttention):
    def __init__(
        self,
        decoder_hidden_state_dim,
        context_dim,
        nheads=1,
        unseen_mask=False,
        src_length_mask=True,
    ) -> None:
        super().__init__()
        assert decoder_hidden_state_dim == context_dim

        self.d_model = decoder_hidden_state_dim
        assert self.d_model % nheads == 0

        if unseen_mask:
            raise NotImplementedError(
                "Unseen mask not supported with sequential decoding"
            )

        self.src_length_mask = src_length_mask

    def forward(
        self, decoder_state, source_hids, src_lengths, squeeze=True, max_src_len=None
    ):
        bsz = decoder_state.shape[0]

        if decoder_state.dim() == 3:
            query = decoder_state
        elif decoder_state.dim() == 2:
            query = decoder_state.unsqueeze(1)
        else:
            raise ValueError("decode state must either 2 or 3 dim")

        query = query.transpose(0, 1)
        value = key = source_hids

        if src_lengths is not None and self.src_length_mask:
            src_len_mask_int = create_src_lengths_mask(bsz, src_lengths, max_src_len)
            src_len_mask = src_len_mask_int != 1

        # attn, attn_weights =
