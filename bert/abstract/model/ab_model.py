# -*- coding: utf-8 -*-
'''
# Created on 11 月-23-20 15:44
# ab_model.py
# @author: tcxia
'''


import torch
import torch.nn as nn


class AbstractModel(nn.Module):
    def __init__(self, bertConfig, bertModel, bertLMPredictionHead, vocab_size, device) -> None:
        super(AbstractModel, self).__init__()
        self.vocab_size = vocab_size
        self.config = bertConfig(self.vocab_size)

        self.model = bertModel(self.config)
        self.decoder = bertLMPredictionHead(
            self.config, self.model.embeddings.word_embeddings.weight)

        self.hidden_dim = self.config.hidden_size
        self.device = device

    def forward(self, inputs, token_type_id, labels=None):
        input_shape = inputs.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]


        ones = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32, device=self.device)
        a_mask = ones.tril()
        s_ex12 = token_type_id.unsqueeze(1).unsqueeze(2).float()
        s_ex13 = token_type_id.unsqueeze(1).unsqueeze(3).float()
        a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask

        enc_layers, _ = self.model(inputs,
                                   token_type_ids=token_type_id,
                                   attention_mask=a_mask)
        sequece_out = enc_layers[-1]
        predictions = self.decoder(sequece_out)

        if labels is not None:
            predictions = predictions[:, :-1].contiguous()
            target_mask = token_type_id[:, 1:].contiguous()
            loss = self.compute_loss(predictions, labels, target_mask)
            return predictions, loss

        return predictions

    def compute_loss(self, predictions, labels, target_mask):
        predictions = predictions.view(-1, self.vocab_size)
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum()
