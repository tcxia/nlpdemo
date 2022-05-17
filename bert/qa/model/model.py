# -*- coding: utf-8 -*-
'''
# Created on 11 月-19-20 16:46
# model.py
# @author: tcxia
'''


import torch
import torch.nn as nn
import numpy as np
from model.crf import CRF

class NetQA(nn.Module):
    def __init__(self, pretrain_model, device) -> None:
        super().__init__()
        self.bert_model = pretrain_model
        self.hidden_size = self.bert_model.config.hidden_size
        self.labelnum = 2
        self.crf_fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.hidden_size, self.labelnum + 2, bias=True))

        self.device = device

        # kwargs = dict({'target_size': self.labelnum, 'device': self.device})
        self.crf = CRF(self.labelnum, self.device)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

        self.fc2 = nn.Linear(self.hidden_size, 2, bias=True)

    def forward(self, token_id_l, token_type_id_l, answer_offset_l, answer_seq_label_l, IsQA_l):
        # print(token_id_l)
        tokens_id_2d = torch.LongTensor(token_id_l).to(self.device)
        token_type_id_2d = torch.LongTensor(token_type_id_l).to(self.device)

        bacth_size, seg_length = tokens_id_2d[:, 1:].size()

        y_2d = torch.LongTensor(answer_seq_label_l).to(self.device)[:, 1:]
        y_IsQA_2d = torch.LongTensor(IsQA_l).to(self.device)

        if self.training:
            self.bert_model.train()
            emb, _ = self.bert_model(input_ids=tokens_id_2d, token_type_ids=token_type_id_2d)

        else:
            self.bert_model.eval()
            with torch.no_grad():
                emb, _ = self.bert_model(input_ids=tokens_id_2d, token_type_ids=token_type_id_2d)

        cls_emb = emb[:, 0, :]
        IsQA_logits = self.fc2(cls_emb)
        IsQA_loss = self.CrossEntropyLoss(IsQA_logits, y_IsQA_2d)

        IsQA_prediction = IsQA_logits.argmax(dim=-1).unsqueeze(dim=-1)

        mask = np.ones(shape=[bacth_size, seg_length], dtype=np.uint8)
        mask = torch.BoolTensor(mask).to(self.device) #
        
        crf_logits = self.crf_fc(emb[:, 1:, :])
        crf_loss = self.crf.neg_log_likelihood_loss(feats=crf_logits, mask=mask, tags=y_2d)
        _, crf_pred = self.crf(feats=crf_logits, mask=mask)

        return IsQA_prediction, crf_pred, IsQA_loss, crf_loss, y_2d, y_IsQA_2d.unsqueeze(dim=-1)
