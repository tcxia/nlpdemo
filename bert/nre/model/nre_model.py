# -*- coding: utf-8 -*-
'''
# Created on 11 月-25-20 19:22
# nre_model.py
# @author: tcxia
'''
import torch
import torch.nn as nn

class NRERelationExtra(nn.Module):
    def __init__(self, pred_num, vocab_size, BertConfig, BertModel, BertLayerNorm) -> None:
        super(NRERelationExtra, self).__init__()
        self.pred_num = pred_num

        self.vocab_size = vocab_size
        self.config = BertConfig(self.vocab_size)
        self.model = BertModel(self.config)
        self.layer_norm = BertLayerNorm(self.config.hidden_size)
        self.layer_norm_cond = BertLayerNorm(self.config.hidden_size, conditional=True)

        self.sub_pred = nn.Linear(self.config.hidden_size, 2)
        self.activation = nn.Sigmoid()
        self.obj_pred = nn.Linear(self.config.hidden_size, 2 * self.pred_num)


    def forward(self, text, sub_ids, device='cpu', sub_labels=None, obj_labels=None, use_layer_num=-1):
        if use_layer_num != -1:
            if use_layer_num < 0 or use_layer_num > 7:
                raise Exception("层数越界， 取值范围0-7, 默认为-1，取最后一层")

        text = text.to(device)
        sub_ids = sub_ids.to(device)

        self.target_mask = (text > 0).float()
        enc_layers, _ = self.model(text, output_all_encoded_layers=True)
        
        sequence_out = enc_layers[use_layer_num]
        sub_out = enc_layers[-2]

        sub_pred_out = self.sub_pred(sequence_out)
        sub_pred_act = self.activation(sub_pred_out)
        sub_pred_act = sub_pred_out**2

        sub_vec = self.extrac_sub(sub_out, sub_ids)
        obj_layer_norm = self.layer_norm_cond([sub_out, sub_vec])
        obj_pred_out = self.obj_pred(obj_layer_norm)
        obj_pred_act = self.activation(obj_pred_out)

        obj_pred_act = obj_pred_act**4

        batch_size, seq_len, target_size = obj_pred_act.shape

        obj_pred_act = obj_pred_act.reshape((batch_size, seq_len, int(target_size / 2), 2))

        predictions = obj_pred_act
        if sub_labels is not None and obj_labels is not None:
            sub_labels = sub_labels.to(device)
            obj_labels = obj_labels.to(device)
            loss = self.compute_loss(sub_pred_act, obj_pred_act, sub_labels, obj_labels)
            return predictions, loss
        else:
            return predictions


    def extrac_sub(self, output, sub_ids):
        batch_size = output.shape[0]
        hidden_size = output.shape[-1]
        start_end = torch.gather(output, index=sub_ids.unsqueeze(-1).expand((batch_size, 2, hidden_size)), dim=1)
        sub = torch.cat((start_end[:, 0], start_end[:, 1]), dim=-1)
        return sub

    def binary_crossentropy(self, labels, pred):
        labels = labels.float()
        loss = (-labels) * torch.log(pred) - (1.0 - labels) * torch.log(1.0 - pred)
        return loss

    def compute_loss(self, sub_preds, obj_preds, sub_labels, obj_labels):
        sub_loss = self.binary_crossentropy(sub_labels, sub_preds)
        sub_loss = torch.mean(sub_loss, dim=2)
        sub_loss = (sub_loss * self.target_mask).sum() / self.target_mask.sum()

        obj_loss = self.binary_crossentropy(obj_labels, obj_preds)
        obj_loss = torch.mean(obj_loss, dim=3).sum(dim=2)
        obj_loss = (obj_loss * self.target_mask).sum() / self.target_mask.sum()

        return sub_loss + obj_loss






