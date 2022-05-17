# -*- coding: utf-8 -*-
'''
# Created on 2021/01/22 15:34:00
# @filename: model.py
# @author: tcxia
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class TuckER(nn.Module):
    def __init__(self, model_name, loss_type, l3_reg, d, d1, d2) -> None:
        super().__init__()
        self.model_name = model_name
        self.loss_type = loss_type
        self.bce_loss = nn.BCELoss()
        self.l3_reg = l3_reg
        
        self.do_batch_norm = True
        self.dropout = nn.Dropout()

        multipler = 3

        self.entity_dim = d1 * multipler
        
        self.E = nn.Embedding(len(d.entities), d1 * multipler, padding_idx=0)

        if self.loss_type == 'BCE':
            self.loss = self.bce_loss_func
        elif self.loss_type == 'CE':
            self.loss = self.ce_loss
        else:
            raise ValueError('loss type is invaild [%s]' % self.loss_type)

        if self.model in ['DistMult', 'RESCAL', 'SimplE', 'TuckER']:
            self.bn = nn.BatchNorm2d(d1 * multipler)
        else:
            self.bn = nn.BatchNorm2d(multipler)

        if self.model_name == 'DistMult':
            multipler = 1
            self.score_func = self.DistMult
        elif self.model_name == 'SimplE':
            multipler = 2
            self.score_func = self.SimplE
        elif self.model_name == 'ComplE':
            multipler = 2
            self.score_func = self.ComplE
        elif self.model_name == 'RESCAL':
            self.score_func = self.RESCAL
            multipler = 1
        elif self.model_name == 'TuckER':
            self.score_func = self.TuckER
            multipler = 1
            self.W = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), dtype=torch.float, device='cuda', requires_grad=True))
        else:
            raise ValueError('model name [%s] is error' % self.model_name)


        if self.model_name == 'RESCAL':
            self.R = nn.Embedding(len(d.relations), d1 * d1, padding_idx=0)
        elif self.model_name == 'TuckER':
            self.R = nn.Embedding(len(d.relations), d2, padding_idx=0)
        else:
            self.R = nn.Embedding(len(d.relations), d1 * multipler, padding_idx=0)

    def bce_loss_func(self, pred, true):
        loss = self.bce_loss(pred, true)
        if self.l3_reg:
            norm = torch.norm(self.E.weight.data, a=-1.0, b=1.0)
            loss += self.l3_reg * torch.sum(norm)
        
        return loss

    def ce_loss(self, pred, true):
        loss = F.log_softmax(pred, dim=-1)
        true = true / true.size(-1)
        loss = -torch.sum(pred * true)
        return loss
    

    def DistMult(self, head, relation):
        if self.do_batch_norm:
            head = self.bn(head)
        head = self.dropout(head)
        relation = self.dropout(relation)
        s = head * relation
        if self.do_batch_norm:
            s = self.bn(s)
        s = self.dropout(s)
        s = torch.mm(s, self.E.weight.transpose(1, 0))
        return s

    def SimplE(self, head, relation):
        if self.do_batch_norm:
            head = self.bn(head)
        head = self.dropout(head)
        relation = self.dropout(relation)
        s = head * relation
        s_head, s_tail = torch.chunk(s, 2, dim=1)
        s = torch.cat([s_tail, s_head], dim=1)
        if self.do_batch_norm:
            s = self.bn(s)
        s = self.dropout(s)
        s = torch.mm(s, self.E.weight.transpose(1, 0))
        s = 0.5 * s
        return s

    def ComplE(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        if self.do_batch_norm:
            head = self.bn(head)
        head = self.dropout(head)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        relation = self.dropout(relation)
        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.E.weight, 2, dim=1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation - im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        if self.do_batch_norm:
            score = self.bn(score)
        
        score = self.dropout(score)
        score = score.permute(1, 0, 2)
        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1, 0)) + torch.mm(im_score, im_tail.transpose(1, 0))
        return score

    def RESCAL(self, head, relation):
        if self.do_batch_norm:
            head = self.bn(head)
        head = self.dropout(head)
        head = head.view(-1, 1, self.entity_dim)
        relation = relation.view(-1, self.entity_dim, self.entity_dim)
        relation = self.dropout(relation)
        x = torch.bmm(head, relation)
        x = x.view(-1, self.entity_dim)
        if self.do_batch_norm:
            x = self.bn(x)
        x = self.dropout(x)
        s = torch.mm(x, self.E.weight.transpose(1, 0))
        return s

    def TuckER(self, head, relation):
        if self.do_batch_norm:
            head = self.bn(head)
        ent_embed = head.size(1)
        head = self.dropout(head)
        head = head.view(-1, 1, ent_embed)

        W_mat = torch.mm(relation, self.W.view(relation.size(1), -1))
        W_mat = W_mat.view(-1, ent_embed, ent_embed)
        W_mat = self.dropout(W_mat)

        s = torch.bmm(head, W_mat)
        s = s.view(-1, ent_embed)
        s = self.bn(s)
        s = self.dropout(s)
        s = torch.mm(s, self.E.weight.transpose(1, 0))
        return s


    def init_(self):
        nn.init.xavier_normal_(self.E.weight.data)
        if self.model_name == 'Rotat3':
            nn.init.uniform_(self.R.weight.data, a=-1.0, b=1.0)
        else:
            nn.init.xavier_normal_(self.R.weight.data)