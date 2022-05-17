# -*- coding: utf-8 -*-
'''
# Created on 2021/02/22 14:09:30
# @filename: bilstm_att.py
# @author: tcxia
'''



import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_ATT(nn.Module):
    def __init__(self, batch, emb_size, emb_dim, hid_dim, tag_size, pos_size,
                 pos_dim, pretrained, embed_pre) -> None:
        """初始化

        Args:
            batch ([int]): [输入网络的batch size大小]
            emb_size ([int]]): [整个单词表的大小]
            emb_dim ([int]): [词向量维度]
            hid_dim ([int]): [隐藏层的维度]
            tag_size ([int]): [预测标签的大小（一般指的是关系类别数量）]
            pos_size ([int]): [位置向量的大小]
            pos_dim ([int]]): [位置向量的维度]
            pretrained ([bool]]): [是否导入预训练词向量]
            embed_pre ([numpy]): [预训练词向量矩阵]
        """
        super().__init__()
        self.batch = batch

        self.emb_size = emb_size
        self.emb_dim = emb_dim

        self.hid_dim = hid_dim
        self.tag_size = tag_size

        self.pos_size = pos_size
        self.pos_dim = pos_dim

        self.pretrained = pretrained
        if self.pretrained:
            self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(embed_pre), freeze=False)
        else:
            self.word_embeds = nn.Embedding(self.emb_size, self.emb_dim)
        
        self.pos1_embeds = nn.Embedding(self.pos_size, self.pos_dim)
        self.pos2_embeds = nn.Embedding(self.pos_size, self.pos_dim)
        self.relation_embeds = nn.Embedding(self.tag_size, self.hid_dim)

        self.lstm = nn.LSTM(input_size=self.emb_size+self.pos_dim*2, hidden_size=self.hid_dim // 2, num_layers=1, bidirectional=True)
        self.hid2tag = nn.Linear(self.hid_dim, self.tag_size)

        self.dropout_emb = nn.Dropout(p=0.5)
        self.dropout_lstm = nn.Dropout(p=0.5)
        self.dropout_att = nn.Dropout(p=0.5)

        self.hid = self._init_hidden()

        self.att_weight = nn.Parameter(torch.randn(self.batch, 1, self.hid_dim))
        self.relation_bias = nn.Parameter(torch.randn(self.batch, self.tag_size, 1))

    def _init_hidden(self):
        """初始化隐藏层参数

        Returns:
            [tensor]: 返回维度为[2, batch_size, hid_dim // 2]
        """
        return torch.randn(2, self.batch, self.hid_dim // 2)

    def _init_hidden_lstm(self):
        """初始化lstm层参数

        Returns:
            [type]: [description]
        """
        return (torch.randn(2, self.batch, self.hid_dim // 2), torch.randn(2, self.batch, self.hid_dim // 2))

    def attention(self, h):
        """attention

        Args:
            h ([tensor]): [隐藏层向量]

        Returns:
            [tensor]: [注意力权重矩阵]
        """
        M = F.tanh(h)
        a = F.softmax(torch.bmm(self.att_weight, M), 2)
        a = torch.transpose(a, 1, 2)
        return torch.bmm(h, a)

    def forward(self, sentence, pos1, pos2):
        """前向计算

        Args:
            sentence ([numpy]]): [传入的句子编码]
            pos1 ([int]): [实体1的位置索引]
            pos2 ([int]): [实体2的位置索引]

        Returns:
            [tensor]: [实体关系的概率值]
        """
        self.hidden = self._init_hidden_lstm()

        embeds = torch.cat((self.word_embeds(sentence), self.pos1_embeds(pos1), self.pos2_embeds(pos2)), 2)
        embeds = torch.transpose(embeds, 0, 1)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)

        lstm_out = self.dropout_lstm(lstm_out)
        att_out = F.tanh(self.attention(lstm_out))

        relation = torch.tensor([i for i in range(self.tag_size)], dtype=torch.long).repeat(self.batch, 1)
        relation = self.relation_embeds(relation)

        res = torch.add(torch.bmm(relation, att_out), self.relation_bias)
        res = F.softmax(res, 1)

        return res.view(self.batch, -1)
