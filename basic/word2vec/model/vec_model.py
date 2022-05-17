# -*- coding: utf-8 -*-
'''
# Created on 11 月-26-20 15:28
# vec_model.py
# @author: tcxia
'''

import os
from black import out

import torch
import torch.nn as nn
import torch.nn.functional as F

class Word2VecModel(nn.Module):
    def __init__(self,vocab_size, embedding_dim) -> None:
        super(Word2VecModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        initrange = 0.5 / self.embedding_dim
        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim) #
        self.embed.weight.data.uniform_(-initrange, initrange)


    def forward(self, input_labels, pos_labels, neg_labels):
        '''
            input_labels: center words, [batch_size] which is one dimentional vector of batch size
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels: negative words, [batch_size, K]
            return: loss, [batch_size]
        '''

        batch_size = input_labels.shape[0]

        # [batch_size]
        input_embedding = self.embed(input_labels) # [batch_size, embed_dim]

        # [batch_size, window_size * 2]
        pos_embedding = self.embed(pos_labels) # [batch_size, window_size * 2, embed_dim]

        # [bacth_size, window*2*neg_num]
        neg_embedding = self.embed(neg_labels)  # [bacth_size, window*2*neg_num, embed_dim]

        input_embedding_suq = input_embedding.unsqueeze(2) # [batch_size, embed_dim, 1]
        #[batch_size, window_size * 2]
        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze()
        #[batch_size, window*2*neg_num]
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze()

        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)

        loss = log_pos + log_neg

        return -loss

    # 获取预训练权重
    def init_weights(self):
        return self.embed.weight.data.cpu().numpy()


    def save_embedding(self, outdir, idx2word):
        embeds = self.embed.weight.data.cpu().numpy();
        f1 = open(os.path.join(outdir, 'vec.tsv'), 'w')
        f2 = open(os.path.join(outdir, 'word.tsv'), 'w')
        for idx in range(len(embeds)):
            word = idx2word[idx]
            embed = '\t'.join([str(x) for x in embeds[idx]])
            f1.write(embed + '\n')
            f2.write(word + '\n')

