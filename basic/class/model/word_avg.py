# -*- coding: utf-8 -*-
'''
# Created on 11 月-27-20 14:41
# @filename: word_avg.py
# @author: tcxia
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class WordAvgModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout_rate, output_dim) -> None:
        super(WordAvgModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(self.embed_dim, self.output_dim)

    def forward(self, text):

        # text: [seq_len, batch_size]
        embed = self.embed(text) # [seq_len, batch_size, embed_dim]
        embed = self.dropout(embed)
        embed = embed.permute(1, 0 ,2) # [batch_size, seq_len, embed_dim]

        #  (minibatch, in_channels, iH, iW)
        # 将seq_len中间层压扁成1
        pooled = F.avg_pool2d(embed, (embed.shape[1], 1)).squeeze(1) # [batch_size, embed_dim]

        return self.fc(pooled) #[batch_size, output_dim]

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout_rate, hidden_size, output_dim, bidirectional=True) -> None:
        super(RNNModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size, num_layers=1, bidirectional=bidirectional, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size * 2, self.output_dim)


    def forward(self, text, hidden):

        # text: [seq_len,  batch_size]
        embedded = self.embed(text) # [seq_len, batch_size, ]
        embedded = self.dropout(embedded)

        output, (hidden, cell) = self.lstm(embedded)
        # hidden: [num_layers * num_directions, batch, hidden_size]
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.fc(hidden.squeeze(0))


class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout_rate, n_filter, filter_size, output_dim) -> None:
        super(CNNModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        self.dropout = nn.Dropout(dropout_rate)

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_filter, kernel_size=(filter_size, self.embed_dim))

        self.fc = nn.Linear(filter_size * n_filter, self.output_dim)


    def forward(self, text):
        # [seq_len, batch_size]
        text = text.permute(1, 0)
        embeded = self.embed(text)  #[batch_size, seq_len, embed_dim]
        embeded = self.dropout(embeded)

        embeded = embeded.unsqueeze(1)  #[batch_size, 1, seq_len, embed_dim]

        conved = F.relu(self.conv(embeded)).squeeze(3) #[batch_size, K(n_filters) * 1, seq_len - filter_size, 1]

        pooled = F.max_pool1d(conved, conved.shape[2]).squeeze(2) # [batch_size, n_filters]

        output = self.dropout(torch.cat(pooled, dim=1)) # [batch_size, n_filters * len(filter_size)]

        return self.fc(output)
