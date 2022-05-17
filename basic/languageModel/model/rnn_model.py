# -*- coding: utf-8 -*-
'''
# Created on 11 月-26-20 20:54
# @filename: rnn_model.py
# @author: tcxia
'''


import torch
import torch.nn as nn



class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout_rate, hidden_size) -> None:
        super(RNNModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.LSTM(self.embed_dim, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inputs, hidden):
        # inputs: [seq_len, batch_size]
        # print(inputs.shape)
        embed = self.embed(inputs) # [seq_len, batch_size, emb_dim]
        embed = self.dropout(embed)

        # self.rnn: input: [seq_len, batch, input_size]
        #           hidden: [num_layers * num_directions, batch, hidden_size]
        #           
        #           output: (seq_len, batch, num_directions * hidden_size)
        output, (hidden, cell) = self.rnn(embed, hidden)

        output = self.dropout(output) # [seq_len, batch_size, 1 * 2]
        output = output.permute(1, 0, 2)

        return self.linear(output) 


    def init_hidden(self, bsz, requires_grad=True):
        weights = next(self.parameters())
        return (weights.new_zeros((1, bsz, self.hidden_size), requires_grad=requires_grad),weights.new_zeros((1, bsz, self.hidden_size), requires_grad=requires_grad))



if __name__ == "__main__":
    VOCAB_SIZE = 30000
    EMBEDDING_DIM = 100
    DROPOUT_RATE = 0.2
    HIDDEN_SIZE = 2
    model = RNNModel(VOCAB_SIZE, EMBEDDING_DIM, DROPOUT_RATE, HIDDEN_SIZE)
    print(model)
