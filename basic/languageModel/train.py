# -*- coding: utf-8 -*-
'''
# Created on 11 月-26-20 17:19
# @filename: train.py
# @author: tcxia
'''

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad
import torchtext

from loguru import logger

from data.dataset import LMDataset
from model.rnn_model import RNNModel


vocab_size = 30000
embedding_size = 100
dropout_rate = 0.2
hidden_size = 2
batch_size = 4
learning_rate = 1e-3
epochs = 10
clip_rate = 5.0


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
loss_fn = nn.CrossEntropyLoss()


def repackage_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(v) for v in hidden)

def Trainer(epoch, model, train_dataloader, optimizer, learning_rate=1e-4, device='cpu'):
    model.train()

    it = iter(train_dataloader)

    hidden = model.init_hidden(batch_size)
    for i, batch in enumerate(it):
        text, target = batch.text, batch.target

        text = text.to(device)
        target = target.to(device)

        hidden = repackage_hidden(hidden)
        model.zero_grad()

        output = model(text, hidden) # [batch_size, seq_len, vocab_size]

        optimizer.zero_grad()

        loss = loss_fn(output.view(-1, vocab_size), target.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        if i % 100 == 0:
            logger.info("Epoch: {} | Iteration: {}| Loss: {}".format(epoch, i, loss.item()))



if __name__ == "__main__":
    root_path = '/data/nlp_dataset/text8'
    text_process = torchtext.data.Field(lower=True)

    train_iter, val_iter, test_iter = LMDataset(root_path, text_process, vocab_size, batch_size, device=device)

    model = RNNModel(vocab_size, embedding_size, dropout_rate, hidden_size)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    for epoch in range(epochs):
        Trainer(epoch, model, train_iter, optimizer, device=device)
