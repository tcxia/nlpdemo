# -*- coding: utf-8 -*-
'''
# Created on 12月-07-20 14:34
# @filename: train.py
# @author: tcxia
'''

import torch
import torch.nn.functional as F

from loguru import logger

from data.process import read_data, create_field, create_dataset, create_mask
from model.models import TransformModel
from model.optim_lr import CosineWithRestarts

epoches = 10
d_model = 512
n_layers = 6
heads = 8
max_strlen = 80
dropout_rate = 0.2
learning_rate = 0.0001

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def Trainer(epoch, model, train_iter, src_pad, trg_pad, optimizer, scheld, device):
    model.train()

    for i, batch in enumerate(train_iter):
        src = batch.src.transpose(0, 1)
        trg = batch.trg.transpose(0, 1)

        trg_input = trg[:, :-1]

        src_mask, trg_mask = create_mask(src, trg_input, src_pad, trg_pad, device)

        preds = model(src, trg_input, src_mask, trg_mask)
        ys = trg[:, 1:].contiguous().view(-1)
        optimizer.zero_grad()

        loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
                               ys,
                               ignore_index=trg_pad)

        loss.backward()

        optimizer.step()

        scheld.step()

    
        if (i+1) % 100 == 0:
            logger.info("Epoch: [{} /{}] | Iter: {} | Loss: {:.4f}".format(
                epoch, epoches, i + 1, loss.item()))


if __name__ == "__main__":
    en_path = '/data/nlp_dataset/translate/english.txt'
    fr_path = '/data/nlp_dataset/translate/french.txt'

    src_data, trg_data = read_data(en_path, fr_path)
    src_field, trg_field = create_field()

    train_iter, src_pad, trg_pad, train_len = create_dataset(src_data, trg_data,
                                                  src_field, trg_field,
                                                  max_strlen, device)

    logger.info("train len: {}".format(train_len))

    src_vocab = len(src_field.vocab)
    trg_vocab = len(trg_field.vocab)

    model = TransformModel(src_vocab, trg_vocab, d_model, n_layers, heads,
                           dropout_rate, max_strlen, device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 betas=(0.9, 0.98),
                                 eps=1e-9)

    scheld = CosineWithRestarts(optimizer, T_max=train_len)
    for epoch in range(epoches):
        Trainer(epoch, model, train_iter, src_pad, trg_pad, optimizer, scheld, device)