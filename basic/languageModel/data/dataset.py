# -*- coding: utf-8 -*-
'''
# Created on 11 月-26-20 18:20
# @filename: dataset.py
# @author: tcxia
'''

import os
import torchtext
from torchtext.vocab import Vectors

import torch

MAX_VOCAB_SIZE = 50000
BATCH_SIZE = 4

root_path = '/data/nlp_dataset/text8'
# Field 决定你数据被如何处理s
TEXT = torchtext.data.Field(lower=True)

#LanguageModelingDataset: 处理成语言模型数据集
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path=root_path,
    train='text8.train.txt',
    validation='text8.dev.txt',
    test='text8.test.txt',
    text_field=TEXT)

TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)

VOCAB_SIZE = len(TEXT.vocab)

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test),
    batch_size=BATCH_SIZE,
    bptt_len=32,
    repeat=False,
    shuffle=True,
    device='cpu')

# [torchtext.data.batch.Batch of size 4]
#         [.text]:[torch.LongTensor of size 32x4]
#         [.target]:[torch.LongTensor of size 32x4]

# it = next(iter(train_iter))
# print(" ".join([TEXT.vocab.itos[i] for i in it.text[:, 1].data]))
# print(" ".join([TEXT.vocab.itos[i] for i in it.target[:, 1].data]))

# print(len(TEXT.vocab)) # 50002



def LMDataset(root_path, text_process, vocab_size, batch_size, bptt_len=32, device='cpu'):
    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
        path=root_path,
        train='text8.train.txt',
        validation='text8.dev.txt',
        test='text8.test.txt',
        text_field=text_process)

    text_process.build_vocab(train, max_size=vocab_size)

    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        (train, val, test),
        batch_size=batch_size,
        bptt_len=bptt_len,
        repeat=False,
        shuffle=True,
        device=device)

    return train_iter, val_iter, test_iter

    
