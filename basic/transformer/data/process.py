# -*- coding: utf-8 -*-
'''
# Created on 12月-07-20 10:53
# @filename: process.py
# @author: tcxia
'''

import os
import pandas as pd
import numpy as np


import torch
from torch.autograd import Variable
from torchtext import data

from data.Tokenize import tokenize


def read_data(source_path, target_path):
    with open(source_path, 'r') as fs:
        src_data = fs.read().strip().split('\n')

    with open(target_path, 'r') as ft:
        trg_data = ft.read().strip().split('\n')

    return src_data, trg_data


def create_field():
    # python -m spacy download fr
    # 保证环境已经安装了对应的库文件
    spacy_lang = ['en', 'fr']

    t_src = tokenize(spacy_lang[0])
    t_trg = tokenize(spacy_lang[1])

    trg_field  = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    src_field = data.Field(lower=True, tokenize=t_src.tokenizer)

    return (src_field, trg_field)


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


global max_src_in_batch, max_trg_in_batch

def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_trg_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_trg_in_batch = 0

    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_trg_in_batch = max(max_trg_in_batch, len(new.trg) + 2)

    src_elements = count * max_src_in_batch
    trg_elements = count * max_trg_in_batch

    return max(src_elements, trg_elements)

def create_dataset(src_data, trg_data, src_field, trg_field, max_strlen, device):
    raw_data = {'src': [line for line in src_data], 'trg': [line for line in trg_data]}

    df = pd.DataFrame(raw_data, columns=['src', 'trg'])

    # 创建mask
    mask = (df['src'].str.count(' ') < max_strlen) & (df['trg'].str.count(' ') < max_strlen)
    df = df.loc[mask]

    df.to_csv('temp.csv', index=False)

    data_field = [('src', src_field), ('trg', trg_field)]
    train_data = data.TabularDataset('./temp.csv', format='csv', fields=data_field)

    train_iter = MyIterator(train_data,
                            batch_size=8,
                            device=device,
                            repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn,
                            train=True,
                            shuffle=True)


    os.remove('temp.csv')

    src_field.build_vocab(train_data)
    trg_field.build_vocab(train_data)

    src_pad = src_field.vocab.stoi['<pad>']
    trg_pad = trg_field.vocab.stoi['<pad>']

    train_len = get_len(train_iter)
    print(train_len)

    train_len_iter = len(train_iter.dataset)
    print(train_len_iter)

    return train_iter, src_pad, trg_pad, train_len

def get_len(train_data):
    for i, b in enumerate(train_data):
        pass
    return i

def nopeak_mask(size, device):
    # 返回函数的上三角矩阵
    # k 表示对角线的起始位置 默认为0
    # k = 1 表示对角线的位置上移一个对角线
    # k = -1 表示对角线的位置下移一个对角线
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)

    np_mask = np_mask.to(device)

    return np_mask



def create_mask(src, trg, src_pad, trg_pad, device):

    # 
    src_mask = (src != src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1)
        np_mask = nopeak_mask(size, device)

        trg_mask = trg_mask.to(device)

        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None

    return src_mask, trg_mask
