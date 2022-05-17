# -*- coding: utf-8 -*-
'''
# Created on 11 月-27-20 14:25
# @filename: dataset.py
# @author: tcxia
'''

import os
import torch
import torchtext
from torchtext import datasets

import random

from torchtext.vocab import Vectors

SEED = 1234
MAX_VOCAB_SIZE = 25000
# torch.manual_seed(123)

TEXT = torchtext.data.Field(tokenize='spacy')
LABEL = torchtext.data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root='./imdb')

# print(len(train_data), type(train_data))

# print(next(iter(train_data)))

# vars: 返回对象Object的属性和属性值的字典对象
print(vars(train_data.examples[0]))

# 切分数据集
train_data, dev_data = train_data.split(random_state=random.seed(SEED))

cache = '/data/nlp_dataset'
vectors = Vectors(name='/data/nlp_dataset/glove.6B.100d.txt', cache=cache)

TEXT.build_vocab(train_data, max_size=25000, vectors=vectors, unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)


print(TEXT.vocab.freqs.most_common(20))