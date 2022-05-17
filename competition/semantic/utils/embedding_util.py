# -*- coding: utf-8 -*-
'''
# Created on 2021/03/31 10:31:20
# @filename: embedding_util.py
# @author: tcxia
'''

import gensim
import numpy as np
import pandas as pd

# 载入预训练词向量
def load_embeddings(embedding_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=False)
    embedding_matrix = np.zeros((len(model.index2word) + 1, model.vector_size))
    for idx, word in enumerate(model.index2word):
        embedding_matrix[idx + 1] = model[word]
    return embedding_matrix

# 随机变量初始化
def random_embedding(word_count, vector_size=100):
    embedding_matrix = np.random.randn(word_count + 1, vector_size)
    return embedding_matrix