#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :dataset.py
@时间        :2020/11/24 11:08:35
@作者        :tcxia
'''
import numpy as np
from collections import Counter


import torch
from torch import neg, neg_
import torch.utils.data as tud

C = 2 # context window size
K = 15 # number of negative samples, K is approximate to C*2*5 for middle size corpus, that is to pick 5 negative samples for each context word selected

class Word2VecDataset(tud.Dataset):
    def __init__(self, text, vocab_size, word2idx, idx2word, word_freqs, word_counts, WINDOW_SIZE=3, NEGATIVE_NUM=100) -> None:
        super().__init__()

        self.text_encoded = [word2idx.get(t, vocab_size - 1) for t in text]
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

        self.window_size = WINDOW_SIZE
        self.neg_num = NEGATIVE_NUM

    def __getitem__(self, index: int):
        '''
        return:
            - center word index
            - C indices of positive words
            - K indices of negative words
        '''
        center_word = self.text_encoded[index]
        pos_indices = list(range(index - self.window_size, index)) + list(range(index + 1, self.window_size + index + 1))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        # multinomial: 对self.word_freqs每一行做self.neg_num * pos_words.shape[0]次取值，输出的张量是每一次取值对应的下标
        # replacement=True: 有放回取出
        neg_words = torch.multinomial(self.word_freqs, self.neg_num * pos_words.shape[0], replacement=True)

        return center_word, pos_words, neg_words


    def __len__(self) -> int:
        return len(self.text_encoded)




if __name__ == "__main__":

    MAX_VOCAB_SIZE = 30000
    WINDOW_SIZE = 3
    NEGATIVE_NUM = 100

    filepath = '/data/nlp_dataset/text8/text8.train.txt'
    with open(filepath, 'r', encoding='utf-8') as fin:
        cont = fin.read()
    cont = cont.split()
    vocab = dict(Counter(cont).most_common(MAX_VOCAB_SIZE - 1))
    # print(len(vocab))
    vocab['<unk>'] = len(cont) - np.sum(list(vocab.values()))
    idx2word = [word for word in vocab.keys()]
    word2idx = {word:i for i, word in enumerate(idx2word)}
    # print(len(idx2word))

    word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
    word_freqs = word_counts / np.sum(word_counts)
    word_freqs = word_freqs ** (3. / 4.)
    word_freqs = word_freqs / np.sum(word_freqs) # negative sampling


    dataset = Word2VecDataset(cont, MAX_VOCAB_SIZE, word2idx, idx2word, word_freqs, word_counts, WINDOW_SIZE, NEGATIVE_NUM)
    cw, pw, nw = next(iter(dataset))
    print(cw)
    print(pw)
    print(nw)
