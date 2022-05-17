# -*- coding: utf-8 -*-
'''
# Created on 11 月-26-20 16:25
# @filename: train.py
# @author: tcxia
'''

import numpy as np
from collections import Counter
from loguru import logger

import torch
import torch.utils.data as tud

from data.dataset import Word2VecDataset
from model.vec_model import Word2VecModel






MAX_VOCAB_SIZE = 30000
WINDOW_SIZE = 3
NEGATIVE_NUM = 100
BATCH_SIZE = 8
EMBEDDING_DIM = 100
LEARNING_RATE = 0.01
EPOCHS = 10

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)


def file_process(path):
    with open(path, 'r', encoding='utf-8') as fr:
        text = fr.read()
    text = text.split()

    vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
    vocab['<unk>'] = len(text) - np.sum(list(vocab.values()))

    idx2word = [word for word in vocab.keys()]
    word2idx = {word: i for i, word in enumerate(idx2word)}

    word_counts = np.array([count for count in vocab.values()],
                           dtype=np.float32)
    word_freqs = word_counts / np.sum(word_counts)
    word_freqs = word_freqs**(3. / 4.)
    word_freqs = word_freqs / np.sum(word_freqs)

    return text, word2idx, idx2word, word_freqs, word_counts


def Trainer(epoch, model, train_dataloader, dev_dataloader, optimizer, device):
    model.train()

    for i, (input_labels, pos_labels,
            neg_labels) in enumerate(train_dataloader):

        input_labels = input_labels.long().to(device)
        pos_labels = pos_labels.long().to(device)
        neg_labels = neg_labels.long().to(device)

        optimizer.zero_grad()

        loss = model(input_labels, pos_labels, neg_labels).mean()

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            logger.info(
                "Epoch: {} | Iteration: {} | Train-Loss: {:.3f}".format(
                    epoch, i, loss.item()))

        if i % 1000 == 0 and i > 0 :
            val_loss = Evaler(model, dev_dataloader, device)
            logger.info("Epoch: {} | Iteration: {} | Val-Loss: {:.3f}".format(
                epoch, i, val_loss))


def Evaler(model, dev_dataloader, device):
    model.eval()
    total_loss = 0.
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dev_dataloader):

        input_labels = input_labels.long().to(device)
        pos_labels = pos_labels.long().to(device)
        neg_labels = neg_labels.long().to(device)

        with torch.no_grad():
            loss = model(input_labels, pos_labels, neg_labels).mean()
        total_loss += loss.item()
    model.train()
    return total_loss


if __name__ == "__main__":

    train_path = '/data/nlp_dataset/text8/text8.train.txt'
    text_train, word2idx_train, idx2word_train, word_freqs_train, word_counts_train = file_process(
        train_path)

    vocab_size_train = len(idx2word_train)
    train_set = Word2VecDataset(text_train, vocab_size_train, word2idx_train,
                                idx2word_train, word_freqs_train,
                                word_counts_train, WINDOW_SIZE, NEGATIVE_NUM)
    train_dataloader = tud.DataLoader(train_set,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      num_workers=0)

    dev_path = '/data/nlp_dataset/text8/text8.dev.txt'
    text_dev, word2idx_dev, idx2word_dev, word_freqs_dev, word_counts_dev = file_process(
        dev_path)

    vocab_size_dev = len(idx2word_dev)
    dev_set = Word2VecDataset(text_dev, vocab_size_dev, word2idx_dev,
                              idx2word_dev, word_freqs_dev, word_counts_dev,
                              WINDOW_SIZE, NEGATIVE_NUM)
    dev_dataloader = tud.DataLoader(dev_set,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=0)

    model = Word2VecModel(vocab_size_train, EMBEDDING_DIM)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        Trainer(epoch, model, train_dataloader, dev_dataloader, optimizer, device)
