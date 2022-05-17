# -*- coding: utf-8 -*-
'''
# Created on 11 月-23-20 11:22
# train.py
# @author: tcxia
'''

import os
import numpy as np
import time
from loguru import logger

import torch
from torch import optim
from torch.utils.data import DataLoader

from data.dataset import AbstractData
from model.ab_model import AbstractModel
from model.bert_model import BertConfig, BertModel, BertLMPredictionHead

from transformers import AutoModel, AutoTokenizer

premodel_path = '/data/nlp_dataset/pre_train_models/chinese-bert-wwm-ext'
tokenizer = AutoTokenizer.from_pretrained(premodel_path)


def read_vocab(file):
    with open(file, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    word2idx = {}
    for index, line in enumerate(lines):
        word2idx[line.strip('\n')] = index
    return len(word2idx)

def collate_fn(batch):
    token_l, token_id_l, token_type_id_l = list(map(list, zip(*batch)))
    maxlen = np.array([len(cont) for cont in token_l]).max()

    for i in range(len(token_l)):
        token = token_l[i]
        token_id = token_id_l[i]
        token_type_id = token_type_id_l[i]

        token_l[i] = token + (maxlen - len(token)) * ['[PAD]']
        token_type_id_l[i] = token_type_id + (maxlen - len(token)) * [1]
        token_id_l[i] = token_id +  (maxlen - len(token)) * tokenizer.convert_tokens_to_ids(['[PAD]'])

    # print(type(token_id_l))
    token_id_l = torch.tensor(token_id_l)
    token_type_id_l = torch.tensor(token_type_id_l)
    target_id_l = token_id_l[:, 1:].contiguous()

    return token_l, token_id_l, token_type_id_l, target_id_l


def Trainer(epoch, model, train_dataloader, optimizer, device):
    model.train()
    total_loss = 0
    start_time = time.time()
    for i, batch in enumerate(train_dataloader):
        _, token_id, token_type_id, target_id = batch

        optimizer.zero_grad()

        token_id = token_id.to(device)
        token_type_id = token_type_id.to(device)
        target_id = target_id.to(device)

        predictions, loss = model(token_id, token_type_id, labels=target_id)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if i % 100 == 0:
            end_time = time.time()
            logger.info("Loss: {:.3f} | Time: {:.3f}".format(loss, end_time - start_time))
    
    logger.info("Epoch {} | Total Loss: {:.3f}".format(epoch, total_loss))


if __name__ == "__main__":
    premodel_path = '/data/nlp_dataset/pre_train_models/chinese-bert-wwm-ext'
    data_path = '/data/nlp_dataset/THUCNews'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    vocab_size = read_vocab(os.path.join(premodel_path, 'vocab.txt'))

    train_set = AbstractData(data_path, premodel_path)
    train_dataloader = DataLoader(train_set, batch_size=4, num_workers=0, shuffle=True, collate_fn=collate_fn)

    model = AbstractModel(BertConfig, BertModel, BertLMPredictionHead, vocab_size, device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)

    for epoch in range(20):
        Trainer(epoch, model, train_dataloader, optimizer, device)
