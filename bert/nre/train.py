# -*- coding: utf-8 -*-
'''
# Created on 11 月-25-20 16:04
# train.py
# @author: tcxia
'''

import os
import time
import json
import numpy as np
from loguru import logger
from numpy.lib.arraypad import pad

import torch
from torch.cuda import check_error
import torch.utils.data as tud

from transformers import AutoModel, AutoTokenizer

from data.dataset import NREDataset
from model.nre_model import NRERelationExtra
from model.bert_model import BertConfig, BertModel, BertLayerNorm


def read_vocab(file):
    with open(file, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    word2idx = {}
    for index, line in enumerate(lines):
        word2idx[line.strip('\n')] = index
    return len(word2idx)

def read_ret(ret_path):
    pred2id = {}
    id2pred = {}
    pred2id['NA'] = 0
    id2pred[0] = 'NA'
    with open(ret_path, 'r', encoding='utf-8') as fr:
        for r in fr:
            ret = json.loads(r)
            if ret['predicate'] not in pred2id:
                id2pred[len(pred2id)] = ret['predicate']
                pred2id[ret['predicate']] = len(pred2id)
    return pred2id, id2pred


def collate_fn(batch):

    def padding(inputs, max_length=None, padding=0):
        if max_length is None:
            max_length = max([len(i) for i in inputs])

        pad_width = [(0, 0) for _ in np.shape(inputs[0])]
        outputs = []
        for i in inputs:
            i = i[:max_length]
            pad_width[0] = (0, max_length - len(i))
            i = np.pad(i, pad_width, 'constant', constant_values=padding)
            outputs.append(i)
        return np.array(outputs)


    token_ids = [data['token_ids'] for data in batch]
    token_type_ids = [data['token_type_ids'] for data in batch]
    sub_ids = [data['sub_ids'] for data in batch]
    sub_labels = [data['sub_labels'] for data in batch]
    obj_labels = [data['obj_labels'] for data in batch]

    maxlen = max([len(t) for t in token_ids])

    token_ids_padded = padding(token_ids, maxlen)
    token_type_ids_padded = padding(token_type_ids, maxlen)
    sub_labels_padded = padding(sub_labels, maxlen)
    obj_labels_padded = padding(obj_labels, maxlen)
    sub_ids = np.array(sub_ids)

    return torch.tensor(token_ids_padded, dtype=torch.long), torch.tensor(token_type_ids_padded, dtype=torch.float32), \
            torch.tensor(sub_labels_padded, dtype=torch.long), torch.tensor(obj_labels_padded,dtype=torch.long), \
            torch.tensor(sub_ids, dtype=torch.long)


def load_pretrain_param(model, pretrain_model_path):
    checkpoint = torch.load(pretrain_model_path)
    checkpoint = {k:v for k, v in checkpoint.items() if k[:4] == 'bert' and 'pooler' not in k}
    model.load_state_dict(checkpoint, strict=False)
    torch.cuda.empty_cache()
    logger.info("Pretrain-Model is loadded!".format(pretrain_path))

def Trainer(epoch, model, optimiter, scheduler, learning_rate, train_dataloader, device):

    model.train()
    total_loss = 0
    start_time = time.time()

    for i, (token_ids, token_type_ids, sub_labels, obj_labels, sub_ids) in enumerate(train_dataloader):

        token_ids = token_ids.to(device)
        sub_ids = sub_ids.to(device)
        sub_labels = sub_labels.to(device)
        obj_labels = obj_labels.to(device)

        preds, loss = model(token_ids,
                            sub_ids,
                            sub_labels=sub_labels,
                            obj_labels=obj_labels,
                            device=device)

        optimiter.zero_grad()
        loss.backward()

        # gradiate clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimiter.step()
        end_time = time.time()
        if i % 100 == 0:
            logger.info("Step: {} | Loss: {:.3f} | Time: {:.3f}".format(i, loss.item(), end_time - start_time))

        # scheduler.step()
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        total_loss += loss.item()
    logger.info("Epoch: {} | Total Loss: {:.3f}".format(epoch, total_loss))


if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    pretrain_path = '/data/nlp_dataset/pre_train_models/chinese-bert-wwm-ext'
    pretrain_model_path = os.path.join(pretrain_path, 'pytorch_model.bin')
    tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
    vocab_size = read_vocab(os.path.join(pretrain_path, 'vocab.txt'))

    ret_path = '/data/nlp_dataset/NRE/extract/all_50_schemas'
    pred2id, id2pred = read_ret(ret_path)

    train_path = '/data/nlp_dataset/NRE/extract/train_data.json'
    train_set = NREDataset(train_path, pred2id, tokenizer)
    train_dataloader = tud.DataLoader(train_set, batch_size=1, num_workers=0, shuffle=True, pin_memory=True, collate_fn=collate_fn)


    model = NRERelationExtra(len(pred2id), vocab_size, BertConfig, BertModel, BertLayerNorm)
    load_pretrain_param(model, pretrain_model_path)
    model = model.to(device)

    learning_rate = 1e-5
    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    for epoch in range(10):
        Trainer(epoch, model, optimizer, scheduler, learning_rate, train_dataloader, device)
