# -*- coding: utf-8 -*-
'''
# Created on 2021/04/13 17:47:01
# @filename: train.py
# @author: tcxia
'''


from data.dataset import QADataset
from model.xlnet import XLNetQA
from model.bert import BertForMultipleChoice
from util.optimation import BertAdam

from transformers import XLNetTokenizer, BertTokenizer
from transformers import AdamW

import torch
import torch.utils.data as tud
import torch.nn as nn
import torch.nn.functional as F


def pred_correct(output_prob, targets):
    _, output_label = output_prob.max(dim=1)
    correct = (output_label == targets).sum()
    return correct.item()


def Train(model, train_loader, optimizer, epoches, device):

    for epoch in range(epoches):
        model.train()

        train_loss = 0.
        correct_num = 0
        for i, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks, segment_ids, labels = batch

            optimizer.zero_grad()

            loss = model(input_ids, segment_ids, input_masks, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            train_loss += loss.item()

        epoch_loss = train_loss / len(train_loader)
        print("Epoch: {} | Loss: {} ".format(epoch, epoch_loss))


if __name__ == '__main__':
    max_seq_len = 128
    batch_size = 4
    epoches = 10
    gradient_acc_step = 1

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    train_file = '/data/nlp_dataset/qa-public/train.json'

    # pretrained_path = '/data/nlp_dataset/pre_train_models/chinese-xlnet-base'
    pretrained_path = '/data/nlp_dataset/pre_train_models/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrained_path, do_lower_case=True)

    train_set = QADataset(train_file, tokenizer, max_seq_len)
    train_loader = tud.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # model = XLNetQA(pretrained_path)

    bert_name = 'bert-base-chinese'
    model = BertForMultipleChoice.from_pretrained(
        bert_name,
        num_choices=4,
        cache_dir='/data/nlp_dataset/pre_train_models')
    model.to(device)
    # print(model)

    param_optimizer = list(model.named_parameters())
    # print(param_optimizer)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    t_total = int(len(train_set) / batch_size / gradient_acc_step * epoches)
    optimizer = BertAdam(optimizer_grouped_parameters, lr=5e-5, warmup=0.1, t_total=t_total)
    # optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



    Train(model, train_loader, optimizer, epoches, device)
