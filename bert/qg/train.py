# -*- coding: utf-8 -*-
'''
# Created on 11 月-18-20 14:17
# train.py
# @author: tcxia
'''


import time
import math

import torch
from torch import device, dropout
import torch.nn as nn
from torch.nn.modules.loss import SmoothL1Loss
from torch.utils.data import DataLoader

from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
from transformers import BertModel, BertTokenizer

from data.dataset import BertDataset
from model.attention import Attention
from model.decoder import Decoder
from model.seq2seq import Seq2seq
from utils.initializer import init_weights
from utils.checkpoint import save_checkpoint

from loguru import logger


# 超参数Bert
bert_hidden_size = 1024
bert_vocab_size = 28996
decoder_hidden_size = 512
decoder_input_size = 512
attention_hidden_size = 512

batch_size = 2
num_layers = 1
dropout = 0.5
encoder_trained = True
weight_decay = 0.001
lr = 0.05
momentum = 0.9
epochs = 10
clip = 1

# 指定cuda设备以及使用的卡号
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model_path = '/data/transModel/bert-large-cased/'
model_name = 'bert-large-cased'
# 导入tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)

# 定义损失函数
pw_criterion = nn.CrossEntropyLoss(ignore_index=0)


def BertTrain(model, device, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    # start_time = time.time()
    for i, (input_, output_) in enumerate(dataloader):
        input_data, input_len = input_
        output_data, output_len = output_
        # print(input_data.shape)

        optimizer.zero_grad()

        prediction = model([x.to(device) for x in input_data], output_data.to(device))
        trg_sent_len = prediction.size(1)

        prediction = prediction[:, 1:].contiguous().view(-1, prediction.shape[-1])
        output_data = output_data[:, 1:].contiguous().view(-1)

        with torch.no_grad():
            pw_loss = pw_criterion(prediction, output_data.to(device))

        loss = criterion(prediction, output_data.to(device))

        loss = loss.view(-1, trg_sent_len - 1)
        loss = loss.sum(1)
        loss = loss.mean(0)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), clip)
        optimizer.step()

        if i % int(len(dataloader) * 0.1) == int(len(dataloader) * 0.1) - 1:
            logger.info('Batch {} | Sentence Loss {} | Word Loss {}'.format(i, loss.item(), pw_loss.item()))

        epoch_loss += pw_loss.item()

    return epoch_loss / len(dataloader)


def BertEval(model, device, dataloader, criterion):
    model.eval()

    epoch_loss = 0
    epoch_bleu = 0

    with torch.no_grad():
        for i, (input_, output_) in enumerate(dataloader):
            input_data, input_len = input_
            output_data, output_len = output_

            prediction = model([x.to(device) for x in input_data], output_data.to(device))

            sample_t = tokenizer.convert_ids_to_tokens(output_data[0].tolist())
            sample_p = tokenizer.convert_ids_to_tokens(prediction[0].max(1)[1].tolist())

            idx1 = sample_t.index('[PAD]') if '[PAD]' in sample_t else len(sample_t)
            idx2 = sample_p.index('[SEP]') if '[SEP]' in sample_p else len(sample_p)

            bleu = bleu_score(prediction, output_data.to(device))

            trg_sent_len = prediction.size(1)

            prediction = prediction[:, 1:].contiguous().view(-1, prediction.shape[-1])
            output_data = output_data[:, 1:].contiguous().view(-1)

            pw_loss = pw_criterion(prediction, output_data.to(device))

            loss = criterion(prediction, output_data.to(device))
            loss = loss.view(-1, trg_sent_len - 1)
            loss = loss.sum(1)
            loss = loss.mean(0)
            if i % int(len(dataloader) * 0.1) == int(len(dataloader) * 0.1) - 1:
                logger.info(
                    "Bacth {} | Sentence Loss: {} | Word Loss: {} | Bleu Score: {} | Target: {} | Prediction: {}"
                    .format(i, loss.item(), pw_loss.item(), bleu, sample_t[1:idx1 - 1], sample_p[1:idx2 - 1]))

            epoch_loss += pw_loss.item()
            epoch_bleu += bleu

    return epoch_loss / len(dataloader), epoch_bleu / len(dataloader)



def bleu_score(prediction, ground_truth):
    prediction = prediction.max(2)[1]
    acc_bleu = 0

    for x, y in zip(ground_truth, prediction):
        x = tokenizer.convert_ids_to_tokens(x.tolist())
        y = tokenizer.convert_ids_to_tokens(y.tolist())

        idx1 = x.index('[PAD]') if '[PAD]' in x else len(x)
        idx2 = y.index('[SEP]') if '[SEP]' in y else len(y)

        acc_bleu += bleu([x[1:idx1 - 1]], y[1:idx2 - 1], smoothing_function=SmoothingFunction().method4)
    return acc_bleu / prediction.size(0)


if __name__ == "__main__":

    train_set = BertDataset('/data/squad/train')
    dev_set = BertDataset('/data/squad/dev')

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True if torch.cuda.is_available() else False)
    valid_loader = DataLoader(dev_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True if torch.cuda.is_available() else False)

    attention = Attention(bert_hidden_size, decoder_hidden_size,
                          attention_hidden_size)
    decoder = Decoder(bert_vocab_size, decoder_input_size, bert_hidden_size,
                      decoder_hidden_size, num_layers, dropout, attention,
                      device)

    encoder = BertModel.from_pretrained(model_path)


    model = Seq2seq(encoder, decoder, device, encoder_trained)

    optimizer = torch.optim.SGD(decoder.parameters(), weight_decay=weight_decay, lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    last_epoch = 0
    valid_loss, train_loss = [], []
    model.apply(init_weights)

    model.to(device)

    for epoch in range(last_epoch, epochs):
        start_time = time.time()

        train_loss_epoch = BertTrain(model, device, train_loader, optimizer, criterion, clip)
        valid_loss_epoch, bleu_score_epoch = BertEval(model, device, valid_loader, criterion)

        train_loss.append(train_loss_epoch)
        valid_loss.append(valid_loss_epoch)

        save_checkpoint('./checkpoint/model_epoch{}.pth'.format(epoch), epoch, model, optimizer, valid_loss, train_loss)

        end_time = time.time()

        logger.info("Epoch: {} completed | Time: {}".format(epoch, end_time - start_time))
        logger.info("Train Loss: {} | Train PPL: {}".format(train_loss_epoch, math.exp(train_loss_epoch)))
        logger.info("Val Loss: {} | Val PPL: {}".format(valid_loss_epoch, math.exp(valid_loss_epoch)))
