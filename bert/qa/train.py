# -*- coding: utf-8 -*-
'''
# Created on 11 月-19-20 15:35
# train.py
# @author: tcxia
'''

import os
from tqdm import tqdm
import numpy as np
from loguru import logger

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel

from data.dataset import WebQADataset
from model.model import NetQA


batch_size = 2
lr = 1e-5
weight_decay= 1e-5
epochs = 10
negweight = 0.01

model_name = '/data/nlp_dataset/pre_train_models/chinese-bert-wwm-ext'
tokenizer = AutoTokenizer.from_pretrained(model_name)

save_model = './checkpoint'


def Train(epoch, model, train_dataloader, val_dataloader, optimizer, device):
    model.train()
    crf_preds, crf_losses, IsQA_losses, y_crfs, IsQA_preds, y_IsQAs = [], [], [], [], [], []

    best_acc = 0
    for i, batch in enumerate(tqdm(train_dataloader)):
        _, token_ids_l, token_type_ids_l, answer_offset_l, answer_seq_label_l, IsQA_l = batch

        optimizer.zero_grad()

        IsQA_pred, crf_pred, IsQA_loss, crf_loss, y_crf, y_IsQA = model(
            token_ids_l, token_type_ids_l, answer_offset_l, answer_seq_label_l,
            IsQA_l)

        crf_preds.append(crf_pred)
        y_crfs.append(y_crf)

        IsQA_preds.append(IsQA_pred)
        y_IsQAs.append(y_IsQA)

        crf_losses.append(crf_loss.to("cpu").item())
        IsQA_losses.append(IsQA_loss.to("cpu").item())

        loss = IsQA_loss + crf_loss

        loss.backward()

        optimizer.step()

        if i % 100 == 0 and i > 0:
            accCRF = result_matric(crf_preds, y_crfs)
            accIsQA = result_matric(IsQA_preds, y_IsQAs)
            logger.info(
                "Epoch: {} | Step: {} | IsQA-Loss: {:.3f} | CRF-Loss: {:.3f} | CRF-Result: accCRF={:.3f} | IsQA-Result: accQA={:.3f}"
                .format(epoch, i, np.mean(IsQA_losses), np.mean(crf_losses), accCRF, accIsQA))

        if i % 1000 == 0 and i != 0:
            accIsQA, accCRF = Eval(model, val_dataloader, device)
            if accIsQA * accCRF > best_acc:
                best_acc = accIsQA * accCRF
                logger.info(best_acc)
                if i > 0:
                    save_model_path = os.path.join(
                        save_model, 'epoch_{}.pth'.format(epoch))
                    logger.info("save model to {}".format(save_model_path))
                    torch.save(
                        model,
                       save_model_path)
            model.train()


def Eval(model, dataloader, device):
    model.eval()

    crf_preds, crf_losses, IsQA_losses, y_crfs, IsQA_preds, y_IsQAs = [], [], [], [], [], []
    final_preds = []

    for i, batch in enumerate(dataloader):
        _, token_ids_l, token_type_ids_l, answer_offset_l, answer_seq_label_l, IsQA_l = batch
        IsQA_pred, crf_pred, IsQA_loss, crf_loss, y_crf, y_IsQA = model(
            token_ids_l, token_type_ids_l, answer_offset_l, answer_seq_label_l,
            IsQA_l)

        crf_preds.append(crf_pred)
        y_crfs.append(y_crf)

        IsQA_preds.append(IsQA_pred)
        y_IsQAs.append(y_IsQA)

        final_pred = torch.LongTensor(np.zeros(crf_pred.size())).to(device)
        final_pred[IsQA_pred.squeeze(dim=-1) == 1] = crf_pred[IsQA_pred.squeeze(dim=-1) == 1]
        final_preds.append(final_pred)

        crf_losses.append(crf_loss.to("cpu").item())
        IsQA_losses.append(IsQA_loss.to("cpu").item())

    accCRF = result_matric(crf_preds, y_crfs)
    accIsQA = result_matric(IsQA_preds, y_IsQAs)
    accFinal = result_matric(final_preds, y_crfs)

    logger.info(
        "Result: IsQA-Loss: {:.3f} | CRF-Loss: {:.3f} | CRF-Result: accCRF = {:.3f} | IsQA-Result: accIsQA = {:.3f} | Final-Result: accFinal = {:.3f}"
        .format(np.mean(IsQA_losses), np.mean(crf_losses), accCRF, accIsQA, accFinal))

    return accIsQA, accCRF

def collate_fn(batch):
    tokens_l, token_id_l, token_type_ids_l, answer_offset_l, answer_seq_label_l, IsQA_l = list(map(list, zip(*batch)))
    maxlen = np.array([len(sen) for sen in tokens_l]).max()

    for i in range(len(tokens_l)):
        token = tokens_l[i]
        token_id = token_id_l[i]
        answer_seq_label = answer_seq_label_l[i]
        token_type_id = token_type_ids_l[i]

        tokens_l[i] = token + (maxlen - len(token)) * ['[PAD]']
        token_type_ids_l[i] = token_type_id + (maxlen - len(token)) * [1]
        token_id_l[i] = token_id + (maxlen - len(token)) * tokenizer.convert_tokens_to_ids(['[PAD]'])
        answer_seq_label_l[i] = answer_seq_label + [0] * (maxlen - len(token))
    return tokens_l, token_id_l, token_type_ids_l, answer_offset_l, answer_seq_label_l, IsQA_l


def result_matric(preds, y_crfs):
    total_num = 0
    total_cur = 0
    for pred, y_crf in zip(preds, y_crfs):
        batch_size, seq_len = pred.size()
        cur = torch.sum(torch.sum(pred == y_crf, dim=1) == seq_len).to("cpu").item()
        total_cur += cur
        total_num += batch_size
    return total_cur / total_num


if __name__ == "__main__":

    train_set = WebQADataset('/data/nlp_dataset/WebQA/me_train.json')
    dev_set = WebQADataset('/data/nlp_dataset/WebQA/me_validation.ann.json')

    sample_weights = train_set.get_samples_weights(negweight)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=False,
                              sampler=sampler,
                              num_workers=0,
                              collate_fn=collate_fn)

    dev_loader = DataLoader(dataset=dev_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=collate_fn)

    # Train()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    pre_model = AutoModel.from_pretrained(model_name)
    model = NetQA(pre_model, device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        Train(epoch, model, train_loader, dev_loader, optimizer, device)