# -*- coding: utf-8 -*-
'''
# Created on 2021/01/08 16:31:43
# @filename: train.py
# @author: tcxia
'''

# -*- coding: utf-8 -*-
'''
# Created on 2021/01/11 14:28:31
# @filename: train.py
# @author: tcxia
'''




from tqdm import tqdm

import torch
import torch.utils.data as tud
import torch.nn as nn

from data.datasets import SentenceREDataset
from models.model import SentenceRE
from utils.util import get_idx2tag, save_checkpoint

from sklearn import metrics

embed_dim = 768
checkpoint_file = 'checkpoints/checkpoint.json'

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
def Trainer(model, optimizer, criterion, train_loader, dev_loader, device, epoches):
    checkpoint_dict = {}
    best_f1 = 0.
    for epoch in range(epoches):
        model.train()

        for i, batch in enumerate(tqdm(train_loader, desc='Training')):
            token_ids = batch['token_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            e1_mask = batch['e1_mask'].to(device)
            e2_mask = batch['e2_mask'].to(device)
            tag_ids = batch['tag_id'].to(device)

            optimizer.zero_grad()

            logits = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
            loss = criterion(logits, tag_ids)
            loss.backward()

            optimizer.step()

            if (i+1) % 1000 == 0:
                print('Epoch: {}/{} | Iteration: {} | Loss: {:.4f}'.format(epoch, epoches, i + 1, loss.item()))

        if epoch % 2 == 0:
            f1, precision, recall = Evaler(model, dev_loader, device)
            print("Evaluation: F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(f1, precision, recall))

            if checkpoint_dict.get('epoch_f1'):
                checkpoint_dict['epoch_f1'][epoch] = f1
            else:
                checkpoint_dict['epoch_f1'] = {epoch:f1}

            if f1 > best_f1:
                best_f1 = f1
                checkpoint_dict['best_f1'] = f1
                checkpoint_dict['best_epoch'] = epoch
                torch.save(model.state_dict(), 'checkpoints/model.pth')
            save_checkpoint(checkpoint_dict, checkpoint_file)


def Evaler(model, data_loader, device):
    model.eval()
    tag_true = []
    tag_pred = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc='Deving')):
            token_ids = batch['token_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            e1_mask = batch['e1_mask'].to(device)
            e2_mask = batch['e2_mask'].to(device)
            tag_ids = batch['tag_id'].to(device)

            logits = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
            pred_tag_ids = logits.argmax(1)
            tag_true.extend(tag_ids.tolist())
            tag_pred.extend(pred_tag_ids.tolist())

    f1 = metrics.f1_score(tag_true, tag_pred, average='weighted')
    precision = metrics.precision_score(tag_true, tag_pred, average='weighted')
    recall = metrics.recall_score(tag_true, tag_pred, average='weighted')
    return f1, precision, recall

if __name__ == "__main__":

    pretrained_model_path = '/data/nlp_dataset/pre_train_models/chinese-bert-wwm-ext'
    train_file = '/data/nlp_dataset/relate/train.data'
    tag_file = '/data/nlp_dataset/relate/relation.txt'
    train_set = SentenceREDataset(train_file, tag_file, pretrained_model_path)
    train_loader = tud.DataLoader(train_set, batch_size=4, shuffle=True)

    dev_file = '/data/nlp_dataset/relate/dev.data'
    dev_set = SentenceREDataset(dev_file, tag_file, pretrained_model_path)
    dev_loader = tud.DataLoader(dev_set, batch_size=4, shuffle=False)


    idx2tag = get_idx2tag(tag_file)
    tag_size = len(idx2tag)
    model = SentenceRE(pretrained_model_path, embed_dim, tag_size, 0.5)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss().to(device)


    Trainer(model, optimizer, criterion, train_loader, dev_loader, device, 20)
