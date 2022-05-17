# -*- coding: utf-8 -*-
'''
# Created on 2021/03/30 18:42:16
# @filename: train_siamese.py
# @author: tcxia
'''

import torch
import torch.utils.data as tud
import torch.nn as nn

import os
from tqdm import tqdm
import numpy as np

from data.datasets import SiameseData
from models.siamese import Siamese
from utils.util import correct_pred

def Trainer(model, train_loader, criterion, optimizer, scheduler, start_epoch, epochs, device, save_dir):
    for epoch in range(start_epoch + 1, epochs + start_epoch + 1):
        model.train()
        epoch_loss = 0.
        epoch_acc = 0.
        for i, batch in enumerate(tqdm(train_loader, desc='[Training]')):
            q, h, labels = batch
            q = q.to(device)
            h = h.to(device)
            labels = labels.to(device)

            logits, probs = model(q, h)
            loss = criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 10.)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += correct_pred(probs, labels)
        epoch_avg_loss = epoch_loss / len(train_loader)
        epoch_avg_acc = epoch_acc / len(train_loader.dataset)
        print("Epoch: {} | Loss: {} | Acc: {}".format(epoch, epoch_avg_loss, epoch_avg_acc))

        scheduler.step(epoch_avg_acc)

        torch.save({"epoch": epoch, "model": model.state_dict()}, os.path.join(save_dir, 'epoch_' + str(epoch) + ".pth.tar"))


if __name__ == "__main__":
    checkpoint_Flag = False
    start_epoch = 0
    epochs = 100

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    train_file = "/data/nlp_dataset/oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv"
    train_set = SiameseData(train_file)
    train_loader = tud.DataLoader(train_set, batch_size=128, shuffle=True)

    # embedding_matrix = train_set.random_embedding()
    # print(embedding_matrix)
    # test_file = "/data/nlp_dataset/oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv"
    # wordcount = get_word_count(train_file, test_file)
    # embedding_matrix = random_embedding(wordcount)
    # print(embedding_matrix.shape)


    # embedding_matrix = np.load("checkpoint/siamese/embed.npy")

    # model = Siamese(embedding_matrix, device=device)

    # Siamese(word_count, word_dim)
    model = Siamese(22000, 100)
    model.to(device)

    if checkpoint_Flag:
        checkpoint_file = 'checkpoint/siamese/epoch_80.pth.tar'
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])


    criterion = nn.CrossEntropyLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.85, patience=0)

    save_dir = os.path.join('checkpoint', 'siamese')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # np.save(os.path.join(save_dir, 'embed.npy'), embedding_matrix)

    Trainer(model, train_loader, criterion, optimizer, scheduler, start_epoch, epochs, device, save_dir)
