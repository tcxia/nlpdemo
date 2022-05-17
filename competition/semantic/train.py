# -*- coding: utf-8 -*-
'''
# Created on 2021/01/13 15:26:42
# @filename: train.py
# @author: tcxia
'''

from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.utils.data as tud

from transformers import BertTokenizer, XLNetTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from data.datasets import TextDataset
from models.bert import TextMatch_Bert
from models.xlnet import TextMatch_XLNet
from utils.util import correct_pred


def Trainer(model, optimizer, scheduler, train_loader, epoches, device, save_dir, start_epoch=0):
    # best_score = 0.0
    for epoch in range(start_epoch + 1, epoches + start_epoch):
        model.train()

        train_loss = 0.
        correct_num = 0
        for i, batch in enumerate(tqdm(train_loader, desc="[Training]")):
            batch_seqs, batch_seq_masks, batch_segments, batch_labels = batch
            seqs = batch_seqs.to(device)
            seq_masks = batch_seq_masks.to(device)
            segments = batch_segments.to(device)
            labels = batch_labels.to(device)

            optimizer.zero_grad()

            loss, logits, prob = model(seqs, seq_masks, segments, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            train_loss += loss.item()
            correct_num += correct_pred(prob, labels)

        epoch_loss = train_loss / len(train_loader)
        epoch_acc = correct_num / len(train_loader.dataset)
        print("Epoch: {} | Loss: {} | Acc: {}".format(epoch, epoch_loss, epoch_acc))

        scheduler.step(epoch_acc)

        torch.save({"epoch": epoch, "model": model.state_dict(), "losses": epoch_loss, "acc": epoch_acc}, os.path.join(save_dir, 'epoch_' + str(epoch) + '.pth.tar'))
# def Evaluation(model, dev_loader)

# scheduler.step()

if __name__ == "__main__":
    checkpoint = True
    epoches = 10


    device = torch.device('cuda:3' if torch.cuda.is_available() else "cpu")
    pretrained_path = '/data/nlp_dataset/pre_train_models/chinese-xlnet-base'
    tokenizer = XLNetTokenizer.from_pretrained(pretrained_path, do_lower_case=True)

    train_file = '/data/nlp_dataset/oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv'
    train_set = TextDataset(tokenizer, train_file)
    train_loader = tud.DataLoader(train_set, batch_size=32, shuffle=False, pin_memory=True, num_workers=0)

    # model = TextMatch_Bert(bert_pretrained_path)
    model = TextMatch_XLNet(pretrained_path)
    model.to(device)

    start_epoch = 0
    if checkpoint:
        checkpoint_file = '/data/nlp_dataset/pre_train_models/xlnet/best.pth.tar'
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
    print(start_epoch)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':0.
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

    # warm_steps = int(t_total * warmup_rate)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.85, patience=0)


    save_dir = os.path.join('checkpoint',  'xlnet')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    Trainer(model, optimizer, scheduler, train_loader, epoches, device, save_dir, start_epoch)
