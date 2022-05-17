# -*- coding: utf-8 -*-
'''
# Created on 12月-04-20 16:29
# @filename: train.py
# @author: tcxia
'''

import torch
import torch.nn as nn

import random
import numpy as np
from loguru import logger

from data.process import load_data, build_vocab, encode, gen_example
from model.attention import Encoder, Decoder, Seq2Seq



batch_size = 64
embed_size = 100
hidden_size = 100
dropout_rate = 0.2
epochs = 20


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# maksed cross entropy loss
class LMCriterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_x, target, mask):

        input_x = input_x.contiguous().view(-1, input_x.size(2))
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)

        output = -input_x.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


min_val_loss = []

def Trainer(epoch, model, train_dataloader, dev_dataloader, optimizer, scheduler, device):
    model.train()
    total_num_words = 0
    total_loss = 0.

    for i, batch in enumerate(train_dataloader):
        mb_x, mb_x_len, mb_y, mb_y_len = batch

        mb_x = torch.from_numpy(mb_x).to(device).long()
        mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
        mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
        mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
        mb_y_len = torch.from_numpy(mb_y_len - 1).to(device).long()

        mb_y_len[mb_y_len<=0] = 1

        # [batch_size, output_len, vocab_size]
        mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

        mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
        mb_out_mask = mb_out_mask.float()

        loss = loss_fn(mb_pred, mb_output, mb_out_mask)

        num_words = torch.sum(mb_y_len).item()
        total_loss += loss.item() * num_words
        total_num_words += num_words

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        if i % 100 == 0:
            logger.info("Epoch: {} | Iter: {} | Loss: {:.4f}".format(epoch + 1, i, loss.item()))

    logger.info("Epoch: {} | Traning Loss: {:.4f}".format(epoch + 1, total_loss / total_num_words))

    val_loss = Evaler(epoch, model, dev_dataloader, device)

    
    if len(min_val_loss) == 0 or val_loss < min(min_val_loss):
        logger.info('save model!')
        torch.save(model.state_dict(), 'checkpoint/seq2seq_attn.pth')
    else:
        scheduler.step()
    min_val_loss.append(val_loss)


def Evaler(epoch, model, dev_dataloader, device):
    model.eval()

    total_num_words = 0.
    total_loss = 0.
    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader):
            mb_x, mb_x_len, mb_y, mb_y_len = batch

            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len - 1).to(device).long()

            mb_y_len[mb_y_len<=0] = 1
            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

    logger.info("Epoch: {} | Evaluate Loss: {:.4f}".format(epoch + 1, total_loss / total_num_words))
    return total_loss / total_num_words




if __name__ == "__main__":
    train_path = '/data/nlp_dataset/nmt/en-cn/train.txt'
    dev_path = '/data/nlp_dataset/nmt/en-cn/dev.txt'
    total_path = '/data/nlp_dataset/nmt/en-cn/cmn.txt'


    pretrain_wordvec = '/data/nlp_dataset/glove_seq2seq.npy'
    wordvec = np.load(pretrain_wordvec)

    train_cn_raw, train_en_raw = load_data(train_path)
    train_cn_dict, train_cn_words = build_vocab(train_cn_raw)
    train_en_dict, train_en_words = build_vocab(train_en_raw)
    train_en, train_cn = encode(train_en_raw, train_cn_raw, train_en_dict, train_cn_dict)
    train_data = gen_example(train_en, train_cn, batch_size)
    random.shuffle(train_data)

    dev_cn_raw, dev_en_raw = load_data(dev_path)
    dev_cn_dict, dev_cn_words = build_vocab(dev_cn_raw)
    dev_en_dict, dev_en_words = build_vocab(dev_en_raw)
    dev_en, dev_cn = encode(dev_en_raw, dev_cn_raw, dev_en_dict, dev_cn_dict)
    dev_data = gen_example(dev_en, dev_cn, batch_size)


    total_cn_raw, total_en_raw = load_data(total_path)
    total_cn_dict, total_cn_words = build_vocab(total_cn_raw)
    total_en_dict, total_en_words = build_vocab(total_en_raw)
    total_en, total_cn = encode(total_en_raw, total_cn_raw, total_en_dict, total_cn_dict)
    total_data = gen_example(total_en, total_cn, batch_size)


    encoder = Encoder(total_en_words, embed_size, hidden_size, hidden_size,
                      dropout_rate, word_vec=wordvec)
    decoder = Decoder(total_cn_words, embed_size, hidden_size, hidden_size, dropout_rate)

    model = Seq2Seq(encoder, decoder)
    model = model.to(device)


    loss_fn = LMCriterion().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    for epoch in range(epochs):
        Trainer(epoch, model, train_data, dev_data, optimizer, scheduler, device)
