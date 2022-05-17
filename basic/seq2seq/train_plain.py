# -*- coding: utf-8 -*-
'''
# Created on 12月-04-20 16:29
# @filename: train.py
# @author: tcxia
'''

import torch
import torch.nn as nn

from loguru import logger

from data.process import load_data, build_vocab, encode, gen_example
from model.plain import PlainSeq2Seq, PlainEncoder, PlainDecoder



batch_size = 16
hidden_size = 100
dropout_rate = 0.2
epochs = 10


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


def Trainer(epoch, model, train_dataloader, optimizer, device):
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
            logger.info("Epoch: {} | Iter: {} | Loss: {:.4f}".format(epoch, i, loss.item()))



if __name__ == "__main__":
    train_path = '/data/nlp_dataset/nmt/en-cn/train.txt'
    # dev_path = ''

    train_cn_raw, train_en_raw = load_data(train_path)
    # dev_cn, dev_en = load_data(dev_path)
    train_cn_dict, train_cn_words = build_vocab(train_cn_raw)
    train_en_dict, train_en_words = build_vocab(train_en_raw)

    train_en, train_cn = encode(train_en_raw, train_cn_raw, train_en_dict,
                                train_cn_dict)

    train_data = gen_example(train_en, train_cn, batch_size)


    plain_encoder = PlainEncoder(train_en_words, hidden_size, dropout_rate)
    plain_decoder = PlainDecoder(train_cn_words, hidden_size, dropout_rate)

    model = PlainSeq2Seq(plain_encoder, plain_decoder)
    model = model.to(device)


    loss_fn = LMCriterion().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        Trainer(epoch, model, train_data, optimizer, device)
