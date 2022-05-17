# -*- coding: utf-8 -*-
'''
# Created on 2021/01/20 10:22:48
# @filename: attention.py
# @author: tcxia
'''

import configparser
import json
import os
import matplotlib.pyplot as plt
import math


import torch
from transformers import BertModel, BertConfig

from utils.preprocess import Preprocessing

class Pretrainer:
    def __init__(self, batch_size, max_seq_len, device='cpu') -> None:
        super().__init__()
        config_ = configparser.ConfigParser()
        config_.read('')
        self.config = config_['DEFAULT']

        self.vocab_size = int(self.config['vocab_size'])
        self.batch_size = batch_size

        self.device = device

        self.max_seq_len = max_seq_len

        bertConfig =  BertConfig.from_pretrained('')
        self.bertModel = BertModel.from_pretrained('')
        self.bertModel.to(device)

        self.word2idx = self.load_dic('')
        self.process_batch = Preprocessing(
            max_position=max_seq_len,
            hidden_dim=bertConfig.hidden_size,
            word2idx=self.word2idx
        )


    def load_dic(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


    def load_model(self, model, dir_path='./output'):
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        checkpoint = torch.load(checkpoint_dir)
        checkpoint['model_state_dict'] = {k[5:]:v for k, v in checkpoint['model_state_dict'].items() if k[:4] == 'bert'}
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        torch.cuda.empty_cache()
        model.to(self.device)

    def __call__(self, text_list, batch_size=1):
        if isinstance(text_list, str):
            text_list = [text_list, ]
        len_ = len(text_list)
        text_list = [i for i in text_list if len(i) != 0]
        if len(text_list) == 0:
            raise NotImplementedError("输入文本为空")

        max_seq_len = max([len(i) for i in text_list])
        text_tokens, pos_enc = self.process_batch(text_list, max_seq_len=max_seq_len)
        pos_enc = torch.unsqueeze(pos_enc, dim=0).to(self.device)

        # 正向
        n_batches = math.ceil(len(text_tokens) / batch_size)

        # 数据按mini batch切片过正向, 这里为了可视化所以吧batch size设为1
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            # 切片
            texts_tokens_ = text_tokens[start:end].to(self.device)
            attention_matrices = self.bert_model.forward(
                input_ids=texts_tokens_,
                positional_enc=pos_enc,
                get_attention_matrices=True)
            # 因为batch size=1所以直接返回每层的注意力矩阵
            return [i.detach().numpy() for i in attention_matrices]



    def find_most_recent_state_dict(self, dir_path):
        dic_lis = [i for i in os.listdir(dir_path)]
        if len(dic_lis) == 0:
            raise FileNotFoundError('can not find any state dict {}'.format(dir_path))
        dic_lis = [i for i in dic_lis if 'model' in i]
        dic_lis = sorted(dic_lis, key=lambda k:int(k.split('.')[-1]))
        return dir_path + '/' + dic_lis[-1]




    def plot_attention(self, text, attention_matrics, layer_num, head_num):
        labels = [i + " " for i in list(text)]
        labels = ['[CLS] ',] + labels + ['[SEP]', ]
        plt.figure(figsize=(8, 8))
        plt.imshow(attention_matrics[layer_num][0][head_num])
        plt.yticks(range(len(labels)), labels, fontsize=18)
        plt.xticks(range(len(labels)), labels, fontsize=18)
        plt.savefig('attention_{}_{}.jpg'.format(layer_num, head_num))
