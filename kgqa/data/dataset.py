# -*- coding: utf-8 -*-
'''
# Created on 2021/01/25 14:38:33
# @filename: dataset.py
# @author: tcxia
'''
from collections import defaultdict
import os
import numpy as np
import torch


class KGQADataset:
    def __init__(self, data_dir, reverse, batch_size) -> None:
        super().__init__()
        self.train_data = self.load_data(data_dir, 'train', reverse=reverse)
        self.valid_data = self.load_data(data_dir, 'valid', reverse=reverse)

        self.data = self.train_data + self.valid_data

        self.entities = self.get_entities(self.data)
        self.entity_idx = {self.entities[i]:i for i in range(len(self.entities))}
        
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)

        self.relations = self.train_relations + [i for i in self.valid_relations if i not in self.train_relations]
        self.relation_idx = {self.relations[i]:i for i in range(len(self.relations))}

        self.batch_size = batch_size
    
    def load_data(self, data_dir, data_type='train', reverse=False):
        with open(os.path.join(data_dir, data_type + '.txt'), 'r') as fr:
            data = fr.read().strip().split('\n')
            data = [i.strip('\t') for i in data]
            if reverse:
                data += [[i[2], i[1] + '_reverse', i[0]] for i in data]
        return data

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities
    
    def get_relations(self, data):
        relations = sorted(list(set(d[1] for d in data)))
        return relations

    def get_data_idx(self, data):
        data_idxs = [(self.entity_idx[data[i][0]], self.relation_idx[data[i][1]], self.entity_idx[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = torch.zeros([len(batch), len(self.entities)], dtype=torch.float32)
        for i, pair in enumerate(batch):
            targets[i, er_vocab[pair]] = 1.
        return np.array(batch), targets