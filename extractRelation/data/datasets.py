# -*- coding: utf-8 -*-
'''
# Created on 2021/01/08 17:47:34
# @filename: datasets.py
# @author: tcxia
'''

from random import shuffle
import torch
import torch.utils.data as tud

from transformers import BertTokenizer

import sys
sys.path.append('..')

from utils.util import read_data, get_tag2idx


class MyTokenizer():
    def __init__(self, pretrained_model_path=None, mask_entity=False) -> None:
        super().__init__()
        self.pretrained_model_path = pretrained_model_path
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
        self.mask_entity = mask_entity

    def tokenize(self, item):
        sentence = item['text']
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            pos_min = pos_head
            pos_max = pos_tail
            rev = False

        sent0 = self.bert_tokenizer.tokenize(sentence[:pos_min[0]])
        ent0 = self.bert_tokenizer.tokenize(sentence[pos_min[0]:pos_max[1]])
        sent1 = self.bert_tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
        ent1 = self.bert_tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
        sent2 = self.bert_tokenizer.tokenize(sentence[pos_max[1]:])

        if rev:
            if self.mask_entity:
                ent0 = ['[unused6]']
                ent1 = ['[unused5]']
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [len(sent0) + len(ent0) + len(sent1), len(sent0) + len(ent0) + len(sent1) + len(ent1)]
        else:
            if self.mask_entity:
                ent0 = ['[unused5]']
                ent1 = ['[unused6]']
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [len(sent0) + len(ent0) + len(sent1), len(sent0) + len(ent0) + len(sent1) + len(ent1)]

        tokens = sent0 + ent0 + sent1 + ent1 + sent2

        re_tokens = ['[CLS]']
        cur_pos = 0
        pos1 = [0, 0]
        pos2 = [0, 0]
        for token in tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                pos1[0] = len(re_tokens)
                re_tokens.append('[unused1]')
            if cur_pos == pos_tail[0]:
                pos2[0] = len(re_tokens)
                re_tokens.append('[unused2]')
            re_tokens.append(token)

            if cur_pos == pos_head[1] - 1:
                re_tokens.append('[unused3]')
                pos1[1] = len(re_tokens)
            if cur_pos == pos_tail[1] - 1:
                re_tokens.append('[unused4]')
                pos2[1] = len(re_tokens)
            cur_pos += 1
        re_tokens.append('[SEP]')
        return re_tokens[1:-1], pos1, pos2


class SentenceREDataset(tud.Dataset):
    def __init__(self, data_file, tagset_file, pretrained_model_path=None, max_len=128) -> None:
        super().__init__()
        self.data_file =  data_file
        self.tagset_file = tagset_file
        self.pretrained_model_path = pretrained_model_path

        self.max_len = max_len
        self.tokenizer = MyTokenizer(pretrained_model_path=self.pretrained_model_path)

        self.token_list, self.e1_mask_list, self.e2_mask_list, self.tags = read_data(data_file, tokenizer=self.tokenizer, max_len=self.max_len)

        self.tag2idx = get_tag2idx(self.tagset_file)

    def __len__(self) -> int:
        return len(self.tags)

    def __getitem__(self, index: int):
        if torch.is_tensor(index):
            index = index.tolist()

        sample_tokens = self.token_list[index]
        sample_e1_mask = self.e1_mask_list[index]
        sample_e2_mask = self.e2_mask_list[index]
        sample_tag = self.tags[index]

        encoder = self.tokenizer.bert_tokenizer.encode_plus(sample_tokens, max_length=self.max_len, pad_to_max_length=True)

        sample_token_ids = encoder['input_ids']
        sample_token_type_ids = encoder['token_type_ids']
        sample_attention_mask = encoder['attention_mask']
        sample_tag_id = self.tag2idx[sample_tag]

        sample = {
            'token_ids': torch.tensor(sample_token_ids),
            'token_type_ids': torch.tensor(sample_token_type_ids),
            'attention_mask': torch.tensor(sample_attention_mask),
            'e1_mask': torch.tensor(sample_e1_mask),
            'e2_mask': torch.tensor(sample_e2_mask),
            'tag_id': torch.tensor(sample_tag_id)
        }

        return sample


if __name__ == "__main__":
    pretrain_model_path = '/data/nlp_dataset/pre_train_models/bert-large-cased'
    train_file = '/data/nlp_dataset/relate/train.data'
    tag_file = '/data/nlp_dataset/relate/relation.txt'
    batch = SentenceREDataset(train_file, tag_file, pretrained_model_path=pretrain_model_path)
    dataloader = tud.DataLoader(batch, batch_size=1, shuffle=True)
    print(next(iter(dataloader)))
