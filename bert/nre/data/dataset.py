# -*- coding: utf-8 -*-
'''
# Created on 11 月-25-20 13:07
# dataset.py
# @author: tcxia
'''
import json
import numpy as np
import torch.utils.data as tud
from transformers import AutoTokenizer


def read_ret(ret_path):
    pred2id = {}
    id2pred = {}
    pred2id['NA'] = 0
    id2pred[0] = 'NA'
    with open(ret_path, 'r', encoding='utf-8') as fr:
        for r in fr:
            ret = json.loads(r)
            if ret['predicate'] not in pred2id:
                id2pred[len(pred2id)] = ret['predicate']
                pred2id[ret['predicate']] = len(pred2id)
    return pred2id, id2pred


class NREDataset(tud.Dataset):
    def __init__(self, data_path, pred2id, tokenizer) -> None:
        super(NREDataset, self).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.pred2id = pred2id
        with open(data_path, 'r', encoding='utf-8') as fin:
            for idx in fin:
                cont = json.loads(idx)
                text = cont['text']
                spos = [(spo['subject'], spo['predicate'], spo['object']) for spo in cont['spo_list']]
                self.data.append({'text': text, 'spos': spos})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        each_data = self.data[index]
        tokens = self.tokenizer.tokenize('[CLS]' + each_data['text'])
        if len(tokens) > 256:
            tokens = tokens[:256]
        tokens.append('[SEP]')

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        token_type_ids = [0 if i <= token_ids.index(102) else 1 for i in range(len(token_ids))]

        spoes = {}
        for sub, pred, obj in each_data['spos']:
            token_sub = self.tokenizer.tokenize(sub)
            token_sub_id = self.tokenizer.convert_tokens_to_ids(token_sub)
            token_sub_id_index = self.search_entity(token_sub_id, token_ids)

            token_obj = self.tokenizer.tokenize(obj)
            token_obj_id = self.tokenizer.convert_tokens_to_ids(token_obj)
            token_obj_id_index = self.search_entity(token_obj_id, token_ids)

            token_ret_id = self.pred2id[pred]

            if token_sub_id_index != -1 and token_obj_id_index != -1:
                token_sub_id = (token_sub_id_index, token_sub_id_index + len(token_sub_id) - 1)
                token_obj_id = (token_obj_id_index, token_obj_id_index + len(token_obj_id) - 1, token_ret_id)
                if token_sub_id not in spoes:
                    spoes[token_sub_id] = []
                spoes[token_sub_id].append(token_obj_id)
        
        if spoes:
            subject_labels = np.zeros((len(token_ids), 2))
            for sub in spoes:
                subject_labels[sub[0], 0] = 1
                subject_labels[sub[1], 1] = 1
            start, end = np.array(list(spoes.keys())).T
            start = np.random.choice(start)
            end = np.random.choice(end[end >= start])
            sub_id = (start, end)

            object_labels = np.zeros((len(token_ids), len(self.pred2id), 2))
            for obj in spoes.get(sub_id, []):
                object_labels[obj[0], obj[2], 0] = 1
                object_labels[obj[1], obj[2], 1] = 1

            output = {
                'token_ids': token_ids,
                'token_type_ids': token_type_ids,
                'sub_ids': sub_id,
                'sub_labels': subject_labels,
                'obj_labels': object_labels,
            }
            return output
        else:
            return self.__getitem__(index + 1)

    def search_entity(self, pattern, sequence):
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i+n] == pattern:
                return i
        return -1


if __name__ == "__main__":
    dev_path = '/data/nlp_dataset/NRE/extract/dev_data.json'
    ret_path = '/data/nlp_dataset/NRE/extract/all_50_schemas'
    pretrain_path = '/data/nlp_dataset/pre_train_models/chinese-bert-wwm-ext'
    tokenizer = AutoTokenizer.from_pretrained(pretrain_path)

    pred2id, id2pred = read_ret(ret_path)

    dev_set = NREDataset(dev_path, pred2id, tokenizer)


    ds = tud.DataLoader(dev_set, batch_size=1, shuffle=True, num_workers=0)
    output = next(iter(ds))
    print(output)
