# -*- coding: utf-8 -*-
'''
# Created on 2021/01/13 16:01:40
# @filename: datasets.py
# @author: tcxia
'''

from functools import partial
import os

import pandas as pd
from tokenizers.pre_tokenizers import Sequence

import torch
import torch.utils.data as tud
from torch.utils.data import dataloader

from transformers import BertTokenizer, InputExample, InputFeatures

from multiprocessing import cpu_count, Pool
from tqdm import tqdm

import gensim
import numpy as np


class TextDataset(tud.Dataset):
    def __init__(self, bert_tokenizer, file_path, max_len=103) -> None:
        super().__init__()

        self.bert_tokenizer = bert_tokenizer
        self.max_len = max_len
        self.file = file_path
        self.seqs, self.seq_masks, self.seq_segments, self.labels = self.get_input(file_path)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.seqs[index], self.seq_masks[index], self.seq_segments[index], self.labels[index]

    def get_input(self, file_path):
        # df = pd.read_csv(self.file)
        # sent1 = df['sentence1'].values
        # sent2 = df['sentence2'].values
        # labels = df['labels'].values

        df = pd.read_table(file_path, names=['q1', 'q2', 'label']).fillna('0')
        labels = df['label'].values
        sent1 = df['q1'].values
        sent2 = df['q2'].values

        token_seq1 = list(map(self.seq_id, sent1.tolist()))
        token_seq2 = list(map(self.seq_id, sent2.tolist()))
        # 切词
        # token_seq1 = list(map(self.bert_tokenizer.tokenize, sent1))
        # token_seq2 = list(map(self.bert_tokenizer.tokenize, sent2))

        res = list(map(self.trunate_pad, token_seq1, token_seq2))
        seqs = [i[0] for i in res]
        seq_masks = [i[1] for i in res]
        seq_segments = [i[2] for i in res]
        return torch.Tensor(seqs).type(torch.long), torch.Tensor(seq_masks).type(torch.long), torch.Tensor(seq_segments).type(torch.long), torch.Tensor(labels).type(torch.long)

    def trunate_pad(self, token_seq1, token_seq2):
        if len(token_seq1) > ((self.max_len - 3) // 2):
            token_seq1 = token_seq1[0:(self.max_len - 3) // 2]
        if len(token_seq2) > ((self.max_len - 3) // 2):
            token_seq2 = token_seq2[0:(self.max_len - 3) // 2]

        # seq = ['[CLS]'] + token_seq1 + ['[SEP]'] + token_seq2 + ['[SEP]']
        seq = [101] + token_seq1 + [102] + token_seq2 + [102]

        seq_segment = [0] * (len(token_seq1) + 2) + [1] * (len(token_seq2) + 1)

        # seq = self.bert_tokenizer.convert_tokens_to_ids(seq)

        padding = [0] * (self.max_len - len(seq))

        seq_mask = [1] * len(seq) + padding

        seq_segment = seq_segment + padding

        seq += padding

        assert len(seq) == self.max_len
        assert len(seq_mask) == self.max_len
        assert len(seq_segment) == self.max_len
        return seq, seq_mask, seq_segment

    def seq_id(self, sent):
        sent = sent.split(' ')
        ret = []
        for s in sent:
            if len(s) == 3:
                s = s.replace('101', '20101').replace('102','20202')
                ret.append(s)
            else:
                ret.append(s)
        return [int(s) for s in ret]


class TextDatasetTest(tud.Dataset):
    def __init__(self, file_path, max_len=103) -> None:
        super().__init__()

        self.max_len = max_len
        self.file = file_path
        self.seqs, self.seq_masks, self.seq_segments = self.get_input(file_path)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, index: int):
        return self.seqs[index], self.seq_masks[index], self.seq_segments[index]

    def get_input(self, file_path):
        # df = pd.read_csv(self.file)
        # sent1 = df['sentence1'].values
        # sent2 = df['sentence2'].values
        # labels = df['labels'].values

        df = pd.read_table(file_path, names=['q1', 'q2']).fillna('0')
        sent1 = df['q1'].values
        sent2 = df['q2'].values

        token_seq1 = list(map(self.seq_id, sent1.tolist()))
        token_seq2 = list(map(self.seq_id, sent2.tolist()))
        # 切词
        # token_seq1 = list(map(self.bert_tokenizer.tokenize, sent1))
        # token_seq2 = list(map(self.bert_tokenizer.tokenize, sent2))

        res = list(map(self.trunate_pad, token_seq1, token_seq2))
        seqs = [i[0] for i in res]
        seq_masks = [i[1] for i in res]
        seq_segments = [i[2] for i in res]
        return torch.Tensor(seqs).type(
            torch.long), torch.Tensor(seq_masks).type(
                torch.long), torch.Tensor(seq_segments).type(
                    torch.long)

    def trunate_pad(self, token_seq1, token_seq2):
        if len(token_seq1) > ((self.max_len - 3) // 2):
            token_seq1 = token_seq1[0:(self.max_len - 3) // 2]
        if len(token_seq2) > ((self.max_len - 3) // 2):
            token_seq2 = token_seq2[0:(self.max_len - 3) // 2]

        # seq = ['[CLS]'] + token_seq1 + ['[SEP]'] + token_seq2 + ['[SEP]']
        seq = [101] + token_seq1 + [102] + token_seq2 + [102]

        seq_segment = [0] * (len(token_seq1) + 2) + [1] * (len(token_seq2) + 1)

        # seq = self.bert_tokenizer.convert_tokens_to_ids(seq)

        padding = [0] * (self.max_len - len(seq))

        seq_mask = [1] * len(seq) + padding

        seq_segment = seq_segment + padding

        seq += padding

        assert len(seq) == self.max_len
        assert len(seq_mask) == self.max_len
        assert len(seq_segment) == self.max_len
        return seq, seq_mask, seq_segment

    def seq_id(self, sent):
        sent = sent.split(' ')
        ret = []
        for s in sent:
            if len(s) == 3:
                s = s.replace('101', '20101').replace('102', '20202')
                ret.append(s)
            else:
                ret.append(s)
        return [int(s) for s in ret]

class PairProcesser:
    def load_data(self, filename):
        datas = pd.read_csv(filename).values.tolist()
        return datas

    def get_labels(self):
        return ['0', '1']

    def get_example(self, data_dir, set_type):
        file_map = {'train': 'train.csv', 'dev': 'dev.csv', 'test': 'test.example.csv'}
        file_name = os.path.join(data_dir, file_map[set_type])
        datas = self.load_data(file_name)
        examples = self.create_examples(datas, set_type)

    def create_examples(self, datas, set_type):
        examples = []
        for i, data in enumerate(datas):
            guid = data[0]
            text_a = data[2].strip()
            text_b = data[3].strip()
            if set_type == 'test':
                label = None
            else:
                label = str(int(data[4]))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


def classification_convert_example_to_feature(example,
                                              max_length=512,
                                              label_map=None,
                                              pad_on_left=False,
                                              pad_token=0,
                                              pad_token_segment_id=0,
                                              mask_padding_with_zero=True,
                                              set_type='train'):
    # 将文本转化为对应的ID
    inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_token=True, max_length=max_length)

    input_ids, token_type_ids = inputs['input_ids'], inputs['token_type_ids']
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    padded_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padded_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padded_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padded_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padded_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padded_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padded_length)

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length
    assert len(token_type_ids) == max_length

    if set_type != 'test':
        label =  label_map[example.label]
    else:
        label = None

    return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label)

def multi_classification_convert_examples_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


# 多线程处理（进程池）
def multi_classification_convert_examples_to_dataset(
        examples,
        tokenizer,
        max_length=512,
        label_list=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        threads=10,
        set_type='train'):
    label_map = dict(zip(label_list, range(len(label_list))))
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=multi_classification_convert_examples_to_features_init, initargs=(tokenizer, )) as p:
        annotate_ = partial(classification_convert_example_to_feature,
                            max_length=max_length,
                            label_map=label_map,
                            pad_on_left=pad_on_left,
                            pad_token=pad_token,
                            pad_token_segment_id=pad_token_segment_id,
                            mask_padding_with_zero=mask_padding_with_zero,
                            set_type=set_type)
        features = list(tqdm(p.imap(annotate_, examples, chunksize=32), total=len(examples), desc='convert squad example to features'))

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    if set_type != 'test':
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = tud.TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    else:
        dataset = tud.TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)

    del features
    return dataset



class SiameseData(tud.Dataset):
    def __init__(self, file, is_train=True) -> None:
        super().__init__()
        self.is_train = is_train
        if self.is_train:
            self.p, self.h, self.label = self.load_sentences(file)
        else:
            self.p, self.h = self.load_sentences(file)
        
        self.p_new = torch.from_numpy(self.p).type(torch.long)
        self.h_new = torch.from_numpy(self.h).type(torch.long)

    def __len__(self) -> int:
        if self.is_train:
            return len(self.label)
        else:
            return len(self.p)

    def __getitem__(self, index):
        if self.is_train:
            return self.p_new[index], self.h_new[index], self.label[index]
        else:
            return self.p_new[index], self.h_new[index]

    def load_sentences(self, file):
        if self.is_train:
            df = pd.read_table(file, names=['text_a', 'text_b', 'label']).fillna('0')
            p = df['text_a'].values.tolist()
            h = df['text_b'].values.tolist()
            label = df['label'].values
        else:
            df = pd.read_table(file, names=['text_a', 'text_b']).fillna('0')
            p = df['text_a'].values.tolist()
            h = df['text_b'].values.tolist()

        p_i = list(map(self.str2int, p))
        h_i = list(map(self.str2int, h))

        p_new = self.pad_sequences(p_i, maxlen=50)
        h_new = self.pad_sequences(h_i, maxlen=50)

        if self.is_train:
            return p_new, h_new, label
        else:
            return p_new, h_new

    def str2int(self, sentence):
        sentence = sentence.strip().split(' ')
        return [int(s) for s in sentence]

    def pad_sequences(self, sequences, maxlen=None, padding='post', truncating='post', value=0.):
        lengths = [len(s) for s in sequences]
        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)
        x = (np.ones((nb_samples, maxlen)) * value).astype('int32')
        for idx, s in enumerate(sequences):
            if len(s) == 0:
                continue
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError("Truncating type '%s' not understand" % padding)

            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError("Padding type '%s' not understand" % padding)
        return x

if __name__ == "__main__":
    train_file = '/data/nlp_dataset/oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv'
    test_file = '/data/nlp_dataset/oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv'
    data = SiameseData(train_file, is_train=True)
    # data.load_sentences(train_file)
    data_loader = tud.DataLoader(data, batch_size=4, shuffle=False)
    print(next(iter(data_loader)))
    # batch = next(iter(data_loader))
    # print(batch[0].dtype)
    # # print(batch[0].shape)



#     bert_pretrained_path = '/data/nlp_dataset/pre_train_models/bert-large-cased'
#     bert_tokenizer = BertTokenizer.from_pretrained(bert_pretrained_path)

# data = TextDataset(bert_tokenizer, train_file)
# print(next(iter(data)))
# data_loader = tud.DataLoader(data, batch_size=15, shuffle=False)
# print(list(data_loader))



# df_train = pd.read_table(train_file, names=['q1', 'q2', 'label']).fillna('0')
# sent1 = df_train["q1"].values
# sent2 = df_train["q2"].values

# sent1 = ['101 1010 1022 1014', '5101 4101 1020 8102 102']
# token_seq1 = list(map(seq_id, sent1))
# print(token_seq1)




# token_seq1 = list(map(bert_tokenizer.tokenize, sent1))
# token_seq2 = list(map(bert_tokenizer.tokenize, sent2))

# if len(token_seq1) > 50:
#     token_seq1 = token_seq1[0:50]
# if len(token_seq2) > 50:
#     token_seq2 = token_seq2[0:50]

# CLS 101
# SEP 102
# seq = ['[CLS]'] + token_seq1[0] + ['[SEP]'] + token_seq2[0] + ['[SEP]']
# print(seq)
# seq_ids = bert_tokenizer.convert_tokens_to_ids(seq)
# print(seq_ids)
