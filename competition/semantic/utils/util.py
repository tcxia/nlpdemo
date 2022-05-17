# -*- coding: utf-8 -*-
'''
# Created on 2021/03/23 17:42:18
# @filename: util.py
# @author: tcxia
'''
import pandas as pd


def correct_pred(output_prob, targets):
    _, out_classes = output_prob.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()

def get_word_count(train_file, test_file):
    df_train = pd.read_table(train_file, names=['test_a', 'test_b',
                                                'label']).fillna('0')
    df_test = pd.read_table(test_file, names=['test_a', 'test_b']).fillna('0')
    df = pd.concat([df_train, df_test], ignore_index=True)
    df['text'] = df['test_a'] + " " + df['test_b']
    data = df['text'].values.tolist()

    data_i = list(map(str2int, data))
    return get_each_word(data_i)

def str2int(sentence):
    sentence = sentence.strip().split(' ')
    return [int(s) for s in sentence]

def get_each_word(sent):
    count = set()
    for s in sent:
        for w in s:
            if w not in count:
                count.add(w)
            else:
                continue
    return len(count)

if __name__ == "__main__":
    train_file = "/data/nlp_dataset/oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv"
    test_file = "/data/nlp_dataset/oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv"
    word = get_word_count(train_file, test_file)
    print(word)
