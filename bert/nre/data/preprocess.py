# -*- coding: utf-8 -*-
'''
# Created on 11 月-25-20 14:28
# preprocess.py
# @author: tcxia
'''


def load_data(relation_path, rel2sent_path, sentence_path):
    with open(relation_path, 'r', encoding='utf-8') as fr:
        rels = fr.readlines()
    rel2id = {}
    for rel in rels:
        rel = rel.strip().split()
        rel2id[rel[0]] = rel[1]

    with open(sentence_path, 'r', encoding='utf-8') as fs:
        sents = fs.readlines()
    id2sent = {}
    for sent in sents:
        sent = sent.strip().split('\t')
        id2sent[sent[0]] = sent[1]

    with open(rel2sent_path, 'r', encoding='utf-8') as frs:
        rel2sents = frs.readlines()
    sent2rel = {}
    for rs in rel2sents:
        rs = rs.strip().split('\t')
        sent2rel[rs[0]] = rs[1]
    return rel2id, id2sent, sent2rel