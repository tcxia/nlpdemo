# -*- coding: utf-8 -*-
'''
# Created on 12月-17-20 13:59
# @filename: translate.py
# @author: tcxia
'''

import torch
from torch.autograd import Variable


from model.beam_search import beamSearch

def translate_sentence(sentence, model, src_field, trg_field, device='cpu'):
    model.eval()

    indexed = []
    sentence = src_field.preprocess(sentence)
    for tok in sentence:
        if src_field.vocab.stoi[tok] != 0:
            indexed.append(src_field.vocab.stoi[tok])

    sentence = Variable(torch.LongTensor([indexed]))
    sentence = sentence.to(device)

    sentence = beamSearch(sentence, model, src_field, trg_field)