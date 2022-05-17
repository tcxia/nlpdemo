# -*- coding: utf-8 -*-
'''
# Created on 12月-08-20 14:57
# @filename: pred_attn.py
# @author: tcxia
'''



import torch
import numpy as np

from data.process import load_data, build_vocab, encode, gen_example
from model.attention import Seq2Seq, Encoder, Decoder


embed_size = 100
hidden_size = 100
dropout_rate = 0.2





device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def Pred(model, index, test_en, test_cn_dict, inv_cn_dict, inv_en_dict):

    en_sent = " ".join(inv_en_dict[w] for w in test_en[index])
    cn_sent = " ".join(inv_cn_dict[w] for w in test_cn[index])
    print(en_sent)
    print(cn_sent)

    mb_x = torch.from_numpy(np.array(test_en[index]).reshape(1, -1)).long().to(device)
    mb_x_len = torch.from_numpy(np.array([len(test_en[index])])).long().to(device)

    bos = torch.Tensor([[test_cn_dict['BOS']]]).long().to(device)

    translate, attn = model.translate(mb_x, mb_x_len, bos)
    translation = [inv_cn_dict[i] for i in translate.data.cpu().numpy().reshape(-1)]

    trans = []
    for word in translation:
        if word != 'EOS':
            trans.append(word)
        else:
            break
    print("".join(trans))



if __name__ == "__main__":
    test_path = '/data/nlp_dataset/nmt/en-cn/test.txt'
    total_path = '/data/nlp_dataset/nmt/en-cn/cmn.txt'

    test_cn_raw, test_en_raw = load_data(test_path)
    test_cn_dict, test_cn_words = build_vocab(test_cn_raw)
    test_en_dict, test_en_words = build_vocab(test_en_raw)
    test_en, test_cn = encode(test_en_raw, test_cn_raw, test_en_dict, test_cn_dict)


    total_cn_raw, total_en_raw = load_data(total_path)
    total_cn_dict, total_cn_words = build_vocab(total_cn_raw)
    total_en_dict, total_en_words = build_vocab(total_en_raw)
    total_en, total_cn = encode(total_en_raw, total_cn_raw, total_en_dict, total_cn_dict)

    inv_en_dict = {v:k for k, v in test_en_dict.items()}
    inv_cn_dict = {v:k for k, v in test_cn_dict.items()}



    encoder = Encoder(total_en_words, embed_size, hidden_size, hidden_size, dropout_rate)
    decoder = Decoder(total_cn_words, embed_size, hidden_size, hidden_size, dropout_rate)

    model = Seq2Seq(encoder, decoder)
    model = model.to(device)

    model.load_state_dict(torch.load('checkpoint/seq2seq_attn.pth'))

    for i in range(10):
        Pred(model, i, test_en, test_cn_dict, inv_cn_dict, inv_en_dict)
