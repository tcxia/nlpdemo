# -*- coding: utf-8 -*-
'''
# Created on 12月-04-20 10:35
# @filename: process.py
# @author: tcxia
'''

from collections import Counter
import numpy as np

# 构建数据
def load_data(file_path):
    cn = []
    en = []

    with open(file_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()

    for line in lines:
        line = line.strip().split('\t')
        en.append(['BOS'] + line[0].lower().split(' ') + ['EOS'])
        cn.append(['BOS'] + [c for c in line[1]] + ['EOS'])
    return cn, en

# 构建单词表
UNK_IDX = 0
PAD_IDX = 1
def build_vocab(sentences, max_words=50000):
    word_count = Counter()
    for sentence in sentences:
        for s in sentence:
            word_count[s] += 1

    ls = word_count.most_common(max_words) # 返回统计字符元组
    # print(ls)
    total_words = len(ls) + 2
    word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
    word_dict['UNK'] = UNK_IDX
    word_dict['PAD'] = PAD_IDX
    return word_dict, total_words


# 编码句子--句子长度排序
def encode(en_sentences, cn_sentences, en_dict, cn_dict, sort_by_len=True):

    out_en_sentences = [[en_dict.get(w, 0) for w in sent] for sent in en_sentences]
    out_cn_sentences = [[cn_dict.get(w, 0) for w in sent] for sent in cn_sentences]

    #.....
    def len_argsort(seq):
        # sorted() 排序， key参数可以自定义规则，按seq[x]的长度排序， seq[0]为第一句话长度
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(out_en_sentences)
        # 同时排序中英文句子
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]
        out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]

    return out_en_sentences, out_cn_sentences


# 数据切分成batch
def get_minibatches(n, minibatch_size, shuffle=True):
    idx_list = np.arange(0, n, minibatch_size)  # [0, minibatch_size, minibatch_size*2, ..., n]
    # print(idx_list)
    if shuffle:
        np.random.shuffle(idx_list) # 打乱数据集

    minibatches = []
    for idx in idx_list:
        # 所有batch放在一个大列表里面
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches



def prepara_data(seqs):
    lengths = [len(seq) for seq in seqs] # 每个batch里面语句长度统计出来
    n_samples = len(seqs) # 一个batch包含多少条语句
    max_len = np.max(lengths) # 取出最长的语句长度，后面用这个做padding基准

    x = np.zeros((n_samples, max_len)).astype('int32')
    x_lengths = np.array(lengths).astype('int32')

    # 取出一个batch的每条语句和对应索引
    for idx, seq in enumerate(seqs):
        # 每条语句按行赋值给x, x会有一些零值没有被赋值
        x[idx, :lengths[idx]] = seq

    return x, x_lengths

def gen_example(en_sentences, cn_sentences, batch_size):
    minibatches = get_minibatches(len(en_sentences), batch_size)

    all_ex = []
    for minibatch in minibatches:
        mb_en_sentences = [en_sentences[t] for t in minibatch]
        mb_cn_sentences = [cn_sentences[t] for t in minibatch]

        # mb_x: [bacth_size, max_len]
        # mb_x_len: max_len
        mb_x, mb_x_len = prepara_data(mb_en_sentences)

        mb_y, mb_y_len = prepara_data(mb_cn_sentences)
        all_ex.append((mb_x, mb_x_len, mb_y, mb_y_len))

    return all_ex


def get_pretained_embed(glove_path, word_dict, embed_size=100):
    with open(glove_path, 'r') as fr:
        contents = fr.readlines()

    # print(content[:10])
    total_len = len(word_dict.keys())
    print(total_len)
    wordvec = np.zeros((total_len, embed_size))
    success_id = []
    for content in contents:
        content = content.strip().split(' ')
        idx = content[0]
        embedding = content[1:]

        if idx in word_dict.keys():
            word_id = word_dict[idx]
            success_id.append(word_id)
            wordvec[word_id, :embed_size] = embedding
        else:
            continue

    for word, word_id in word_dict.items():
        if word_id not in success_id:
            wordvec[word_id, :] = np.random.randn(1, embed_size)
        else:
            continue
    # print(success_id)
    np.save('/data/nlp_dataset/glove_seq2seq.npy', wordvec)
    return wordvec


if __name__ == "__main__":
    file_path = '/data/nlp_dataset/nmt/en-cn/cmn.txt'
    
    cn, en = load_data(file_path)
    # # print(cn[:10])
    # # print(en[:10])

    # cn_dict, cn_words = build_vocab(cn)
    en_dict, en_words = build_vocab(en)
    # print(en_dict)
    # # print(cn_dict, cn_words)


    # train_en, train_cn = encode(en, cn, en_dict, cn_dict)
    # # print(train_cn[:10])
    # # print(train_en[:10])

    # print(get_minibatches(len(train_en), 8)[:5])

    inv_en_dict = {v:k for k, v in en_dict.items()}
    # print(inv_en_dict)
    glove_path = '/data/nlp_dataset/glove.6B.100d.txt'
    word_vec = get_pretained_embed(glove_path, en_dict)
    print(en_dict['UNK'])
    print(word_vec[en_dict['UNK'], :])