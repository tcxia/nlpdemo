# -*- coding: utf-8 -*-
'''
# Created on 2021/01/08 17:55:59
# @filename: util.py
# @author: tcxia
'''

import os
from tqdm import tqdm
import json
import re
import random

def save_checkpoint(checkpoint_dict, file):
    with open(file, 'w', encoding='utf-8') as f_out:
        json.dump(checkpoint_dict, f_out, ensure_ascii=False, indent=2)

def load_checkpoint(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        checkpoint_dict = json.load(f_in) 
    return checkpoint_dict



def get_tag2idx(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((tag, idx) for idx, tag in enumerate(tagset))

def get_idx2tag(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((idx, tag) for idx, tag in enumerate(tagset))


def read_data(file, tokenizer=None, max_len=128):
    token_list = []
    e1_mask_list = []
    e2_mask_list = []
    tags = []

    with open(file, 'r', encoding='utf-8') as f_in:
        # lines = f_in.readlines()
        # print(lines[0])
        for line in tqdm(f_in):
            line = line.strip()
            item = json.loads(line)
            tokens, pos_e1, pos_e2 = tokenizer.tokenize(item)
            if pos_e1[0] < max_len - 1 and pos_e1[1] < max_len and pos_e2[0] < max_len - 1 and pos_e2[1] < max_len:
                token_list.append(tokens)
                e1_mask = convert_pos_to_mask(pos_e1, max_len)
                e2_mask = convert_pos_to_mask(pos_e2, max_len)
                e1_mask_list.append(e1_mask)
                e2_mask_list.append(e2_mask)
                tag = item['relation']
                tags.append(tag)
    return token_list, e1_mask_list, e2_mask_list, tags


def convert_pos_to_mask(e_pos, max_len=128):
    e_pos_mask = [0] * max_len
    for i in range(e_pos[0], e_pos[1]):
        e_pos_mask[i] = 1
    return e_pos_mask


def convert_data(line):
    head_name, tail_name, relation, text = re.split(r'\t', line)
    obj1 = re.search(head_name, text)
    obj2 = re.search(tail_name, text)
    if obj1 and obj2:
        head_pos = obj1.span()
        tail_pos = obj2.span()
        item = {
            'h':{
                'name': head_name,
                'pos': head_pos
            },
            't':{
                'name': tail_name,
                'pos':tail_pos
            },
            'relation': relation,
            'text': text
        }
        return item
    else:
        return None

def save_data(lines, file):
    unk_count = 0
    with open(file, 'w', encoding='utf-8') as f_out:
        for line in lines:
            item = convert_data(line)
            if item is None:
                continue
            if item['relation'] == 'unknown':
                unk_count += 1
            json_str = json.dumps(item, ensure_ascii=False)
            f_out.write('{}\n'.format(json_str))
    print("unk的比例:{}/{}={}".format(unk_count, len(lines), unk_count / len(lines)))

def build_data(data_dir):
    file = os.path.join(data_dir, 'all_data.txt')
    train_file = os.path.join(data_dir, 'train.data')
    dev_file = os.path.join(data_dir, 'dev.data')
    with open(file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    lines = [line.strip() for line in lines]
    random.shuffle(lines)
    lines_len = len(lines)
    train_data = lines[:lines_len * 7 // 10]
    dev_data = lines[lines_len * 7 // 10:]
    print("===== generate train =====")
    save_data(train_data, train_file)
    print("===== generate val =====")
    save_data(dev_data, dev_file)






if __name__ == "__main__":
    file_dir = '/data/nlp_dataset/relate'
    build_data(file_dir)
