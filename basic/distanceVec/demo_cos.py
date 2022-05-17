# -*- coding: utf-8 -*-
'''
# Created on 2021/03/08 11:02:22
# @filename: demo_cos.py
# @author: tcxia
'''


from scipy.spatial.distance import cosine
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from loguru import logger

with open('/data/nlp_dataset/glove.small.txt', 'r') as ft:
    datat = ft.readlines()

labels_t = []
vectors_t = []
for line in datat:
    line =  line.strip().split(' ')
    label, vector = line[0], line[1:]
    labels_t.append(label)
    vectors_t.append(np.array([float(vec) for vec in vector]))

logger.info('tibete vocab read finished!')

with open('/data/nlp_dataset/glove.small.txt', 'r') as fc:
    datac = fc.readlines()

labels_c = []
vectors_c = []
for line in datac:
    line = line.strip().split(' ')
    label, vector = line[0], line[1:]
    labels_c.append(label)
    vectors_c.append(np.array([float(vec) for vec in vector]))

logger.info('chinese vocab read finished!')

cos_mat = defaultdict(list)
## 值越大越相近

with open('/data/nlp_dataset/mat_ct_each.txt', 'w') as fw:
    for i in tqdm(range(len(vectors_t)), desc='tibete vocab'):
        for j in tqdm(range(len(vectors_c)), desc='chinese vocab'):
            vec_c = {}
            s = cosine(vectors_t[i], vectors_c[j])
            # print(s)
            if round(s, 2) > 0.5:
                vec_c[labels_c[j]] = round(s, 2)
                cos_mat[labels_t[i]].append(vec_c)
                fw.write(labels_t[i] + '\t' + str(vec_c) + '\n')

# print(cos_mat)
with open('/data/nlp_dataset/mat_ct_total.txt', 'w') as fw:
    for label, cos_label in cos_mat.items():
        fw.write(label + ' ' + "".join(str(cos_label)) + '\n')
