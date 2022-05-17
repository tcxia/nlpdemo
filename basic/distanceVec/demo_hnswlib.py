# -*- coding: utf-8 -*-
'''
# Created on 2021/03/08 10:46:06
# @filename: demo_hnswlib.py
# @author: tcxia
'''


import hnswlib
import numpy as np
import logging

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def get_vector(path, batch_size):
    index = 0
    labels = []
    vectors = []
    for line in open(path, 'r'):
        line = line.strip().split(' ')
        label, vector = line[0], line[1:]
        labels.append(label)
        vectors.append([float(dim) for dim in vector])
        index += 1
        if index >= batch_size:
            yield np.array(labels), np.array(vectors)
            labels.clear()
            vectors.clear()
            index = 0
    yield np.array(labels), np.array(vectors)



def build_hnsw_index(path, dim, num_elements, batch_size):
    count = 0
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=200, M=16)
    for labels, vectors in get_vector(path, batch_size):
        index.add_items(vectors, labels)
        count += 1
        logging.info("add items index: {}".format(count * batch_size))
    return index


def top_k(index, data, k):
    index.set_ef(int(k * 1.2))
    labels, distance = index.knn_query(data, k=k)
    return labels, distance


if __name__ == "__main__":
    path = '/data/nlp_dataset/glove.small.txt'  # 词向量文件输入路径
    dim = 200
    num_elements = 10
    batch_size = 10
    index = build_hnsw_index(path, dim, num_elements, batch_size)
    k = 2
    count = 0
    output_path = '/data/nlp_dataset/mat_cos.txt'
    output = open(output_path, 'w')
    for labels, vectors in get_vector(path, batch_size):
        targets, distances = top_k(index, vectors, k)
        for label, label_targets, label_distances in zip(labels, targets, label_distances):
            all_targets = []
            for target, distance in zip(label_targets, label_distances):
                line = "{}:{}".format(target, distance)
                all_targets.append(line)
            line = "{}\t{}\n".format(label, ",".join(all_targets))
            output.write(line)
        count += 1
        logging.info("build top k index: {}".format(count * batch_size))
