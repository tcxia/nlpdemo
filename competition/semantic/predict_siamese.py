# -*- coding: utf-8 -*-
'''
# Created on 2021/03/31 14:23:04
# @filename: predict_siamese.py
# @author: tcxia
'''


import torch
import torch.utils.data as tud

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.datasets import SiameseData
from models.siamese import Siamese



probs = []
def SiaPred(model, test_dataloader, device):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, desc="[Testing]")):
            q, h = batch
            q = q.to(device)
            h = h.to(device)
            # print(q[0, :])
            logits, prob = model(q, h)
            # print(prob)
            probs.extend(prob[:,1].cpu().numpy())



if __name__ == "__main__":
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    test_file = '/data/nlp_dataset/oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv'
    test_set = SiameseData(test_file, is_train=False)
    # print(len(test_set))
    test_loader = tud.DataLoader(test_set, batch_size=128, shuffle=False)
    # print(next(iter(test_loader)))
    # embedding_matrix = np.load('checkpoint/siamese/embed.npy')
    # print(embedding_matrix.shape)

    # train_file = '/data/nlp_dataset/oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv'
    # wordcount = get_word_count(train_file, test_file)

    model = Siamese(22000, 100)
    model.to(device)

    checkpoint_file = 'checkpoint/siamese/epoch_100.pth.tar'
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model'])

    SiaPred(model, test_loader, device)

    pd.DataFrame(probs).to_csv('/data/nlp_dataset/oppo_breeno_round1_data/result_siamese.csv', index=False, header=False)