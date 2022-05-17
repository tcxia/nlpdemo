# -*- coding: utf-8 -*-
'''
# Created on 2021/03/25 13:13:43
# @filename: demo.py
# @author: tcxia
'''

import pandas as pd

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.utils.data as tud

# from models.model import TextMatch_Bert_test
from models.xlnet import TextMatch_XLNet_test
from data.datasets import TextDatasetTest


probs = []
def Prediction(model, test_dataloader, device):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, desc="[Predicting]")):
            segs, seg_masks, seg_segments = batch
            segs = segs.to(device)
            seg_masks = seg_masks.to(device)
            seg_segments = seg_segments.to(device)

            outputs = model(segs, seg_masks, seg_segments)
            logits = outputs.logits
            # print(logits)
            prob = F.softmax(logits, dim=-1)
            # print(prob)
            prob = prob[:, 1].cpu().numpy()
            # print(prob)

            # prob = torch.max(F.softmax(logits), dim=1)[0]
            # prob = prob.data.cpu().numpy().tolist()
            # probs.extend(prob)

            probs.extend(prob)

if __name__ == "__main__":
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    test_file = '/data/nlp_dataset/oppo_breeno_round1_data/gaiic_track3_round1_testB_20210317.tsv'
    test_set = TextDatasetTest(test_file)
    test_loader = tud.DataLoader(test_set, batch_size=32, shuffle=False)


    pretrained_path = '/data/nlp_dataset/pre_train_models/chinese-xlnet-base/config.json'
    model = TextMatch_XLNet_test(pretrained_path)
    model.to(device)

    checkpoint_file = 'checkpoint/xlnet/epoch_14.pth.tar'
    checkpoint = torch.load(checkpoint_file)
    # print(checkpoint['model'])
    model.load_state_dict(checkpoint['model'])

    Prediction(model, test_loader, device)

    # pred = torch.max(F.softmax(outputs[0]), dim=1)[1]
    # pred_label = pred.data.cpu().numpy().squeeze()
    # print(pred, pred_label)

    print(len(probs))

    pd.DataFrame(probs).to_csv('/data/nlp_dataset/oppo_breeno_round1_data/result_xlnet_b.csv', index=False, header=False)