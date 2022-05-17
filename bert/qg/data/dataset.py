# -*- coding: utf-8 -*-
'''
# Created on 11 月-18-20 11:14
# dataset.py
# @author: tcxia
'''

import pickle
from torch.utils.data import Dataset, DataLoader

class BertDataset(Dataset):
    def __init__(self, path) -> None:
        with open(path, 'rb') as fr:
            self.data = pickle.load(fr)

    def __len__(self) -> int:
        return self.data['input_ids'].shape[0]
    
    def __getitem__(self, index: int):
        return (((self.data['input_ids'][index],
                self.data['attention_mask'][index],
                self.data['token_type_ids'][index]),

                self.data['input_len'][index]),

                (self.data['output_ids'][index],
                self.data['output_len'][index])
                )

if __name__ == "__main__":
    ds = BertDataset('/data/nlp_dataset/squad/dev')
    dl = DataLoader(ds, batch_size=8, num_workers=0, shuffle=True, pin_memory=True)
    src, trg = next(iter(dl))
    src, src_len = src

    print(src[0][0].shape)
    print(src[1][0].shape)
    print(src[2][0].shape)