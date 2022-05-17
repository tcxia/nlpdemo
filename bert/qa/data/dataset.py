# -*- coding: utf-8 -*-
'''
# Created on 11 月-19-20 15:36
# dataset.py
# @author: tcxia
'''

import json
import numpy as np

from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer

model_path = '/data/nlp_dataset/pre_train_models/chinese-bert-wwm-ext'
tokenizer = AutoTokenizer.from_pretrained(model_path)

class WebQADataset(Dataset):
    def __init__(self, path) -> None:
        self.questions = []
        self.evidences = []
        self.answers = []

        with open(path, 'r', encoding='utf-8') as fr:
            data = json.load(fr)
            for key in data:
                item = data[key]
                question = item['question']

                evidence = item['evidences']
                for key_evi in evidence:
                    evi_item = evidence[key_evi]
                    self.questions.append(question)
                    self.evidences.append(evi_item['evidence'])
                    self.answers.append(evi_item['answer'][0])

    def __len__(self) -> int:
        return len(self.answers)

    def __getitem__(self, index: int):
        each_question = self.questions[index]
        # print(each_question)
        each_evidence = self.evidences[index]
        each_answer = self.answers[index]

        tokens = tokenizer.tokenize('[CLS]' + each_question + '[SEP]' + each_evidence)
        if len(tokens) > 512:
            tokens = tokens[:512]

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [
            # [CLS] : 101
            # [SEP] : 102
            0 if i <= token_ids.index(102) else 1
            for i in range(len(token_ids))
        ]
        answer_offset = (-1, -1)
        IsQA = 0
        answer_seq_label = len(token_ids) * [0]
        if each_answer != 'no_answer':
            answer_tokens = tokenizer.tokenize(each_answer)
            answer_offset = self.FindOffset(tokens, answer_tokens)
            if answer_offset:
                answer_seq_label[answer_offset[0]:answer_offset[1]] = [1] * len(answer_tokens)
                IsQA = 1
            else:
                answer_offset = (-1, -1)
        return tokens, token_ids, token_type_ids, answer_offset, answer_seq_label, IsQA


    def FindOffset(self, token_id, answer_id):
        n = len(token_id)
        m = len(answer_id)

        if n < m:
            return False

        for i in range(n - m + 1):
            if token_id[i: i + m] == answer_id:
                return (i, i + m)
        return False


    def get_samples_weights(self, negweight):
        samples_weight = []
        for ans in self.answers:
            if ans != 'no_answer':
                samples_weight.append(1.0)
            else:
                samples_weight.append(negweight)
        return np.array(samples_weight)


if __name__ == "__main__":
    ds = WebQADataset('/data/nlp_dataset/WebQA/me_train.json')
    dl = DataLoader(ds, batch_size=1, num_workers=0, shuffle=True, pin_memory=True)
    tokens, token_ids, _, answer_offset, answer_seq_label, _ = next(iter(dl))
    # print(answer_seq_label)
    # print(answer_offset)
    # print(tokens)
    print(token_ids)

