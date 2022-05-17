# -*- coding: utf-8 -*-
'''
# Created on 2021/04/13 13:55:01
# @filename: dataset.py
# @author: tcxia
'''


import json

import torch
import torch.utils.data as tud

from transformers import XLNetTokenizer


class QADataset(tud.Dataset):
    def __init__(self, path, tokenizer, max_seq_len) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(path, 'r', encoding='utf-8') as fr:
            datas = json.load(fr)
        # print(len(datas)) # 6313
        examples = [] # 15421
        for data in datas:
            content = data['Content']
            questions = data['Questions']
            for item in questions:
                q_id = item['Q_id']
                question = item['Question']

                choices = item['Choices']
                choices_len = len(choices)

                choices += (4 - choices_len) * ['无效答案']

                assert len(choices) == 4

                answer_label = ord(item['Answer']) - ord('A')
                example = {
                    'q_id': q_id,
                    'question': question,
                    'content': content,
                    'answer_label': answer_label,
                    'choices_len': choices_len,
                    'choices': choices
                }
                examples.append(example)
        # print(self.examples[:2])
        features = self.convert_features(examples)
        # print(features[0])

        self.all_input_ids = torch.tensor(self._field(features, 'input_ids'), dtype=torch.long)
        self.all_input_masks = torch.tensor(self._field(features, 'input_masks'), dtype=torch.long)
        self.all_segment_ids = torch.tensor(self._field(features, 'segment_ids'), dtype=torch.long)
        self.all_labels = torch.tensor([f['label'] for f in features], dtype=torch.long)

    def __getitem__(self, index: int):
        return self.all_input_ids[index], self.all_input_masks[index], self.all_segment_ids[index], self.all_labels[index]
        
    def __len__(self) -> int:
        return len(self.all_labels)

    def convert_features(self, examples):
        features = []
        for i, example in enumerate(examples):
            content_tokens = self.tokenizer.tokenize(example['content'])
            question_tokens = self.tokenizer.tokenize(example['question'])
            answer_label = example['answer_label']

            choice_features = []
            for i, choice in enumerate(example['choices']):
                content_tokens_choice = content_tokens[:]
                choice_tokens = question_tokens + self.tokenizer.tokenize(choice)

                self._trunc_seq(content_tokens_choice, choice_tokens)

                tokens = ['[CLS]'] + content_tokens_choice + ['[SEP]'] + choice_tokens + ['[SEP]']
                segment_ids = [0] * (len(content_tokens_choice) + 2) + [1] * (len(choice_tokens) + 1)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_masks = [1] * len(input_ids)

                padding = [0] * (self.max_seq_len - len(input_ids))
                input_ids += padding
                input_masks += padding
                segment_ids += padding

                assert len(input_ids) == self.max_seq_len
                assert len(input_masks) == self.max_seq_len
                assert len(segment_ids) == self.max_seq_len

                choice_features.append((tokens, input_ids, input_masks, segment_ids))


            # print(len(choice_features))

            
            feature = {
                'label': answer_label,
                'q_id': example['q_id'],
                'choice_features': [
                    {
                    'input_ids': input_ids,
                    'input_masks': input_masks,
                    'segment_ids': segment_ids
                    } 
                    for _, input_ids, input_masks, segment_ids in choice_features
                ]
            }
            features.append(feature)
        return features

    def _trunc_seq(self, token_a, token_b):
        while 1:
            total_len = len(token_a) + len(token_b)
            if total_len <= self.max_seq_len - 3:
                break
            if len(token_a) > len(token_b):
                token_a.pop()
            else:
                token_b.pop()
    
    def _field(self, features, field):
        return [
            [
                choice[field] for choice in feature['choice_features']
            ] 
            for feature in features
        ]



if __name__ == '__main__':
    file_path = '/data/nlp_dataset/qa-public/train.json'

    pretrained_path = '/data/nlp_dataset/pre_train_models/chinese-xlnet-base'
    tokenizer = XLNetTokenizer.from_pretrained(pretrained_path, do_lower_case=True)

    data = QADataset(file_path, tokenizer, 128)
    # print(next(iter(data)))
    # print(len(data)) # 15421

    data_loader = tud.DataLoader(data, batch_size=4, shuffle=False)
    # print(data_loader)
