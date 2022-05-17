# -*- coding: utf-8 -*-
'''
# Created on 11 月-18-20 11:19
# preprocess.py
# @author: tcxia
'''

import json
import pickle
from transformers import BertTokenizer


class Preprocess(object):
    def __init__(self, squad_path, bert_model) -> None:
        with open(squad_path, 'r') as fr:
            data = json.load(fr)

        inputs, outputs = self._extract_squad_data(data)
        self.data = self._tokenize_data(inputs, outputs, bert_model)

    def save(self, path):
        with open(path, 'wb') as fw:
            pickle.dump(self.data, fw)

    def _extract_squad_data(self, data):
        data = data['data']

        inputs = []
        outputs = []

        for document in data:
            for paragraph in document['paragraphs']:
                context = paragraph['context']
                for qas in paragraph['qas']:
                    if len(qas['answers']) < 1:
                        continue
                    answer = qas['answers'][0]['text']
                    question = qas['question']
                    inputs.append((context, answer))
                    outputs.append(question)
        assert len(inputs) == len(outputs)
        inputs = inputs[:int(0.1 * len(inputs))]
        outputs = outputs[:int(0.1 * len(outputs))]
        return inputs, outputs


    def _tokenize_data(self, inputs, outputs, bert_model):
        # 加载预训练模型
        tokenizer = BertTokenizer.from_pretrained(bert_model)

        # 处理输入的inputs, 返回类型为pytorch
        data = tokenizer.batch_encode_plus(inputs,
                                           padding=True,
                                           return_tensors='pt')

        # 处理out
        out_dict = tokenizer.batch_encode_plus(outputs,
                                               padding=True,
                                               return_tensors='pt')
        # print(out_dict)

        data['output_ids'] = out_dict['input_ids']
        data['output_len'] = out_dict['attention_mask'].sum(dim=1) # 统计句子实际包含单词的数目
        data['input_len'] = data['attention_mask'].sum(dim=1) # 
        # print(data['input_len'])

        idx = (data['input_len'] <= 512) # 截断最长为512
        in_m = max(data['input_len'][idx])
        out_m = max(data['output_len'][idx])

        data['input_ids'] = data['input_ids'][idx, :in_m]
        data['attention_mask'] = data['attention_mask'][idx, :in_m]
        data['token_type_ids'] = data['token_type_ids'][idx, :in_m]
        data['input_len'] = data['input_len'][idx]

        data['output_ids'] = data['output_ids'][idx, :out_m]
        data['output_len'] = data['output_len'][idx]

        return data

if __name__ == "__main__":
    dataset = Preprocess('/data/squad/dev-v2.0.json',
                         '/data/transModel/bert-large-cased')
    # dataset.save('/data/squad/train')