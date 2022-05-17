# -*- coding: utf-8 -*-
'''
# Created on 2021/04/12 15:52:49
# @filename: dataset.py
# @author: tcxia
'''

import json
from transformers import XLNetTokenizer

import torch

class Example():
    def __init__(self, q_id, content, question, choices, label):
        self.q_id = q_id
        self.content = content
        self.question = question
        self.label = label
        self.choices = choices

    """
        1. __str__功能：将实例对象按照自定义的格式用字符串的形式显示出来，提高可读性
        2. 实例化的对象在打印或时会默认调用__str__方法，如果类没有重写这个方法，默认调用父类object的__str__方法
        3. object的__str__方法内部是pass，所以打印的是内存地址。如果当前类重写了这个方法，会自动调用重写后的方法
    """
    def __str__(self) -> str:
        return self.__repr__()

    """
        1. __repr__如果用IDE软件操作，功能与__str__完全一样，都是实例可视化显示
        2. 开发中如果用户需要可视化实例内容，只需要重写__str__或者__repr__方法之一即可。如果两个都有的话，默认调用__str__
        3. 两者的区别就是使用命令行操作
            3.1 __str__重写后，如果直接实例stu回车的话话，显示的是stu实例在内存中的地址，跟print(stu)不一样
            3.2 __repr__重写后，如果直接实例stu回车的话，效果跟使用print(stu)一样，返回内容，不是内存地址
    """
    def __repr__(self) -> str:
        data = [
            f"q_id: {self.q_id}",
            f"content: {self.content}",
            f"question: {self.question}",
            f"choice_A: {self.choices[0]}",
            f"choice_B: {self.choices[1]}",
            f"choice_C: {self.choices[2]}",
            f"choice_D: {self.choices[3]}",
            f"answer_label:{self.label}"
        ]
        return ",".join(data)

class InputFeatures():
    def __init__(self, example_id, choice_feature, label) -> None:
        self.example_id = example_id
        self.choice_features = [
            {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_id,
            }
            for _, input_ids, input_mask, segment_id in choice_feature
        ]
        self.label = label

def read_file(path):
    with open(path, 'r', encoding='utf-8') as fr:
        datas = json.load(fr)

    examples = []
    for data in datas:
        content = data['Content']
        questions = data['Questions']
        for item in questions:
            question_id = item['Q_id']
            question = item['Question']
            choices = item['Choices']
            # print(choices[3])
            answer_label = ord(item['Answer']) - ord('A')
            # answer_label = item['Answer']
            examples.append(
                Example(question_id, content, question, choices, answer_label)
            )
    return examples

def convert_features(examples, tokenizer, max_seq_len):
    features = []
    for index, example in enumerate(examples):
        content_tokens = tokenizer.tokenize(example.content)
        question_tokens = tokenizer.tokenize(example.question)

        choice_feature = []
        for i, choice in enumerate(example.choices):
            content_tokens_choice = content_tokens[:]
            choice_tokens = question_tokens + tokenizer.tokenize(choice)

            trunc_seq(content_tokens_choice, choice_tokens, max_seq_len - 3)

            tokens = ['[CLS]'] + content_tokens_choice + ['[SEP]'] + choice_tokens + ['[SEP]']
            segment_ids = [0] * (len(content_tokens_choice) + 2) + [1] * (len(choice_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_len - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_len
            assert len(input_mask) == max_seq_len
            assert len(segment_ids) == max_seq_len

            choice_feature.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        features.append(
            InputFeatures(example_id=example.q_id, choice_feature=choice_feature, label=label)
        )
    return features


def trunc_seq(tokens_a, tokens_b, max_len):
    while 1:
        total_len = len(tokens_a) + len(tokens_b)
        if total_len <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def select_field(features, field):
    return [
        [
            choice[field] for choice in feature.choice_features
        ]
        for feature in features
    ]





if __name__ == '__main__':
    file_path = '/data/nlp_dataset/qa-public/train.json'
    examples = read_file(file_path)
    # print(examples[0])

    pretrained_path = '/data/nlp_dataset/pre_train_models/chinese-xlnet-base'
    tokenizer = XLNetTokenizer.from_pretrained(pretrained_path, do_lower_case=True)
    features = convert_features(examples, tokenizer, 128)
    # print(features[0])

    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_masks = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segments_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    print(all_input_ids)
