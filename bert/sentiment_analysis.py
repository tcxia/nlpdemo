# -*- coding: utf-8 -*-
'''
# Created on Nov-11-20 13:52
# sentiment_analysis.py
# @author: tcxia
'''

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from transformers import pipeline

classifier = pipeline('sentiment-analysis')
result = classifier("We are happy to include pipeline into the transformers repository.")
print(result)


tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
model = AutoModelForSequenceClassification.from_pretrained(
    "hfl/chinese-bert-wwm-ext", return_dict=True)

classes = ["not paraphrase", "is paraphrase"]

sequence_0 = "今天 天气 真的 不错"
sequence_1 = "苹果 公司 发布 新品 mac"
sequence_2 = "又 是 一个 好 天气"

paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="pt")
not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="pt")

paraphrase_classification_logits = model(**paraphrase).logits
not_paraphrase_classification_logits = model(**not_paraphrase).logits

paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]

for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")
