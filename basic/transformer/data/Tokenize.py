# -*- coding: utf-8 -*-
'''
# Created on 12月-07-20 10:57
# @filename: Tokenize.py
# @author: tcxia
'''

import spacy
import re


class tokenize(object):
    def __init__(self, lang) -> None:
        super().__init__()
        self.nlp = spacy.load(lang)

    def tokenizer(self, sentence):
        sentence = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ",
                          str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]
