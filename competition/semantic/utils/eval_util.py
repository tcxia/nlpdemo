# -*- coding: utf-8 -*-
'''
# Created on 2021/03/29 17:53:39
# @filename: eval_util.py
# @author: tcxia
'''

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def compute_metrics(y_true, y_pred, average='micro'):
    result = {}
    f1 = f1_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    accuracy = accuracy_score(y_true, y_pred)
    result['f1'] = round(f1, 4)
    result['recall'] =  round(recall, 4)
    result['precision'] = round(precision, 4)
    result['accuracy'] = round(accuracy, 4)
    return result