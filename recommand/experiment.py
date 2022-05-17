# -*- coding: utf-8 -*-
'''
# Created on 2021/03/19 10:47:42
# @filename: experiment.py
# @author: tcxia
'''

from evalFunc import Metric
from model.userIIF import userIIF
from model.userCF import UserCF
from data.dataset import Dataset


class Experiment():
    def __init__(self,
                 M,
                 K,
                 N,
                 fp='/data/nlp_dataset/recommandation/ml-1m/ratings.dat',
                 rt='userCF') -> None:
        self.M = M
        self.N = N
        self.K = K
        self.fp = fp
        self.rt =  rt
        self.alg = {"userCF": UserCF, "userIIF": userIIF}


    def epoch(self, train, test):
        getRecommandation = self.alg[self.rt](train, self.K, self.N)
        metric = Metric(train, test, getRecommandation)
        return metric.eval()

    def run(self):
        metrics = {'precision': 0, 'recall': 0, 'coverage': 0, 'popularity': 0}
        dataset = Dataset(self.fp)
        for ii in range(self.M):
            train, test, _  = dataset.splitData(self.M, ii)
            print('Experiment {}'.format(ii))
            metric = self.epoch(train, test)
            metrics = {k:metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print("Average Result (M={}, K={}, N={}): {}".format(self.M, self.K, self.N, metrics))


if __name__ == '__main__':
    M, N = 8, 10
    for K in [5, 10, 20]:
        exp = Experiment(M, K, N, rt='userCF')
        exp.run()
