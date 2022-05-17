# -*- coding: utf-8 -*-
'''
# Created on 2021/03/18 15:22:14
# @filename: evalFunc.py
# @author: tcxia
'''

import math

class Metric():
    def __init__(self, train, test, GetRecommandation) -> None:
        """评估矩阵

        Args:
            train ([type]): [训练集]
            test ([type]): [测试集]
            GetRecommandation ([type]): [为某个用户获取推荐物品的接口函数]
        """
        self.train = train
        self.test = test
        self.GetRecommandation = GetRecommandation
        self.recs = self.getRec()
    
    # 为test中每个用户进行推荐
    def getRec(self):
        recs = {}
        for user in self.test:
            rank = self.GetRecommandation(user) # 得到排名最高的物品，以及得分
            recs[user] = rank
        return recs

    # 定义精确率指标计算方式
    def precision(self):
        all_data, hit_data = 0, 0
        for user in self.test:
            test_items = set(self.test[user]) # 测试集中每个用户对应的商品集合
            rank = self.recs[user] # 得到推荐的物品
            for item, _ in rank:
                if item in test_items:
                    hit_data += 1
            all_data += len(rank)
        return round(hit_data / all_data * 100, 2)
    
    # 定义召回率指标计算方式
    def recall(self):
        all_data, hit_data = 0, 0
        for user in self.test:
            test_items = set(self.test[user])
            rank = self.recs[user]
            for item, _ in rank:
                if item in test_items:
                    hit_data += 1
            all_data += len(test_items)
        return round(hit_data / all_data * 100, 2)
    
    # 定义覆盖率指标计算方式
    def coverage(self):
        all_item, recom_item = set(), set()
        for user in self.test:
            for item in self.train[user]:
                all_item.add(item)
            rank = self.recs[user]
            for item, _ in rank:
                recom_item.add(item)
        return round(len(recom_item) /  len(all_item) *  100, 2)

    # 计算新颖度指标
    def popularity(self):
        # 计算物品的流行度
        item_pop = {}
        for user in self.train:
            for item in self.train[user]:
                if item not in item_pop:
                    item_pop[item] = 0
                item_pop[item] += 1
        num, pop = 0, 0
        for user in self.test:
            rank = self.recs[user]
            for item, score in rank:
                # 取对数，防止因长尾问题带来的被流行物品所主导
                pop += math.log(1 + item_pop[item])
                num += 1
        return round(pop / num, 6)

    def eval(self):
        metric = {
            'precision': self.precision(),
            'recall': self.recall(),
            'coverage': self.coverage(),
            'popularity': self.popularity()
        }
        print('metric:', metric)
        return metric