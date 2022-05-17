# -*- coding: utf-8 -*-
'''
# Created on 2021/03/18 14:58:59
# @filename: dataset.py
# @author: tcxia
'''
import random

class Dataset():
    def __init__(self, filepath) -> None:
        self.data = self.loadData(filepath)
    
    def loadData(self, filepath):
        """协同过滤

        Args:
            filepath ([type]): [数据文件]

        Returns:
            [type]: [description]
        """
        data = []
        for l in open(filepath):
            data.append(tuple(map(int, l.strip().split('::')[:2])))
        return data

    
    def loadData_cold_start(self, fp, ip):
        """冷启动问题

        Args:
            fp ([type]): [数据文件]
            ip ([type]): [物品内容文件]
        """
        data = []
        for l in open(fp):
            data.append(tuple(map(int, l.strip().split('::')[:2])))
        contents = {}
        for l in open(ip, 'rb'):
            l = str(l)[2:-1]
            contents[int(l.strip().split('::')[0])] = l.strip().split('::')[-1].split('|')
        return data, contents

    def splitData(self, M, k, seed=1):
        """切分训练集和测试集

        Args:
            M ([type]): [划分的数目，最后需要取M折的平均]
            k ([type]): [本次是第几次划分，k~[0, M)]
            seed (int, optional): [随机种子数]. Defaults to 1.

        Returns:
            [type]: [train, test]
        """
        train, test = [], []
        random.seed(seed)
        for user, item in self.data:
            if random.randint(0, M-1) == k:
                test.append((user, item))
            else:
                train.append((user, item))
        
        # 将数据转换成user: [item1, item2, ..., itemN]的格式
        def convert_dict(data):
            data_dict = {}
            for user, item in data:
                if user not in data_dict:
                    data_dict[user] = set()
                data_dict[user].add(item)
            data_dict = {k:list(data_dict[k]) for k in data_dict}
            return data_dict
        return convert_dict(train), convert_dict(test), self.content

# if __name__ == "__main__":
#     fp = '/data/nlp_dataset/recommandation/ml-1m/ratings.dat'
#     data = Dataset(filepath=fp)
#     d = data.loadData(fp)
#     print(d)