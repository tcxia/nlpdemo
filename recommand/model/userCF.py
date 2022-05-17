# -*- coding: utf-8 -*-
'''
# Created on 2021/03/18 14:56:48
# @filename: userCF.py
# @author: tcxia
'''

import math


def UserCF(train, K, N):
    """[基于用户余弦相似度推荐]

    Args:
        train ([type]): [训练集]
        K ([type]): [设置取TopK相似用户数目]
        N ([type]): [设置取TopN推荐物品数目]

    Returns:
        [getRecommandation]: [推荐接口函数]
    """

    # 计算item -> user的倒排索引
    item_users = {}
    for user in train:
        for item in train[user]:
            if item not in item_users:
                item_users[item] = []
            item_users[item].append(user)
    
    # 计算用户相似度矩阵
    sim = {}
    num = {}
    for item in item_users:
        users = item_users[item]
        for i in range(len(users)):
            u = users[i]
            if u not in num:
                num[u] = 0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for j in range(len(users)):
                if j == i: continue
                v = users[j]
                if v not in sim[u]:
                    sim[u][v] = 0
                sim[u][v] += 1
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(num[u] * num[v])
    
    # 按照相似度排序
    sorted_user_sim = {k: list(sorted(v.items(), key=lambda x:x[1], reverse=True)) for k, v in sim.items()}

    # 获取接口函数
    def getRecommandation(user):
        items = {}
        seen_items = set(train[user])
        for u, _ in sorted_user_sim[user][:K]:
            for item in train[u]:
                # 去掉用户见过的（过滤）
                if item not in seen_items:
                    if item not in items:
                        items[item] = 0
                    items[item] += sim[user][u]
        recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs
    
    return getRecommandation