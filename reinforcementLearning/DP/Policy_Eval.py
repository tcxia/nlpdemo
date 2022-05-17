# -*- coding: utf-8 -*-
'''
# Created on 2021/04/26 15:04:07
# @filename: Policy_Eval.py
# @author: tcxia
'''

import numpy as np

from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    V = np.zeros(env.nS) # 随机初始化全零的价值函数
    while True:
        delta = 0
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v 
        if delta < theta:
            break
    return np.array(V)


if __name__ == "__main__":
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v = policy_eval(random_policy, env)

    print('Value Function:')
    print(v)

    print('Reshaped Grid Value Function')
    print(v.reshape(env.shape))