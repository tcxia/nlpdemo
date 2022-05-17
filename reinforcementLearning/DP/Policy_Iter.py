# -*- coding: utf-8 -*-
'''
# Created on 2021/04/26 15:23:21
# @filename: Policy_Iter.py
# @author: tcxia
'''

import numpy as np

from lib.envs.gridworld import GridworldEnv
from Policy_Eval import policy_eval


env = GridworldEnv()

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    
    def one_step_lookahead(state, V):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        V = policy_eval_fn(policy, env, discount_factor)
        policy_stable = True

        for s in range(env.nS):
            choose_a = np.argmax(policy[s])

            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)

            if choose_a != best_a:
                policy_stable = False
            
            policy[s] = np.eye(env.nA)[best_a]
        
        if policy_stable:
            return policy, V


if __name__ == "__main__":
    policy, v = policy_improvement(env)
    print('Policy Probability Distribution:')
    print(policy)

    print('Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):')
    print(np.reshape(np.argmax(policy, axis=1), env.shape))