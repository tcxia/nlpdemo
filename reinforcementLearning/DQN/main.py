# -*- coding: utf-8 -*-
'''
# Created on 2021/04/16 14:07:29
# @filename: main.py
# @author: tcxia
'''


import torch
import torch.nn as nn

import copy
import gym

from models.dqn import DQN



env = gym.make("CartPole-v0")
env = env.unwrapped

NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.9
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100


if __name__ == "__main__":
    dqn = DQN(NUM_STATES, NUM_ACTIONS, MEMORY_CAPACITY, LR, EPISILO, ENV_A_SHAPE)
    episodes = 400
    print("Collecting Experience...")
    reward_list = []

    for i in range(episodes):
        state = env.reset()
        ep_reward = 0
        while True:
            env.render()
            action = dqn.choose_action(state)

            next_state, _, done, info = env.step(action)

            x, _, theta, theta_dot = next_state
            
            reward = dqn.reward_func(env, x, theta)

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn(Q_NETWORK_ITERATION, BATCH_SIZE, GAMMA)
                if done:
                    print("episode: {}, the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                break
            state =  next_state
        r = copy.copy(reward)
        reward_list.append(r)
