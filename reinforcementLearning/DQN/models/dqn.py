# -*- coding: utf-8 -*-
'''
# Created on 2021/04/16 14:12:51
# @filename: dqn.py
# @author: tcxia
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class BaseNet(nn.Module):
    def __init__(self, num_states, num_actions) -> None:
        super().__init__()

        self.fc1 = nn.Linear(num_states, 50)
        self.fc1.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(50, 30)
        self.fc2.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(30, num_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

        
class DQN():
    def __init__(self, num_states, num_actions, memory_capacity, lr, episilo, env_a_shape):
        super(DQN, self).__init__()

        self.episilo = episilo
        self.env_a_shape = env_a_shape
        self.num_actions = num_actions
        self.num_states = num_states

        self.eval_net, self.target_net = BaseNet(num_states, num_actions), BaseNet(num_states, num_actions)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory_capacity = memory_capacity
        self.memory = np.zeros((memory_capacity, num_states * 2 + 2))
        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.randn() <= self.episilo:
            action_value = self.eval_net(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        else:
            action = np.random.randint(0, self.num_actions)
            action = action if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        return action
    
    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1
    

    def learn(self, q_netword_iter, batch_size, gamma):
        if self.learn_step_counter % q_netword_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
    
        sample_index = np.random.choice(self.memory_capacity, batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.num_states])
        batch_action = torch.LongTensor(batch_memory[:, self.num_states:self.num_states + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.num_states + 1:self.num_states + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.num_states:])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + gamma * q_next.max(1)[0].view(batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reward_func(self, env, x, theta):
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        return reward

