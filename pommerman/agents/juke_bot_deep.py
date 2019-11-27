'''The base simple agent use to train agents.
This agent is also the benchmark for other agents.
'''
from collections import defaultdict
import queue
import random

import numpy as np

from . import BaseAgent
from .. import constants
from .. import utility

import torch
import torch.nn as nn
import torch.nn.functional as F

gamma = 0.99

class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        qvalue = self.fc2(x)
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).float()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)

        pred = online_net(states).squeeze(1)
        next_pred = target_net(next_states).squeeze(1)

        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + masks * gamma * next_pred.max(1)[0]

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        qvalue = self.forward(input)
        #print(qvalue)
        _, action = torch.max(qvalue, 1)
        return action.numpy()[0]


class JukeBotDeep(BaseAgent):
    """This is a baseline agent. After you can beat it, submit your agent to
    compete.
    """

    def __init__(self, *args, **kwargs):
        super(JukeBotDeep, self).__init__(*args, **kwargs)

        self.target_net = QNet(11*11, 6)

    def act(self, obs, action_space):
        state = obs['board']
        flat = state.flatten()
        #print(flat)
        eps = 0
        flat = torch.tensor([flat]).float()
        
        return self.get_action(flat,self.target_net,eps,action_space)

    def get_action(self, state, target_net, epsilon, action_space):
        if np.random.rand() <= epsilon:
            return action_space.sample()
        else:
            act = target_net.get_action(state)
            return act
