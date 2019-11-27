'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents

from collections import defaultdict
import queue
import random

import numpy as np


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

def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    """
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent()
    ]

    print("------")
    print(agent_list)
    print("-------")
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    """
    INPUT_SIZE = 11*11
    OUTPUT_SIZE = 6

    TRAINING_NET = QNet(INPUT_SIZE,OUTPUT_SIZE)
    NUM_EPOCHS = 3
    NUM_ROLLOUTS = 2
    EPSILON = 0.2

    for epoch in range(NUM_EPOCHS):
        agent_list = [
                agents.JukeBotDeep(train_net = TRAINING_NET, epsilon=EPSILON, debug=True),
                agents.SimpleAgent(),
                agents.SimpleAgent(),
                agents.SimpleAgent()
            ]
        
        env = pommerman.make('PommeFFACompetition-v0', agent_list)

        for num_rollout in range(NUM_ROLLOUTS):

            state = env.reset()
            done = False

            while not done:
                env.render()
                actions = env.act(state)
                state, reward, done, info = env.step(actions)

            env.close()

        print("DONE EPOCH:",epoch)


if __name__ == '__main__':
    main()
