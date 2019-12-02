'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
from pommerman.agents.memory import Memory

from collections import defaultdict
import queue
import random

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

gamma = 0.99
initial_exploration = 10
batch_size = 32
lr = 0.001


def main():
    print(pommerman.REGISTRY)

    #for epoch in range(2):
    agent_list = [
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent()
        ]

    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    for i_episode in range(1):
        state = env.reset()
        done = False
        
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

if __name__ == '__main__':
    main()
