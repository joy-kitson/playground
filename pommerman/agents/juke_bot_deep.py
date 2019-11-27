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
class JukeBotDeep(BaseAgent):
    """This is a baseline agent. After you can beat it, submit your agent to
    compete.
    """


    def __init__(self, *args, **kwargs):
        #new_args = args
        net = kwargs["train_net"]
        epsilon = kwargs["epsilon"]
        debug = kwargs["debug"]
        del kwargs["train_net"]
        del kwargs["epsilon"]
        del kwargs["debug"]

        super(JukeBotDeep, self).__init__(*args, **kwargs)


        
        #self.target_net = QNet(11*11, 6)
        self.target_net = net
        self.epsilon = epsilon

        if debug:
            print(self.epsilon)

    def act(self, obs, action_space):
        state = obs['board']
        flat = state.flatten()
        #print(flat)

        flat = torch.tensor([flat]).float()
        
        return self.get_action(flat,self.target_net,self.epsilon,action_space)

    def get_action(self, state, target_net, epsilon, action_space):
        if np.random.rand() <= epsilon:
            return action_space.sample()
        else:
            act = target_net.get_action(state)
            return act
