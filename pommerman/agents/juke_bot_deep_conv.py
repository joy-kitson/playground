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
class JukeBotDeepConv(BaseAgent):
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

        super(JukeBotDeepConv, self).__init__(*args, **kwargs)


        #print("JUKE DEEP")
        #self.target_net = QNet(11*11, 6)
        self.target_net = net
        self.epsilon = epsilon
        self.debug = debug
        #print(self.target_net)
        
        if debug:
            print(self.epsilon)

    def act(self, obs, action_space):
        board = torch.tensor(obs['board']).float()
        bombs = torch.tensor(obs['bomb_blast_strength']).float()
        
        state = torch.stack([board,bombs])
        state = state.unsqueeze(0)
        #print(board)
        #print(state)
        
        
        #board = board.flatten()
        #ombs = bombs.flatten()
        
        #flat = np.concatenate((board,bombs),axis=None)
        #print(flat)
        
        if self.debug:
            print(state)

        #flat = torch.tensor([flat]).float()
        
        return self.get_action(state,self.target_net,self.epsilon,action_space)

    def get_action(self, state, target_net, epsilon, action_space):
        if np.random.rand() <= epsilon:
            return action_space.sample()
        else:
            act = target_net.get_action(state)
            return act
