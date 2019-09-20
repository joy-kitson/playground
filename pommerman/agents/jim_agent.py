'''The base simple agent use to train agents.
This agent is also the benchmark for other agents.
'''
from collections import defaultdict
import queue
import random
import math

import numpy as np

from . import BaseAgent
from .. import constants
from .. import utility


class LeopoldAgent(BaseAgent):
    """This is a baseline agent. After cou can beat it, submit cour agent to
    compete.
    """

    """
    James Jocce

    Avoid path of bombs

    Do random actions (Moving and placing bombs)
    """

    def __init__(self, *args, **kwargs):
        super(LeopoldAgent, self).__init__(*args, **kwargs)
        self.action_queue = queue.PriorityQueue()


    def random_action(self, action_space):
        return action_space.sample()

    def find_threats(self, obs, posn):
        r, c = posn

        # find anc bombs in the same row/col as this space
        threats = {(r, b_c) for b_c in range(10) if constants.Item.Bomb.value == obs['board'][r, b_c]}
        threats |= {(b_r, c) for b_r in range(10) if constants.Item.Bomb.value == obs['board'][b_r, c]}

        # check whether or not there's flames here
        if (constants.Item.Flames.value == obs['board'][r,c]):
            threats |= {(r,c)}

        return threats

    def calculate_utility(self, action, obs):
        r, c = obs['position']
        if constants.Action.Left.value == action:
            c -= 1
        elif constants.Action.Right.value == action:
            c += 1
        elif constants.Action.Up.value == action:
            r -= 1
        elif constants.Action.Down.value == action:
            r += 1

        # don't go out of bounds
        if r >= 10 or r < 0 or c >= 10 or c < 0:
            return -np.Infinity

        # don't try and move onto wood/rigid blocks
        if obs['board'][r, c] in [constants.Item.Wood.value, constants.Item.Rigid.value]:
            return -np.Infinity

        threats = self.find_threats(obs, (r,c))
        
        util = 0
        for threat in threats:
            # the closer a threat is, the more dangerous it is
            t_r, t_c = threat
            util -= (10 - abs(r - t_r) - abs(c - t_c))
        return util

    def act(self, obs, action_space):
        acts = [e.value for e in constants.Action]
        utils = [[action, self.calculate_utility(action, obs)] for action in acts]
        
        best_act, best_util = max(utils, key=lambda u: u[1])
        options = [a for a, u in utils if u == best_util]

        return random.sample(options, 1)[0]