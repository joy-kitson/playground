'''The base simple agent use to train agents.
This agent is also the benchmark for other agents.
'''
from collections import defaultdict
import queue
import random
import math
from enum import Enum

import numpy as np

from . import BaseAgent
from .. import constants
from .. import utility


#https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return self.position.__hash__()


def astar(maze, start, end, passables):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = set()
    closed_list = set()

    # Add the start node
    open_list.add(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = min(open_list, key=lambda n: n.f)
        open_list -= {current_node}

        # Pop current off open list, add to closed list
        closed_list.add(current_node)
        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] not in passables:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            if child in closed_list:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            flag = False
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    flag = True
            if flag:
                continue

            # Add the child to the open list
            open_list.add(child)


class LeopoldAgent(BaseAgent):
    """This is a baseline agent. After cou can beat it, submit cour agent to
    compete.
    """

    """
    James Jocce

    Avoid path of bombs

    Do random actions (Moving and placing bombs)
    """

    class Desires(Enum):
        FLEE = 0
        KILL = 1
        POWER_UP = 2
        CLEAR_ENV = 3
        HIDE = 4

    def __init__(self, *args, **kwargs):
        super(LeopoldAgent, self).__init__(*args, **kwargs)
        self.action_queue = queue.PriorityQueue()

        self.beliefs = {'position': (), 'obs': [], 'threats': [], 'powerups': [], 'neighbors':[], 'enemies': [], 'wood': [], 'routes': {}}
        self.desires = [LeopoldAgent.Desires.POWER_UP]
        self.intentions = {'go_to': None, 'avoid': [], 'drop_bomb_at': None, 'wait': None}
        self.current_plan = []

    def is_threatened(self, obs, posn):
        r, c = posn

        #TODO: go back and look at distance vs bomb strength

        # find any bombs in the same row/col as this space
        for b_c in range(10):
            if obs['bomb_life'][r, b_c] > 0 or obs['flame_life'][r, b_c] > 0:
                return True
        for b_r in range(10):
            if obs['bomb_life'][b_r, c] or obs['flame_life'][b_r, c] > 0:
                return True

        # check whether or not there's flames here
        if (constants.Item.Flames.value == obs['board'][r,c]) or obs['bomb_life'][r,c] > 0:
            return True

        return False

    def find_threatened_spaces(self, obs):
        threatened = set()
        for r in range(len(obs['board'])):
            for c in range(len(obs['board'])):
                if self.is_threatened(obs, (r,c)):
                    threatened.add((r, c))
        return threatened
                
    def get_neighbors(self, obs,posn):
        neighbors = []
        BOARD_SIZE =10
        r,c = posn

        if r < BOARD_SIZE-1:
            neighbors.append((r+1,c,obs["board"][r+1,c]))
       
        if r > 0:
            neighbors.append((r-1,c,obs["board"][r-1,c]))

        if c < BOARD_SIZE-1:
            neighbors.append((r,c+1,obs["board"][r,c+1]))

        if c > 0:
            neighbors.append((r,c-1,obs["board"][r,c-1]))

        return neighbors
                
    def find_objects(self,obs,item_types):
        found_items = []
        for r in range(len(obs['board'])):
            for c in range(len(obs['board'][0])):
                if obs['board'][r,c] in item_types:
                    found_items.append( (r, c) )

        return found_items

    power_ups = [constants.Item.ExtraBomb.value, constants.Item.IncrRange.value, constants.Item.Kick.value]
    passables = power_ups + [constants.Item.Passage.value]

    def brf(self,beliefs,obs):
        r, c = obs['position']

        beliefs['obs'] = obs
        beliefs['position'] = (r,c)
        beliefs['neighbors'] = self.get_neighbors(obs,beliefs['position'])
        beliefs['threatened'] = self.find_threatened_spaces(obs)

        power_up_list = self.find_objects(obs,LeopoldAgent.power_ups)
        power_up_paths = {power_up: astar(beliefs['obs']['board'], beliefs['position'], power_up, LeopoldAgent.passables) for power_up in power_up_list}

        valid_powerups = []
        for power_up in power_up_paths:
            if power_up_paths[power_up]:
                valid_powerups.append(power_up)

        
        beliefs['powerups'] = valid_powerups

        beliefs['enemies'] = self.find_objects(obs, obs['enemies'])
        beliefs['wood'] = self.find_objects(obs, [constants.Item.Wood.value])

        return beliefs

    ##TODO: Fix where there's wood left but its impossible to reach and we don't start fleeing.

    def reconsider(self, intentions, beliefs):
        dest = intentions['go_to']
        
        if dest and dest in beliefs['threatened']:
            return True

        #if next move is threatned
        if dest and dest == beliefs['position']:
            intentions['go_to'] = None
            return True
        
        bomb_dest = intentions['drop_bomb_at']
        if bomb_dest and beliefs['obs']['bomb_life'][bomb_dest[0], bomb_dest[1]] > 0:
            intentions['drop_bomb_at'] = None
            return True
    
    def contains_threat(self, path, threats):
        for node in path:
            if node in threats:
                return True

    # Figure out what we should intend to do based off the world and out desires
    def intention_filter(self, beliefs, desires, intentions):

        # If we are already at a place we we're intending to go to
        dest = intentions['go_to']
        if dest and dest == beliefs['position']:
            intentions['go_to'] = None

        # If we already dropped a bomb at a place we wanted to drop a bomb at
        bomb_dest = intentions['drop_bomb_at']
        if bomb_dest and beliefs['obs']['bomb_life'][bomb_dest[0], bomb_dest[1]] > 0:
            intentions['drop_bomb_at'] = None

        # If we want to FLEE
        if desires[0] == LeopoldAgent.Desires.FLEE:
            # Find nearest spot that's empty and not in threats
            safe_spots = set(self.find_objects(beliefs['obs'], [constants.Item.Passage.value])) - beliefs['threatened']
            paths = {location: astar(beliefs['obs']['board'], beliefs['position'], location[0:2], LeopoldAgent.passables+[constants.Item.Bomb.value]) for location in safe_spots}
            
            # Go to that spot
            if paths:
                nearest_safe = min(paths, key=lambda p: len(paths[p]) if paths[p] else np.Infinity)
                dest = nearest_safe[0:2]
                intentions['go_to'] = dest
                beliefs['routes'][dest] = paths[nearest_safe]  

        # If we want to POWERUP
        elif desires[0] == LeopoldAgent.Desires.POWER_UP and beliefs['powerups']:
            # Find a path to all the powerups
            paths = {powerup: astar(beliefs['obs']['board'], beliefs['position'], powerup[0:2], LeopoldAgent.passables) for powerup in beliefs['powerups']}
            
            if paths:
                # Take the shortest path
                nearest_powerup = min(paths, key=lambda p: len(paths[p]) if paths[p] and not self.contains_threat(paths[p],beliefs['threatened']) else np.Infinity)
                dest = nearest_powerup[0:2]
                path = paths[nearest_powerup]

                # If the path contatins a threat, wait
                if path and self.contains_threat(path,beliefs['threatened']):
                    intentions['wait'] = True

                intentions['go_to'] = dest
                beliefs['routes'][dest] = paths[nearest_powerup]  

            
        elif desires[0] == LeopoldAgent.Desires.CLEAR_ENV and beliefs['wood']:
            # Find the paths to wood on the board
            
            paths = {wood: astar(beliefs['obs']['board'], beliefs['position'], wood[0:2], LeopoldAgent.passables+[constants.Item.Wood.value]) for wood in beliefs['wood']}
            if paths:

                # Find the closest
                nearest_wood = min(paths, key=lambda p: len(paths[p]) if paths[p] and not self.contains_threat(paths[p],beliefs['threatened']) else np.Infinity) 

                # However, we should check to see if that space is threatened
                path = paths[nearest_wood]
                if path and self.contains_threat(path,beliefs['threatened']):
                    intentions['wait'] = True

                if path:
                    # We also just drop the bomb at the adjacent space, not the actual wood space
                    path.pop()
                    dest = path[-1]

                    intentions['go_to'] = dest
                    intentions['drop_bomb_at'] = dest
                    beliefs['routes'][dest] = path

        # Killing is wrong, and bad. 
        # There should be a new, stronger word for killing like badwrong or badong. 
        # YES, killing is badong!
        # From this moment, I will stand for the opposite of killing, gnodab.
        elif desires[0] == LeopoldAgent.Desires.KILL:
            pass

        # If we desire to HIDE from the enemy
        elif desires[0] == LeopoldAgent.Desires.HIDE:
            safe_spots = set(self.find_objects(beliefs['obs'], [constants.Item.Passage.value])) - beliefs['threatened']
            found_enemy = self.find_objects(beliefs['obs'], [enemy.value for enemy in beliefs['obs']['enemies']])[0]
            paths = {location: astar(beliefs['obs']['board'], found_enemy, location[0:2], LeopoldAgent.passables+[constants.Item.Bomb.value]) for location in safe_spots}
            
            if paths:
                farthest_path_location = max(paths, key=lambda p: len(paths[p]) if paths[p] and not self.contains_threat(paths[p],beliefs['threatened']) else -np.Infinity)
                path_to_farthest_location = astar(beliefs['obs']['board'], beliefs['position'], farthest_path_location, LeopoldAgent.passables)
                dest = farthest_path_location

                if path_to_farthest_location and self.contains_threat(path_to_farthest_location,beliefs['threatened']):
                    intentions['wait'] = True
                
                intentions['go_to'] = dest
                beliefs['routes'][dest] = path_to_farthest_location 

        return intentions

    # Take an A* path of squares and turn it into a list of actions for the agent
    def path_to_actions(self, path):
        actions = []
        for i in range(1,len(path)):
            r_l, c_l = path[i - 1]
            r, c = path[i]

            delta = (r - r_l, c - c_l)
            if delta == (-1, 0):
                actions.append(constants.Action.Up)
            elif delta == (1,0):
                actions.append(constants.Action.Down)
            elif delta == (0,1):
                actions.append(constants.Action.Right)
            elif delta == (0,-1):
                actions.append(constants.Action.Left)

        # Reverse the order of the actions, so we can pop them properly
        return actions[::-1]

    # Make a plan to achieve our intentions
    def plan(self, beliefs, intentions):
        actions = []

        # If we intend to wait, do nothing
        if intentions['wait']:
            intentions['wait'] = None
            return [constants.Action.Stop]

        # If we indend to go somewhere, pathfind
        elif intentions['go_to']:
            r,c = dest = intentions['go_to']
            if beliefs['routes'][dest]:
                actions = self.path_to_actions(beliefs['routes'][dest])
                
                # Drop a bomb if we intend to
                if intentions['drop_bomb_at'] == dest:
                    actions.insert(0,constants.Action.Bomb.value)

                beliefs['routes'][dest] == None
        
        # Else, we just stop
        if actions:
            return actions
        else:
            return [constants.Action.Stop]

    # Is our plan OK? (Will it get us killed?)
    def sound(self, current_plan, intentions, beliefs):

        # If our next move in the plan puts us in danger, replan
        if intentions['go_to']:
            dest = intentions['go_to']
            next_move = current_plan[-1]
            r,c = beliefs['position']
            next_pos = (r,c)

            if next_move == constants.Action.Up:
                next_pos = (r-1,c)
            elif next_move == constants.Action.Down:
                next_pos = (r+1,c)
            elif next_move == constants.Action.Left:
                next_pos = (r,c-1)
            elif next_move == constants.Action.Right:
                next_pos = (r,c+1)

            if next_pos in beliefs['threatened']:
                safe_board = beliefs['obs']['board'].copy()

                for r in range(len(safe_board)):
                    for c in range(len(safe_board[0])):
                        if (r,c) in beliefs['threatened']:
                            safe_board[r,c] = constants.Item.Rigid.value

                new_path = astar(safe_board, beliefs['position'], dest, LeopoldAgent.passables)
                beliefs['routes'][dest] = new_path
                return False

        # Also reconsider if we don't actually have a plan
        return len(current_plan) > 0

    # Set our desires based off of our beliefs and intentions
    def options(self, beliefs, intentions):
        # Most important: if we think we are threatened, get out of danger!
        if beliefs['position'] in beliefs['threatened']:
            return [LeopoldAgent.Desires.FLEE]

        #If there's a powerup we can get to, go get it!
        elif beliefs['powerups']:
            return [LeopoldAgent.Desires.POWER_UP]

        #If there's still wood to be cleared, clear it!
        elif beliefs['wood']:
            return [LeopoldAgent.Desires.CLEAR_ENV]

        #We're pacifists, so run away if there's nothing better to do!
        else:
            return [LeopoldAgent.Desires.HIDE]

    def act(self,obs,action_space):
        # A custom agent using the BDI arch.

        self.beliefs = self.brf(self.beliefs, obs)

        #If we don't have a plan, or we reconsider our current plan, change desires and intentions
        if not self.current_plan or self.reconsider(self.intentions, self.beliefs):
            self.desires = self.options(self.beliefs, self.intentions)
            self.intentions = self.intention_filter(self.beliefs, self.desires, self.intentions)
        
        #If our current plan is not sound, revise current plan
        if not self.current_plan or not self.sound(self.current_plan, self.intentions, self.beliefs):
            self.current_plan = self.plan(self.beliefs, self.intentions)

        debug = False
        if debug:
            print("D:",self.desires)
            print("I:",self.intentions)
            print("Pos:",self.beliefs['position'])
            print("Plan:",self.current_plan)
            print("---------")

        #This is in case something slips through
        #If we don't have anything to do, just do nothing 
        if len(self.current_plan) == 0:
            self.current_plan.append([constants.Action.Stop])

        return self.current_plan.pop()


