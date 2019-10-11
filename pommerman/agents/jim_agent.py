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

"""


self.beliefs
---------
[neighbors, threats, powerup locations]

self.desires
-------------
[flee, kill, get power ups, clear env]

self.intentions
-------------

go_to(x,y)
avoid(x,y)
drop_bomb_at(x,y)


flee - go as far away from enemy as possible
clear_env - if we have nothing better to do
kill - hunt down enemy (do this if we are more powered up than the enemy)
get power ups - hunt for powerups if close enough and not obstructed (to a point) by wood

Reconsidering
----
if we are threatned by a bomb, flee from bomb
if we think we are more powered up than the other player, hunt them down
if the other player is really close to us and is more powerful than us, flee


"""


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
        CHILL = 4
        HIDE = 5

    def __init__(self, *args, **kwargs):
        super(LeopoldAgent, self).__init__(*args, **kwargs)
        self.action_queue = queue.PriorityQueue()

        self.beliefs = {'position': (), 'obs': [], 'threats': [], 'powerups': [], 'neighbors':[], 'enemies': [], 'wood': [], 'routes': {}}
        self.desires = [LeopoldAgent.Desires.POWER_UP]
        self.intentions = {'go_to': None, 'avoid': [], 'drop_bomb_at': None, 'wait': None}
        self.current_plan = []


    def random_action(self, action_space):
        return action_space.sample()

    
    def find_threats(self, obs, posn):
        r, c = posn

        # find any bombs in the same row/col as this space
        threats = {(r, b_c) for b_c in range(10) if obs['bomb_life'][r,c] > 0}
        threats |= {(b_r, c) for b_r in range(10) if obs['bomb_life'][r,c] > 0}

        # check whether or not there's flames here
        if (constants.Item.Flames.value == obs['board'][r,c]) or obs['bomb_life'][r,c] > 0:
            threats |= {(r,c)}

        return threats

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
        #for i in range(0,2):
            #for j in range(0,2):
        return neighbors
                



    def calculate_utility(self, action, obs):
        util = 0
        
        r, c = obs['position']
        if constants.Action.Left.value == action:
            c -= 1
        elif constants.Action.Right.value == action:
            c += 1
        elif constants.Action.Up.value == action:
            r -= 1
        elif constants.Action.Down.value == action:
            r += 1
        elif constants.Action.Bomb.value == action:
            if 0 == obs['ammo'] or constants.Item.Bomb.value == obs["board"][r, c]:
                return -np.Infinity
            
            neighbors = self.get_neighbors(obs,r,c)

            if constants.Item.Wood.value in neighbors or constants.Item.Agent1.value in neighbors:
                util += 1
            else:
                util -= 1

            #print(neighbors)
            #If there is distructable in neighboring tile



        # don't go out of bounds
        if r >= 10 or r < 0 or c >= 10 or c < 0:
            return -np.Infinity

        # don't try and move onto wood/rigid blocks
        if obs['board'][r, c] in [constants.Item.Wood.value, constants.Item.Rigid.value]:
            return -np.Infinity

        threats = self.find_threats(obs, (r,c))
        
        for threat in threats:
            # the closer a threat is, the more dangerous it is
            t_r, t_c = threat
            util -= (10 - abs(r - t_r) - abs(c - t_c))
        return util

    def act_old(self, obs, action_space):
        acts = [e.value for e in constants.Action]
        #acts = range(0,5)
        

        utils = [[action, self.calculate_utility(action, obs)] for action in acts]
        
        best_act, best_util = max(utils, key=lambda u: u[1])
        options = [a for a, u in utils if u == best_util]

        return random.sample(options, 1)[0]

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
        #beliefs['threats'] = {(r_n,c_n): self.find_threats(obs,(r_n,c_n)) for r_n,c_n,t_n in beliefs['neighbors'] + [(r,c,-1)]}
        beliefs['threatened'] = self.find_threatened_spaces(obs)

        power_up_list = self.find_objects(obs,LeopoldAgent.power_ups)
        power_up_paths = {power_up: astar(beliefs['obs']['board'], beliefs['position'], power_up, LeopoldAgent.passables) for power_up in power_up_list}
        #valid_powerups = min(paths, key=lambda p: len(paths[p]) if paths[p] and not self.contains_threat(paths[p],beliefs['threatened']) else np.Infinity) 
        #power_ups = [ for x in power_ups]
        valid_powerups = []
        for power_up in power_up_paths:
            if power_up_paths[power_up]:
                valid_powerups.append(power_up)

        
        beliefs['powerups'] = valid_powerups



        beliefs['enemies'] = self.find_objects(obs, obs['enemies'])
        beliefs['wood'] = self.find_objects(obs, [constants.Item.Wood.value])

        return beliefs

    def reconsider(self, intentions, beliefs):
        #print("RECONSIDERING")
        dest = intentions['go_to']
        
        if dest and dest in beliefs['threatened']:
            #print("RUN AWAAAAY")
            return True

        #if next move is threatned
        
        if dest and dest == beliefs['position']:
            #print("Reconsider")
            intentions['go_to'] = None
            return True
        
        bomb_dest = intentions['drop_bomb_at']
        #bomb_locs = [(r,c) for r,c,s in beliefs['obs']['bombs']]
        #if bomb_dest and bomb_dest in beliefs['obs']['bombs']:
        if bomb_dest and beliefs['obs']['bomb_life'][bomb_dest[0], bomb_dest[1]] > 0:
            intentions['drop_bomb_at'] = None
            return True
    


    def contains_threat(self, path, threats):
        for node in path:
            if node in threats:
                return True

    def intention_filter(self, beliefs, desires, intentions):
        dest = intentions['go_to']
        if dest and dest == beliefs['position']:
            #print("HERE")
            intentions['go_to'] = None

        bomb_dest = intentions['drop_bomb_at']
        #bomb_locs = [(r,c) for r,c,s in beliefs['obs']['bombs']]
        #if bomb_dest and bomb_dest in beliefs['obs']['bombs']:
        if bomb_dest and beliefs['obs']['bomb_life'][bomb_dest[0], bomb_dest[1]] > 0:
            intentions['drop_bomb_at'] = None

        if desires[0] == LeopoldAgent.Desires.FLEE:
            #print(beliefs['threatened'])

            #Find nearest spot that's empty and not in threats
            safe_spots = set(self.find_objects(beliefs['obs'], [constants.Item.Passage.value])) - beliefs['threatened']
            #print(safe_spots)
            paths = {location: astar(beliefs['obs']['board'], beliefs['position'], location[0:2], LeopoldAgent.passables+[constants.Item.Bomb.value]) for location in safe_spots}
            nearest_safe = min(paths, key=lambda p: len(paths[p]) if paths[p] else np.Infinity)

            #print(safe_spots)

            dest = nearest_safe[0:2]
            #print("FLEEING TO:",nearest_safe)

            intentions['go_to'] = dest
            beliefs['routes'][dest] = paths[nearest_safe]  

            path = paths[nearest_safe]

            pass
        elif desires[0] == LeopoldAgent.Desires.POWER_UP and beliefs['powerups']:
            paths = {powerup: astar(beliefs['obs']['board'], beliefs['position'], powerup[0:2], LeopoldAgent.passables) for powerup in beliefs['powerups']}
            nearest_powerup = min(paths, key=lambda p: len(paths[p]) if paths[p] and not self.contains_threat(paths[p],beliefs['threatened']) else np.Infinity)

            #print("POWERING TO:",nearest_powerup)
            dest = nearest_powerup[0:2]

            path = paths[nearest_powerup]
            if path and self.contains_threat(path,beliefs['threatened']):
                intentions['wait'] = True

            intentions['go_to'] = dest
            beliefs['routes'][dest] = paths[nearest_powerup]  

            path = paths[nearest_powerup]
            
        elif desires[0] == LeopoldAgent.Desires.CLEAR_ENV and beliefs['wood']:
            #print(beliefs['threatened'])
            paths = {wood: astar(beliefs['obs']['board'], beliefs['position'], wood[0:2], LeopoldAgent.passables+[constants.Item.Wood.value]) for wood in beliefs['wood']}
            nearest_wood = min(paths, key=lambda p: len(paths[p]) if paths[p] and not self.contains_threat(paths[p],beliefs['threatened']) else np.Infinity) 


            #We don't actually go to the wood, but rather an adjacent space
            path = paths[nearest_wood]
            if path and self.contains_threat(path,beliefs['threatened']):
                intentions['wait'] = True

            if path:
                #print(path)
                path.pop()
                dest = path[-1]
                #print("WANT TO DROP BOMB AT",dest)
                #However, we should check to see if that space is threatened

                intentions['go_to'] = dest
                intentions['drop_bomb_at'] = dest
                beliefs['routes'][dest] = path

        elif desires[0] == LeopoldAgent.Desires.KILL:
            pass
        elif desires[0] == LeopoldAgent.Desires.CHILL:
            intentions = {'go_to': None, 'avoid': [], 'drop_bomb_at': None, 'wait': None}
        elif desires[0] == LeopoldAgent.Desires.HIDE:
            safe_spots = set(self.find_objects(beliefs['obs'], [constants.Item.Passage.value])) - beliefs['threatened']

            found_enemy = self.find_objects(beliefs['obs'], [enemy.value for enemy in beliefs['obs']['enemies']])[0]

            paths = {location: astar(beliefs['obs']['board'], found_enemy, location[0:2], LeopoldAgent.passables+[constants.Item.Bomb.value]) for location in safe_spots}
            farthest_path_location = max(paths, key=lambda p: len(paths[p]) if paths[p] and not self.contains_threat(paths[p],beliefs['threatened']) else -np.Infinity)

            path_to_farthest_location = astar(beliefs['obs']['board'], beliefs['position'], farthest_path_location, LeopoldAgent.passables)
            print(farthest_path_location)
            print(path_to_farthest_location)

            dest = farthest_path_location

            if path_to_farthest_location and self.contains_threat(path_to_farthest_location,beliefs['threatened']):
                intentions['wait'] = True
            
            intentions['go_to'] = dest
            beliefs['routes'][dest] = path_to_farthest_location 

        
        
        return intentions

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

        # reverse the order of the actions, so we can pop them properly
        return actions[::-1]

    def plan(self, beliefs, intentions):
        actions = []

        if intentions['wait']:
            #print("JUST CHILLIN")
            intentions['wait'] = None
            return [constants.Action.Stop]
        elif intentions['go_to']:
            r,c = dest = intentions['go_to']
            if beliefs['routes'][dest]:
                actions = self.path_to_actions(beliefs['routes'][dest])
                
                if intentions['drop_bomb_at'] == dest:
                    actions.insert(0,constants.Action.Bomb.value)

                beliefs['routes'][dest] == None
        
        if actions:
            return actions
        else:
            return [constants.Action.Stop]

    def sound(self, current_plan, intentions, beliefs):

        #check if next move will put us in danger
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
                #print("BAD PLAN")
                safe_board = beliefs['obs']['board'].copy()

                for r in range(len(safe_board)):
                    for c in range(len(safe_board[0])):
                        if (r,c) in beliefs['threatened']:
                            safe_board[r,c] = constants.Item.Rigid.value

                new_path = astar(safe_board, beliefs['position'], dest, LeopoldAgent.passables)
                beliefs['routes'][dest] = new_path
                return False



        return len(current_plan) > 0

    def options(self, beliefs, intentions):
        #print("CHANGING DESIRE")
        #print(beliefs['threatened'])
        if beliefs['position'] in beliefs['threatened']:
            #print("IM IN DANGER")
            return [LeopoldAgent.Desires.FLEE]
        #If there are powerpus that we can get to
        if beliefs['powerups']:
            #print("I HAVE THE POWER")
            return [LeopoldAgent.Desires.POWER_UP]
        elif beliefs['wood']:
            #print("IM GONNA WRECK IT")
            return [LeopoldAgent.Desires.CLEAR_ENV]
        else:
            print("Hiding")
            return [LeopoldAgent.Desires.HIDE]
        """
        elif beliefs['powerups']:
            return [LeopoldAgent.Desires.POWER_UP]
        elif beliefs['wood']:
            return [LeopoldAgent.Desires.CLEAR_ENV]
        else:
            return [LeopoldAgent.Desires.KILL]
        """
        #if beliefs['powerups']:
        #    return [LeopoldAgent.Desires.POWER_UP]
        #else:
        #    return [LeopoldAgent.Desires.CHILL]

        #return [LeopoldAgent.Desires.CLEAR_ENV]

    def act(self,obs,action_space):
        #print(obs)
        #print("BEFORE PLAN:", self.current_plan)
        self.beliefs = self.brf(self.beliefs, obs)
        if not self.current_plan or self.reconsider(self.intentions, self.beliefs):
            self.desires = self.options(self.beliefs, self.intentions)
            self.intentions = self.intention_filter(self.beliefs, self.desires, self.intentions)
        
        if not self.current_plan or not self.sound(self.current_plan, self.intentions, self.beliefs):
            self.current_plan = self.plan(self.beliefs, self.intentions)

        #print("PLAN:",self.current_plan,"DESIRES:",self.desires)
        debug = False
        if debug:
            #print(obs)
            print("D:",self.desires)
            print("I:",self.intentions)
            print("Pos:",self.beliefs['position'])
            print("Plan:",self.current_plan)
            print("---------")
        if len(self.current_plan) == 0:
            #print("NO PLAN")
            self.current_plan.append([constants.Action.Stop])
        return self.current_plan.pop()


