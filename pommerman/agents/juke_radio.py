from collections import defaultdict
import queue
import random
import math
from enum import Enum

import numpy as np

from . import BaseAgent
from .. import constants
from .. import utility

import pickle


# "No, I think you can ignore that rule for our class." - Dr. Decker on if our message ints have to be in the range [0,8].
# Whatever you say, Mr. Agent man.
def struct_to_int(struct):
    b = pickle.dumps(struct)
    i = int.from_bytes(b, byteorder='little', signed=False)

    return i

def int_to_struct(i):
    length = math.ceil(math.log(i,2))
    b = i.to_bytes(length, byteorder='little')
    struct = pickle.loads(b)

    return struct

# Need a placeholder for the second message since we only use the first one
PLACEHOLDER = struct_to_int(" 'Do you think that god stays in heaven, for he too fears what he created?' - Steve Buscemi, Spy Kids 2")

# Code for A* pathfinding based off of this article
# https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
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


class JukeRadio(BaseAgent):

    class Desires(Enum):
        FLEE = 0
        KILL = 1
        POWER_UP = 2
        CLEAR_ENV = 3
        HIDE = 4

    class Performatives(Enum):
        INFORM = 0
        REQUEST = 1
        INQUIRE = 2

    class Content(Enum):
        NOTHING = 0
        ACKNOWLEDGED = 1
        STATUS = 2
        

    def __init__(self, *args, **kwargs):
        super(JukeRadio, self).__init__(*args, **kwargs)
        self.action_queue = queue.PriorityQueue()

        self.beliefs = {
            'position': (), 
            'obs': [], 
            'threats': [], 
            'powerups': [], 
            'neighbors':[], 
            'enemies': [], 
            'wood': [], 
            'routes': {},
            'board': np.full((11, 11), constants.Item.Fog.value),
            'bomb_blast_strength': np.full((11,11), 0)
        }
        self.desires = [JukeRadio.Desires.POWER_UP]
        self.intentions = {'go_to': None, 'avoid': [], 'drop_bomb_at': None, 'wait': None}
        self.current_plan = []

    # Find spaces that threaten us
    def find_threatened_spaces(self, obs):
        threatened_spaces = set()
        for i in range(len(self.beliefs['bomb_blast_strength'])):
            for j in range(len(self.beliefs['bomb_blast_strength'])):
                strength = self.beliefs['bomb_blast_strength'][i, j]
                if strength > 0:
                    radius = int(strength)-1
                    for row in range(i-radius,i+radius+1):
                        for col in range(j-radius,j+radius+1):
                            threatened_spaces.add((row,col))


                if self.beliefs['board'][i, j] == constants.Item.Flames.value:
                    threatened_spaces.add((i, j))

        return threatened_spaces
                
    # Get the spaces adjacent to us
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
                
    # Returns back board locations of certain objects
    def find_objects(self,board,item_types):
        found_items = []
        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r,c] in item_types:
                    found_items.append( (r, c) )

        return found_items

    power_ups = [constants.Item.ExtraBomb.value, constants.Item.IncrRange.value, constants.Item.Kick.value]
    passables = power_ups + [constants.Item.Passage.value]

    # Change our behavior and update our worldview based off of the messages we recieved
    def process_messages(self, obs, beliefs):
        message_received = None
        if obs["message"][0] != 0:
            message_received = int_to_struct(obs["message"][0])
        
        def update_if_visible(r, c, v,board):
            return board[r, c] != constants.Item.Fog.value

        board_defaults = {
            'board': update_if_visible,
            'bomb_blast_strength': update_if_visible,
        }

        for b in board_defaults:
            my_board = obs[b]

            their_board = np.array([])
            # Parse the message, if we got one
            if message_received:
                their_board = message_received["inform"]["beliefs"]["obs"][b]


            beliefs[b] = self.update_board(beliefs[b], my_board, their_board, board_defaults[b])

        return beliefs

    #Keeps track of past states of the board so we at least remember some things
    def update_board(self, board, new_board_0, new_board_1, should_update):
        for i in range(len(board)):
            for j in range(len(board)):
                if should_update(i, j, new_board_0[i, j], new_board_0):
                    board[i, j] = new_board_0[i, j]
                elif new_board_1.any() and should_update(i, j, new_board_1[i, j],new_board_1):
                    board[i, j] = new_board_1[i, j]

        return board

    # Change our current beliefs of the world
    def brf(self,beliefs,obs):
        beliefs['obs'] = obs

        beliefs = self.process_messages(obs, beliefs)

        r, c = obs['position']

        beliefs['position'] = (r,c)
        beliefs['neighbors'] = self.get_neighbors(obs,beliefs['position'])
        beliefs['threatened'] = self.find_threatened_spaces(beliefs['obs'])

        def find_accessible(items):
            item_list = self.find_objects(beliefs['board'], items)
            item_paths = {i: astar(beliefs['board'], beliefs['position'], i, JukeRadio.passables + items) for i in item_list}

            valid_items = []
            for item in item_paths:
                if item_paths[item]:
                    valid_items.append(item)

            return valid_items

        # Only list powerups that are within range
        beliefs['powerups'] = find_accessible(JukeRadio.power_ups)

        beliefs['enemies'] = self.find_objects(beliefs['board'], obs['enemies'])
        beliefs['wood'] = find_accessible([constants.Item.Wood.value])
        #print(beliefs["wood"])
        

        return beliefs

    ##TODO: Fix where there's wood left but its impossible to reach and we don't start fleeing.

    # See if our intention makes sense
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
    
    # If the given path contains a threat
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
        if desires[0] == JukeRadio.Desires.FLEE:
            # Find nearest spot that's empty and not in threats
            safe_spots = set(self.find_objects(beliefs['board'], [constants.Item.Passage.value])) - beliefs['threatened']
            paths = {location: astar(beliefs['board'], beliefs['position'], location[0:2], JukeRadio.passables+[constants.Item.Bomb.value]) for location in safe_spots}
            
            # Go to that spot
            if paths:
                nearest_safe = min(paths, key=lambda p: len(paths[p]) if paths[p] else np.Infinity)
                dest = nearest_safe[0:2]
                intentions['go_to'] = dest
                beliefs['routes'][dest] = paths[nearest_safe]  

        # If we want to POWERUP
        elif desires[0] == JukeRadio.Desires.POWER_UP and beliefs['powerups']:
            # Find a path to all the powerups
            paths = {powerup: astar(beliefs['board'], beliefs['position'], powerup[0:2], JukeRadio.passables) for powerup in beliefs['powerups']}
            
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

        # If we want to CLEAR THE ENVIRONMENT
        elif desires[0] == JukeRadio.Desires.CLEAR_ENV and beliefs['wood']:
            # Find the paths to wood on the board
            
            paths = {wood: astar(beliefs['board'], beliefs['position'], wood[0:2], JukeRadio.passables+[constants.Item.Wood.value]) for wood in beliefs['wood']}
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

        elif desires[0] == JukeRadio.Desires.KILL:
            found_enemies = self.find_objects(beliefs['board'], [enemy.value for enemy in beliefs['obs']['enemies']])

            if found_enemies:
                #found_enemy = found_enemies[0]
                paths = {enemy: astar(beliefs['board'], beliefs['position'], enemy, JukeRadio.passables+[constants.Item.Bomb.value, constants.Item.Agent0.value,constants.Item.Agent1.value,constants.Item.Agent2.value,constants.Item.Agent3.value]) for enemy in found_enemies}
                
                if paths:
                    shortest_path = min(paths, key=lambda p: len(paths[p]) if paths[p] and not self.contains_threat(paths[p],beliefs['threatened']) else np.Infinity)

                    path = paths[shortest_path]

                    if path and self.contains_threat(path, beliefs['threatened']):
                        intentions['wait'] = True
                    
                    if path:
                    # We also just drop the bomb at the adjacent space, not the actual wood space
                        path.pop()
                        dest = path[-1]

                        intentions['go_to'] = dest
                        intentions['drop_bomb_at'] = dest
                        beliefs['routes'][dest] = path

            

        # If we desire to HIDE from the enemy
        elif desires[0] == JukeRadio.Desires.HIDE:
            safe_spots = set(self.find_objects(beliefs['board'], [constants.Item.Passage.value])) - beliefs['threatened']
            found_enemies = self.find_objects(beliefs['board'], [enemy.value for enemy in beliefs['obs']['enemies']])

            if found_enemies:
                found_enemy = found_enemies[0]
                paths = {location: astar(beliefs['board'], found_enemy, location[0:2], JukeRadio.passables+[constants.Item.Bomb.value]) for location in safe_spots}
                
                if paths:
                    farthest_path_location = max(paths, key=lambda p: len(paths[p]) if paths[p] and not self.contains_threat(paths[p],beliefs['threatened']) else -np.Infinity)
                    path_to_farthest_location = astar(beliefs['board'], beliefs['position'], farthest_path_location, JukeRadio.passables)
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



    def get_following_pos(self, current_plan, intentions, beliefs):
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

        return next_pos

    # Is our plan OK? (Will it get us killed?)
    def sound(self, current_plan, intentions, beliefs):

        if self.desires[0] == JukeRadio.Desires.KILL:
            return True

        # If our next move in the plan puts us in danger, replan
        if intentions['go_to']:
            dest = intentions['go_to']

            next_pos = self.get_following_pos(current_plan, intentions, beliefs)

            if next_pos in beliefs['threatened']:
                safe_board = beliefs['board'].copy()

                for r in range(len(safe_board)):
                    for c in range(len(safe_board[0])):
                        if (r,c) in beliefs['threatened']:
                            safe_board[r,c] = constants.Item.Rigid.value

                new_path = astar(safe_board, beliefs['position'], dest, JukeRadio.passables)
                beliefs['routes'][dest] = new_path
                return False

        # Also reconsider if we don't actually have a plan
        return len(current_plan) > 0

    # Set our desires based off of our beliefs and intentions
    def options(self, beliefs, intentions):
        # Most important: if we think we are threatened, get out of danger!
        if beliefs['position'] in beliefs['threatened']:
            return [JukeRadio.Desires.FLEE]

        #If there's a powerup we can get to, go get it!
        elif beliefs['powerups']:
            return [JukeRadio.Desires.POWER_UP]

        #If there's still wood to be cleared, clear it!
        elif beliefs['wood']:
            return [JukeRadio.Desires.CLEAR_ENV]

        #We're pacifists, so run away if there's nothing better to do!
        else:
            return [JukeRadio.Desires.HIDE]

   

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
            self.current_plan.append(constants.Action.Stop)

        self.message_to_send = {"inform": {"beliefs": self.beliefs}}
        self.message_to_send = struct_to_int(self.message_to_send)


        return self.current_plan.pop(), self.message_to_send, PLACEHOLDER


