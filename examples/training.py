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
    INPUT_SIZE = 11*11*2
    OUTPUT_SIZE = 5

    TRAINING_NET = QNet(INPUT_SIZE,OUTPUT_SIZE)
    NUM_EPOCHS = 10000
    NUM_ROLLOUTS = 1
    EPSILON = 1
    
    device = "cpu"
    
    PLAYER_INDEX = 0
    
    update_target = 30
    
    def update_target_model(online_net, target_net):
    # Target <- Net
        target_net.load_state_dict(online_net.state_dict())
        
    online_net = QNet(INPUT_SIZE, OUTPUT_SIZE)
    target_net = QNet(INPUT_SIZE, OUTPUT_SIZE)
    update_target_model(online_net, target_net)
    
    online_net.train()
    target_net.train()
    
    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    #writer = SummaryWriter('logs')
    print(INPUT_SIZE)

    for epoch in range(NUM_EPOCHS):
        agent_list = [
                #agents.SimpleAgent(),
                agents.JukeBotDeep(train_net = target_net, epsilon=EPSILON, debug=False),
                agents.SimpleAgent(),
                agents.SimpleAgent(),
                agents.SimpleAgent()
            ]
        
        NOODLE_INDEX = 0
        
        env = pommerman.make('PommeFFACompetition-v0', agent_list)

        for num_rollout in range(NUM_ROLLOUTS):

            state = env.reset()
            
            
            memory = Memory(600)
            done = False
            
            steps = 0
            

            # TODO Terminate loop when agent dies
            while not done:
                steps += 1
                if EPSILON == 0:
                    env.render()
                actions = env.act(state)
                #print(actions[0])
                next_state, reward, done, info = env.step(actions)
               
                #print(next_state)
                
                # we'll remember this!
                mask = 0 if done else 1
                
                player_action = actions[PLAYER_INDEX]
                action_one_hot = np.zeros(5)
                action_one_hot[player_action] = 1
                
                #print(state)
                #print(action_one_hot)
                
                #print(state[PLAYER_INDEX])
                
                
                
                
                c_state = state[PLAYER_INDEX]['board']
                n_state = next_state[PLAYER_INDEX]['board']
                
                c_state = torch.tensor(c_state).flatten().float()
                n_state = torch.tensor(n_state).flatten().float()
                
                #print(c_state)
                c_bombs = torch.tensor(state[PLAYER_INDEX]['bomb_blast_strength']).flatten().float()
                n_bombs = torch.tensor(next_state[PLAYER_INDEX]['bomb_blast_strength']).flatten().float()
                
                
                
                c_state = torch.cat([c_state,c_bombs])
                n_state = torch.cat([n_state,n_bombs])
                
                r = reward[PLAYER_INDEX]
                
                if r == -1:
                    r = (500-steps)
                elif r == 1:
                    r = -(500-steps)
                    
                memory.push(c_state, n_state, action_one_hot, r, mask)
                state = next_state

                if steps > initial_exploration:
                    EPSILON -= 0.00005
                    EPSILON = max(EPSILON, 0.0)
                    
                    batch = memory.sample(1)
                    loss = QNet.train_model(online_net, target_net, optimizer, batch)
                    
                    if steps % update_target == 0:
                        update_target_model(online_net, target_net)
                
                
                if reward[PLAYER_INDEX] == -1:
                    break
            #print(steps)
            env.close()

        if epoch%2 == 0:
            
            print("DONE EPOCH:",epoch,"REWARD:",r,"EPS",EPSILON)


if __name__ == '__main__':
    main()
