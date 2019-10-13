'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents


def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        #agents.SimpleAgent(),
        agents.JukeBot(),
        agents.SimpleAgent()
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]

    print("------")
    print(agent_list)
    print("-------")
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    #n = 100
    winners = [0,0,0]
    n=1
    for i_episode in range(n):
        state = env.reset()
        done = False
        while not done:
            env.render()
            #input()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        print('Episode {} finished'.format(i_episode))
        print(info)
        if info['result'].value == 0:
            winners[info['winners'][0]]+=1
    
    print(winners)
    env.close()


if __name__ == '__main__':
    main()
