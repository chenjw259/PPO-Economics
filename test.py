from economics import EconomicsEnv
from economics.agents import ConstantAgent
from economics.agents import EnvAgent
import numpy as np
import random
from PPO import Model 
from memory import Memory
from collections import deque


agents = [ConstantAgent("tucker", 100, 10, 25)]
env_agent = EnvAgent("env", 100, 10, 20)
env = EconomicsEnv(agents, env_agent)

m = Model(10, continuous=False)
memory = Memory()
results = []
game_rewards = []
games = 0
all_intrinsic = []
last_100 = deque(maxlen=100)
std = 1

for i in range(int(1e+6)):
    rewards = []
    episode_i_rewards = []

    obs = env.reset()
    done = False
    x = 0
    while not done:
        # gym_env.render()
        obs = np.array(obs)
        # print (obs)
        intrinsic_reward = m.get_intrinsic_reward([obs])

        action, dist = m.get_action([obs])

        obs, reward, done, _ = env.step(action)

        intrinsic_reward = intrinsic_reward / std

        memory.remember(obs, dist.logits[0], action, reward, intrinsic_reward, not done)
        rewards.append(reward)
        episode_i_rewards.append(intrinsic_reward)
        x += 1

    
    all_intrinsic.extend(episode_i_rewards)

    game_rewards.append(sum(rewards))
    last_100.append(sum(rewards))
    games += 1
    print (sum(rewards), sum(episode_i_rewards), i)

    if len(memory) > 1024 * 2:
        results.append([sum(memory.rewards)/games, max(game_rewards), 
                        "mean reward for last 100 episodes: %s" % (sum(last_100)/len(last_100))])
        std = np.std(all_intrinsic)
        print ("stddev of intrinsic: %s" % std)
        games = 0
        with open("results.txt", "w") as f:
            for r in results:
                f.write(str(r) + "\n")
                
        # memory.combine_intrinsic(std)
        m.train(memory, 5, 4)

        del memory
        memory = Memory()
        game_rewards = []

done = False
while not done:
    _, reward, done, _ = env.step(random.choice(list(range(10))))
    print (reward)



