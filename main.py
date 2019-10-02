import numpy as np
import bsuite
from bsuite import sweep
from bsuite.utils import gym_wrapper # DMEnvFromGym
# from bsuite.experiments.cartpole import analysis as cartpole_analysis
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from PPO import Model 
from memory import Memory
from pyvirtualdisplay import Display
import os
import glob
from collections import deque

files = glob.glob('videos/*')
for f in files:
    os.remove(f)

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# print('All possible values for bsuite_id:', sweep.SWEEP)
# gym_env = gym.make('MountainCar-v0')
# gym_env = gym.make('Acrobot-v1')
# gym_env = gym.make('LunarLander-v2')
gym_env = gym.make('MountainCar-v0')
env = gym_wrapper.DMEnvFromGym(gym_env)
m = Model(env.action_spec().num_values, continuous=False)
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

    timestep = env.reset()
    # print (timestep.observation)
    # exit()

    recorder = VideoRecorder(gym_env, path="videos/%s.mp4" % i)
    recorder.ansi_mode = False
    x = 0
    while not timestep.last():
        recorder.capture_frame()
        # gym_env.render()
        obs = np.array(timestep.observation)
        intrinsic_reward = m.get_intrinsic_reward([obs])
        action, dist = m.get_action([obs])

        # print ([v.name for v in m.ppo.trainable_variables])
        # exit()
        
        x += 1
        # if x % 20 == 0:
        #     print (dist.pi)
        # array = np.array([0, 0, 0])
        # print (action)
        # array[action] = 1
        timestep = env.step(action)

        reward = timestep.reward #+ intrinsic_reward
        intrinsic_reward = intrinsic_reward / std
        # print (dist.logits)
        memory.remember(obs, dist.logits[0], action, reward, intrinsic_reward, not timestep.last())
        rewards.append(timestep.reward)
        episode_i_rewards.append(intrinsic_reward)

    
    all_intrinsic.extend(episode_i_rewards)

    game_rewards.append(sum(rewards))
    last_100.append(sum(rewards))
    games += 1
    print (sum(rewards), sum(episode_i_rewards), i)
    recorder.close()
    del recorder
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

    


    

    
