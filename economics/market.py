from collections import deque
import gym
from gym import spaces
from gym.envs.registration import register
import random
import numpy as np

AGENTS = 1

class EconomicsEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, agents, env_agent):
        super(EconomicsEnv, self).__init__()

        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=100000, shape=(5, 2*len(agents)))
        self.agents = agents 
        self.alpha = 0.75
        self.max_len = 2
        self.observations = deque(maxlen=self.max_len) 
        self.agents.append(env_agent)
        self.agent = env_agent
        self.n_steps = 128
        self.steps = 0
        self.max_balance = 5 * self.agent.balance
        self.prev_sales = 0

    @property
    def available_sales(self):
        sum_sales = 0
        for agent in self.agents:
            a_sum = agent.balance // agent.cost + agent.available_products
            sum_sales += a_sum
        return int(sum_sales * self.alpha)

    def demand(self):
        max_price = 100
        demands = []
        agents = list(self.agents)
        key = list(range(len(agents)))
        for agent in agents:
            demand = max_price - 0.5 * agent.price 
            demands.append(demand)

        key = [x for _,x in sorted(zip(demands,key))][::-1]
        demands = sorted(demands)[::-1]
        out_demands = []

        for d in range(len(demands) - 1):
            out_demands.append(int(demands[d] - demands[d + 1]))
        out_demands.append(int(demands[-1]))


        demands = [None] * len(demands)
        for d, i in zip(out_demands, key):
            demands[i] = d 
        
        return demands

        

    def step(self, action):
        demands = self.demand()
        pre_balance = self.agent.balance
        total_sales = 0
        choices = [-.5, -.4, -.3, -.1, 0, .1, .2, .3, .4, .5]
        self.agent.generate_new_price(choices[action])
        for agent in self.agents:
            agent.create_products()

        sorted_by_price = sorted(self.agents)
        sorted_demands = [x for _,x in sorted(zip(self.agents,demands))]
        for agent, demand in zip(sorted_by_price, sorted_demands):
            possible_sales = agent.available_products
            available_sales = demand
            if available_sales > 0:
                if possible_sales <= available_sales:
                    amount_to_sell = possible_sales
                else:
                    amount_to_sell = possible_sales - available_sales
                if amount_to_sell > 0:
                    agent.sell_products(amount_to_sell)
                    total_sales += amount_to_sell
                    
            if agent == self.agent:
                self.prev_sales = amount_to_sell

        post_balance = self.agent.balance
        if pre_balance < post_balance:
            reward = 1
        elif pre_balance == post_balance:
            reward = -0.5
        else:
            reward = -1
        
        self.steps += 1
        for agent in self.agents:
            if agent.balance > self.max_balance:
                agent.balance = self.max_balance
        obs = self.next_observation()

        return obs, reward, self.steps >= self.n_steps, None

    def reset(self):
        for agent in self.agents:
            agent.reset()

        self.observations = deque(maxlen=self.max_len) 
        self.steps = 0

        return self.next_observation()

    def render(self, mode="human", close=False):
        raise NotImplementedError

    def generate_observation(self):
        obs = [self.prev_sales]
        for agent in self.agents:
            agent_obs = [agent.price]
            obs.extend(agent_obs)

        return obs

    def next_observation(self):
        if len(self.observations) == 0:
            for i in range(self.max_len):
                self.observations.append(self.generate_observation())
        else:
            self.observations.append(self.generate_observation())

        return list(np.hstack(self.observations))