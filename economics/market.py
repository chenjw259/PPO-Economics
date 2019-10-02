from collections import deque
import gym
from gym import spaces
from gym.envs.registration import register
import random
import numpy as np
from .utility import Utility as util

AGENTS = 1

class EconomicsEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    max_price = 100

    def __init__(self, agents, env_agent):
        super(EconomicsEnv, self).__init__()

        self.observation_space = spaces.Box(low=0, high=100000, shape=(5, 2*len(agents)))
        self.agents = agents 
        self.base_balance = env_agent.balance
        self.alpha = 0.75
        self.max_len = 5
        self.observations = deque(maxlen=self.max_len) 
        self.agents.append(env_agent)
        self.agent = env_agent
        self.n_steps = 256
        self.steps = 0
        self.max_balance = 5 * self.agent.balance
        self.prev_sales = 0
        self.range = 0.1
        self.split = 0.1
        self.action_array = list(np.arange(-self.range, self.range, self.split))
        self.action_array.append(self.range)
        self.action_space = spaces.Discrete(len(self.action_array))

    def demand(self):
        demands = []
        agents = list(self.agents)
        key = list(range(len(agents)))
        for agent in agents:
            demand = EconomicsEnv.max_price - 0.5 * agent.price 
            demands.append(demand)

        demands, key = util.sort_together(demands, key)

        demands = demands[::-1]
        key = key[::-1]
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

        self._generate_new_prices(action)
        for agent in self.agents:
            agent.create_products()

        sorted_agents, sorted_demands = util.sort_together(self.agents, demands)
        self._sell(sorted_agents, sorted_demands)

        post_balance = self.agent.balance
        reward = self._calculate_reward(pre_balance, post_balance)
        
        self.steps += 1

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

        obs = self.normalize_observation(obs)

        return obs

    def next_observation(self):
        if len(self.observations) == 0:
            for _ in range(self.max_len):
                self.observations.append(self.generate_observation())
        else:
            self.observations.append(self.generate_observation())

        return np.hstack(self.observations).astype(np.float32)

    def normalize_observation(self, observation):
        observation = np.array(observation)
        observation = observation / EconomicsEnv.max_price
        return list(observation)

    def _generate_new_prices(self, action):
        choices = self.action_array
        for agent in self.agents:
            if agent == self.agent:
                self.agent.generate_new_price(choices[action])
            else:
                agent.generate_new_price()

    def _sell(self, sorted_agents, sorted_demands):
        total_sales = 0
        for agent, demand in zip(sorted_agents, sorted_demands):
            available_products = agent.available_products
            amount_to_sell = 0
            if available_products > 0:
                if demand <= available_products:
                    amount_to_sell = demand
                else:
                    amount_to_sell = available_products

                if amount_to_sell > 0:
                    agent.sell_products(amount_to_sell)
                    total_sales += amount_to_sell

            if agent == self.agent:
                self.prev_sales = amount_to_sell
        
        return total_sales

    def _calculate_reward(self, pre_balance, post_balance):
        if post_balance > pre_balance:
            if post_balance > self.base_balance:
                reward = 1
            else:
                reward = 0.5
        elif post_balance == pre_balance:
            reward = -0.5
        else:
            reward = -1

        return reward