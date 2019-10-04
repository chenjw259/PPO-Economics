from collections import deque
import gym
from gym import spaces
from gym.envs.registration import register
import random
import numpy as np
from .utility import Utility as util
import matplotlib.pyplot as plt

AGENTS = 1

class EconomicsEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}
    max_price = 150

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
        self.n_steps = 128
        self.steps = 0
        self.prev_sales = 0
        self.range = 0.1
        self.split = 0.1
        self.action_array = list(np.arange(-self.range, self.range, self.split))
        self.action_array.append(self.range)
        self.action_space = spaces.Discrete(len(self.action_array))
        self.std_profits = []

    def demand(self):
        demands = []
        agents = list(self.agents)
        key = list(range(len(agents)))
        for agent in agents:
            demand = -(1/0.1) * (agent.price - EconomicsEnv.max_price) 
            
            demands.append(demand)


        demands, key = util.sort_together(demands, key)
        for i in range(len(demands)):
            if demands[i] <= 0:
                demands[i] = 0

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
        
        # Each step is considered to be one quarter
        # A company produces X # of goods, sells them over the quarter

        pre_balance = self.agent.balance

        self._generate_new_prices(action)
        demands = self.demand()

        for agent in self.agents:
            agent.create_products()

        sorted_agents, sorted_demands = util.sort_together(self.agents, demands)
        quarterly_profits = self._sell(sorted_agents, sorted_demands)

        post_balance = self.agent.balance
        reward = self._calculate_reward(pre_balance, post_balance)
        std_profits = np.array(quarterly_profits) / 1e+5
        self.std_profits.append(std_profits)
        if np.argmax(std_profits) == len(std_profits) - 1:
            reward += 1
        else:
            reward -= 1
        
        self.steps += 1

        obs = self.next_observation()

        return obs, reward, self.steps >= self.n_steps, None

    def reset(self):
        for agent in self.agents:
            agent.reset()

        self.observations = deque(maxlen=self.max_len) 
        self.steps = 0
        self.std_profits = []

        return self.next_observation()

    def render(self, mode="human", title="", close=False):
        labels = ["price", "balance", "standardized profits"]
        std_profits = np.array(self.std_profits).transpose()
        prices = [agent.prices for agent in self.agents]
        balances = [agent.balances for agent in self.agents]
        all_ = [prices, balances, std_profits]
        fig, axis = plt.subplots(nrows=len(all_), ncols=1, figsize=(16, 3 * len(all_)))
        fig.suptitle(title, fontsize=16)

        c = 0
        for ax in axis:
            for agent, data in zip(self.agents, all_[c]):
                name = agent.name
                ax.plot(list(range(len(data))), data, label=name)
                ax.set_ylabel(labels[c])
                ax.set_xlabel('day')
                ax.legend(loc='upper left')
            c += 1

        fig.canvas.draw()

        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        del fig

        return data

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
        quarterly_profit = []
        for agent, demand in zip(sorted_agents, sorted_demands):
            available_products = agent.available_products
            if agent.price > EconomicsEnv.max_price:
                assert demand == 0

            amount_to_sell = 0
            sales = False
            if available_products > 0:
                if demand <= available_products:
                    amount_to_sell = demand
                else:
                    amount_to_sell = available_products

                if amount_to_sell > 0:
                    sales = True
                    profit = agent.sell_products(amount_to_sell)
                    quarterly_profit.append(profit)
                    total_sales += amount_to_sell

            if not sales:
                quarterly_profit.append(0)


            if agent == self.agent:
                self.prev_sales = amount_to_sell
        
        return quarterly_profit

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
