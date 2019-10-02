from . import BaseAgent
from ..market import EconomicsEnv
import random

class RandomAgent(BaseAgent):

    def __init__(self, name, balance, cost, price):

        self.range = 0.1
        super().__init__(name, balance, cost, price)

    def reset(self):
        return RandomAgent(*self.backup)

    def generate_new_price(self):
        change = random.uniform(-self.range, self.range)
        self.price = self.price + (self.price * change)
        if self.price <= self.cost:
            self.price = self.cost + 1
        if self.price > 2 * EconomicsEnv.max_price: # TODO: Shared value from env
            self.price = 2 * EconomicsEnv.max_price