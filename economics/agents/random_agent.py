from . import BaseAgent
from ..market import EconomicsEnv
import random

class RandomAgent(BaseAgent):

    def __init__(self, name, balance, cost, price):

        self.range = 0.1
        super().__init__(name, balance, cost, price)

    def generate_new_price(self):
        change = random.uniform(-self.range, self.range)
        self.price = self.price + (self.price * change)
        self.check_price()
        