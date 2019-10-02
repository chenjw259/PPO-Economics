from . import BaseAgent
from ..market import EconomicsEnv

class EnvAgent(BaseAgent):

    def __init__(self, name, balance, cost, price):
        super().__init__(name, balance, cost, price)

    def generate_new_price(self, change):
        self.price = self.price + (self.price * change)
        self.check_price()

    def reset(self):
        return EnvAgent(*self.backup)