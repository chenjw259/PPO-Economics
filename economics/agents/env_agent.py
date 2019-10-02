from . import BaseAgent
from ..market import EconomicsEnv

class EnvAgent(BaseAgent):

    def __init__(self, name, balance, cost, price):
        super().__init__(name, balance, cost, price)

    def generate_new_price(self, change):
        self.price = self.price + (self.price * change)
        if self.price <= self.cost // 2:
            self.price = self.cost // 2
        if self.price > 2 * EconomicsEnv.max_price: # TODO: Shared value from env
            self.price = 2 * EconomicsEnv.max_price 

    def reset(self):
        return EnvAgent(*self.backup)