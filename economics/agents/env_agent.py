from . import BaseAgent

class EnvAgent(BaseAgent):

    def __init__(self, name, balance, cost, price):
        super().__init__(name, balance, cost, price)

    def generate_new_price(self, change):
        self.price = self.price + (self.price * change)
        if self.price <= 0.01:
            self.price = 0.01

    def reset(self):
        return EnvAgent(*self.backup)