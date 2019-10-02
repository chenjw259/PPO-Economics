from .base_agent import BaseAgent
import random
class SwappingAgent(BaseAgent):

    def __init__(self, name, balance, cost, price, change=.15, steps=10):

        super().__init__(name, balance, cost, price)
        self.swap_backup = [change, steps]
        self.change = change 
        self.steps = steps 
        self.n = 0
        self.plus = True

    def generate_new_price(self):
        self.n += 1
        price_change = random.uniform(0, self.change) * self.price
        if self.n % self.steps == 0:
            self.price += price_change if self.plus else -price_change
            self.plus = not self.plus

        self.check_price()
        

    