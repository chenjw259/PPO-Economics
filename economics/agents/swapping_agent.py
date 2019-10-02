from .base_agent import BaseAgent

class SwappingAgent(BaseAgent):

    def __init__(self, name, balance, cost, price, change=5, steps=10):

        super().__init__(name, balance, cost, price)
        self.swap_backup = [change, steps]
        self.change = change 
        self.steps = steps 
        self.n = 0
        self.plus = True

    def generate_new_price(self):
        self.n += 1
        if self.n % self.steps == 0:
            self.price += self.change if self.plus else -self.change 
            self.plus = not self.plus

    