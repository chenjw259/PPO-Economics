from . import BaseAgent

class ConstantAgent(BaseAgent):

    def __init__(self, name, balance, cost, price):
        super().__init__(name, balance, cost, price)

    def generate_new_price(self):
        return self.price 

    def reset(self):
        return ConstantAgent(*self.backup)
    