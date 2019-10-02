import random
from ..market import EconomicsEnv
class BaseAgent():

    def __init__(self, name, balance, cost, price):
        self.name = name 
        self.balance = balance
        self.cost = cost 
        self.price = price 
        self.available_products = 0
        self.backup = [name, balance, cost, price]
        self.prices = [self.price]
        self.balances = [self.balance]

    def __gt__(self, other):
        return self.price > other.price 
    
    def __lt__(self, other):
        return self.price < other.price 

    def __eq__(self, other):
        return self.price == other.price

    def reset(self):
        return self.__init__(*self.backup)

    def create_products(self):
        products_to_create = self.balance // self.cost * 0.8

        self.available_products += products_to_create
        self.balance -= self.cost * products_to_create

    def sell_products(self, count):
        self.available_products -= count 
        self.balance += count * self.price
        self.balances.append(self.balance)

    def generate_new_price(self):
        raise NotImplementedError("generate_new_price not implemented in BaseAgent")

    def check_price(self):
        self._check_min_price()
        self._check_max_price()
        self.prices.append(self.price)

    def _check_min_price(self):
        if self.price <= self.cost // 2:
            self.price = self.cost + 1
    
    def _check_max_price(self):
        if self.price > 50 + EconomicsEnv.max_price: # TODO: Shared value from env
            self.price = 50 + EconomicsEnv.max_price 