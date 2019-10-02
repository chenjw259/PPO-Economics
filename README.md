## PPO Economics (WIP)
 Can the Proximal Policy Optimization created by OpenAI (Shulman et al.), in conjunction with Random Network Distillation (also created by OpenAI (Burda et al.)) be used to solve an Economics based enviroment with a linear demand curve?
 
## Properties of Environment (WIP)
* Objective: Maximize Profit
* State: A vector containing the sales of the agent, and the prices of the agent along with all other agents in the environment, for the last N steps
* Action: A vector containing values from -0.5 to 0.5, split into denominations of 0.1 (-0.5, -0.4, etc)
* Reward: 
    * If balance after sales is more than the original, base balance: +1
    * If balance after sales is less than original, but more than before sales: +0.5
    * If balance after sales is equal to the balance before sales: -0.5
    * If balance after sales is less than the balance before sales: -1
* Demand is determined by a linear demand function (Qd = a - bP), where "a" is considered to maximum price (althought in true economics, a is the value of non price determinants), "b" is the coefficient of demand, where a b in the range of (0, 1) is elastic demand (large changes in price cause large changes in demand), and a b in the range of (1, inf) is inelastic demand (large changes in price cause minimal changes in demand).

## TODO (No particular order)
* Complete basic test with very simple model (Constant Agents, no overhead costs)
* Implement overhead costs
* Allow for changing of cost, at a fee
* Flesh out all classes and subclasses (different types of agents, different policies, etc)
* Rotating demand (not all consumers buy each days)
