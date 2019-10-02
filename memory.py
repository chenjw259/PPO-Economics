import numpy as np
import tensorflow as tf
class Memory():

    def __init__(self):
        self.states = []
        self.logits = []
        self.distributions = []
        self.actions = []
        self.rewards = []
        self.i_rewards = []
        self.masks = []

    def __len__(self):
        return len(self.states)

    def remember(self, s, p, a, r, i, m):
        self.states.append(s)
        self.distributions.append(p)
        self.actions.append(a)
        self.rewards.append(r)
        self.i_rewards.append(i)
        self.masks.append(m)

    def combine_intrinsic(self, std):
        print (self.i_rewards)
        self.i_rewards = np.array(self.i_rewards) / std
        print (list(self.i_rewards))
        self.rewards = np.array(self.rewards) + self.i_rewards

    def get(self, one_hot_action=True):
        if one_hot_action:
            actions = tf.one_hot(self.actions, len(self.distributions[0]))
        else:
            actions = self.actions
        
        return (np.array(self.states).astype(np.float32),
                np.array(self.distributions).astype(np.float32), 
                np.array(actions).astype(np.float32), 
                np.array(self.rewards).astype(np.float32),
                np.array(self.i_rewards).astype(np.float32),
                np.array(self.masks).astype(np.float32))
                
    def clear(self):
        self.states = []
        self.logits = []
        self.distributions = []
        self.actions = []
        self.rewards = []
        self.i_rewards = []
        self.masks = []