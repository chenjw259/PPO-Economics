import random
import numpy as np
import tensorflow as tf
from .rnd import RND
from .ppo import PPO
def calc_adv(rewards, value_pred, masks, intrinsic=False):
    vp = list(value_pred)
    vp.append(0)
    gam = 0.99
    lam = 0.95
    gae = 0
    if intrinsic:
        masks = [1] * len(masks)

    returns = np.zeros_like(rewards)
    advantages = list(np.zeros_like(rewards))
    
    for i in range(len(rewards)-1, -1, -1):
        current_reward = rewards[i]
        next_value = vp[i + 1]
        current_value = vp[i]
        mask = masks[i]

        delta = current_reward + gam * next_value * mask - current_value
        gae = delta + gam * lam * mask * gae 
        advantages[i] = gae 
    
    returns = np.array(advantages) + np.array(value_pred)

    mean = sum(advantages) / len(advantages)
    std = np.sqrt(sum([(x - mean) ** 2 for x in advantages]) / len(advantages))
    advantages = (advantages - mean) / (std + 1e-10)

    return returns, advantages

def display(data):
    string = "\n"
    lengths = []
    for d in data:
        s = "    " + d + ": %s\n"  % (data[d])
        lengths.append(len(s))
        string += s
    string = "-" * (max(lengths) + 4) + "\n" + string + "\n"
    string += "-" * (max(lengths) + 4) + "\n"
    print (string)

def shuffle(*args):
    indices = np.arange(len(args[0]))
    np.random.shuffle(indices)
    arrays = []
    for arg in args:  
        arrays.append(np.array(arg)[indices])

    return tuple(arrays)

class Model(tf.Module):

    def __init__(self, output_size, continuous=False, sample_mode="random"):

        self.sample_mode = sample_mode
        self.rnd = RND(32)

        self.ppo = PPO(output_size)
        self.beta = 1e-2
        self.lr = 1e-4
        self.scale_factor = 0.95
        self.opt = tf.keras.optimizers.Adam(lr=self.lr)
        self.intrinsic_std = 1 
        self.running_intrinsic_returns = []


    def __call__(self, x):
        return self.actor(x), self.critic(x)

    def get_action(self, x, return_array=True):
        x = np.array(x).astype(np.float32)
        
        dist = self.ppo.get_distribution(x)
        
        return dist.sample(), dist

    def get_intrinsic_reward(self, x):
        x = np.array(x).astype(np.float32)
        return self.rnd.get_reward(x)[0]

    def train(self, memory, epochs, batch_size):
        states, prev_logits, actions, rewards, i_rewards, masks = memory.get()

        e_rewards = rewards 
        
        ext_pred = list(self.ppo.get_extrinsic_pred(states))
        int_pred = list(self.ppo.get_extrinsic_pred(states))

        e_returns, ext_adv = calc_adv(rewards, ext_pred, masks)
        i_returns, int_adv = calc_adv(i_rewards, int_pred, masks, intrinsic=True)
     
        advantages = 2 * ext_adv + 1 * int_adv

        returns = e_returns + i_returns

        print ("-" * 75)


        for e in range(epochs):
            
            a = [states, prev_logits, actions, rewards, returns, advantages, e_returns, i_returns]
            states, prev_logits, actions, rewards, returns, advantages, e_returns, i_returns = shuffle(*a)

            start = 0
            end = batch_size
            
            sum_loss, sum_p_loss, sum_v_loss, sum_r_loss = 0, 0, 0, 0
            sum_ext, sum_int = 0, 0
            sum_entropy, sum_approx_kl = 0, 0
            sum_norm = 0
            sum_ratio, sum_min_ratio, sum_max_ratio = 0, 0, 0

            count = 0

            its = len(states) // batch_size
            
            for _ in range(its):
                # print (i)
                states_ = states[start:end]
                prev_logits_ = prev_logits[start:end]
                actions_ = actions[start:end]
                rewards_ = rewards[start:end]
                returns_ = returns[start:end]
                advantages_ = advantages[start:end]
                e_returns_ = e_returns[start:end]
                i_returns_ = i_returns[start:end]
                e_rewards_ = e_rewards[start:end]
                i_rewards_ = i_rewards[start:end]
                # i_returns_ = i_returns[start:end]

                ph_labels = [prev_logits_, actions_, advantages_]
                ext_labels = [e_rewards_, e_returns_]
                int_labels = [i_rewards_, i_returns_]

                labels = [ph_labels, ext_labels, int_labels]

                total_loss = self.ppo.fit_batch(states_, labels)
                r_loss = self.rnd.fit_batch(states_)
           
                start += batch_size
                end += batch_size

                sum_loss += total_loss.numpy()
                sum_p_loss += self.ppo.policy_loss.numpy()
                sum_v_loss += self.ppo.ext_loss.numpy() + self.ppo.int_loss.numpy()
                sum_r_loss += r_loss.numpy()
                sum_ext += self.ppo.ext_loss.numpy()
                sum_int += self.ppo.int_loss.numpy()
                sum_ratio += self.ppo.avg_ratio.numpy()
                sum_min_ratio += self.ppo.min_ratio.numpy()
                sum_max_ratio += self.ppo.max_ratio.numpy()
                sum_entropy += self.ppo.entropy.numpy()
                sum_approx_kl += self.ppo.approx_kl.numpy()
                sum_norm += self.ppo.global_grad_norm.numpy()


                count += 1

            
            data = {
                "epoch": e + 1,
                "total_loss": sum_loss/count,
                "p_loss": sum_p_loss/count,
                "v_loss": sum_v_loss/count,
                "r_loss": sum_r_loss/count,
                "ext_loss": sum_ext/count,
                "int_loss": sum_int/count,
                "entropy": sum_entropy/count,
                "max_ratio": sum_max_ratio/count,
                "min_ratio": sum_min_ratio/count,
                "approx_kl": sum_approx_kl/count,
                "global_grad_norm": sum_norm/count
                # "avg_intrinsic": avg_int
            }

            display(data)
            # exit()
        print ("-" * 75)