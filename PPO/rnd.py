# Random network distillation from OpenAI
# Paper: Exploration by Random Network Distillation (Burda et al.) https://arxiv.org/abs/1810.12894

import tensorflow as tf 
import numpy as np
import random
from .distribution import Distribution
from .shared import DenseBlock, EmptyLayer

tf.keras.backend.set_floatx("float32")

NOISE = 1
K = tf.keras.backend


def prediction_network_class(label, prediction):
    x = tf.square(label-prediction)
    x = tf.reduce_sum(x, axis=1)
    return tf.reduce_mean(x), x
             

class RNDNetwork(tf.keras.Model):

    def __init__(self, output_size, freeze=False):
        super(RNDNetwork, self).__init__()
        lr = 0.001
        self.output_size = output_size
        self.freeze = freeze

        self.init = tf.keras.initializers.glorot_normal
        self._create_model()
        self.opt = tf.keras.optimizers.Adam(lr=lr)


    def _create_model(self):

        # activation_fcn = tf.keras.layers.LeakyReLU
        activation_fcn = tf.keras.layers.ReLU

        self.layer1 = DenseBlock(32, self.init, activation_fcn, freeze=True)
        self.layer2 = DenseBlock(32, self.init, activation_fcn, freeze=self.freeze)
        self.layer3 = DenseBlock(32, self.init, activation_fcn, freeze=self.freeze)
        self.layer4 = DenseBlock(32, self.init, activation_fcn, freeze=self.freeze)
        self.layer5 = tf.keras.layers.Dense(self.output_size, activation="linear", trainable=not self.freeze)

    def call(self, x, return_logits=False, return_distribution=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

    def calculate_intrinsic_reward(self, x, label):
        prediction = self(x)
        return prediction_network_class(label, prediction)[1].numpy()
        
    def calculate_loss(self, label, prediction):

        return prediction_network_class(label, prediction)
      
    def fit_batch(self, x, label):
        with tf.GradientTape() as tape:
            out = self(np.array(x).astype(np.float32))

            loss = self.calculate_loss(label, out)

        grads = tape.gradient(loss, self.trainable_variables)

        self.opt.apply_gradients(zip(grads,
                                       self.trainable_variables))

        return loss[0]


class RND():

    def __init__(self, size):
        self.target_network = RNDNetwork(size, freeze=True)
        self.prediction_network = RNDNetwork(size)
        
    def get_reward(self, states):

        label = self.target_network(states)
        return self.prediction_network.calculate_intrinsic_reward(states, label)

    def fit_batch(self, states):

        with tf.GradientTape() as tape:
            out = self.prediction_network(np.array(states).astype(np.float32))
            labels = self.target_network(np.array(states).astype(np.float32))

            loss = self.prediction_network.calculate_loss(labels, out)

        grads = tape.gradient(loss, self.prediction_network.trainable_variables)

        self.prediction_network.opt.apply_gradients(zip(grads,
                                       self.prediction_network.trainable_variables))

        return loss[0]
        
        
