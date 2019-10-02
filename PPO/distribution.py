import tensorflow as tf
import numpy as np

class Distribution():

    def __init__(self, logits):
        if tf.is_tensor(logits):
            logits = logits.numpy()
        self.logits = logits
        self.pi = tf.nn.softmax(logits).numpy()
    
    def sample(self):
        u = tf.random.uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        action = tf.argmax(self.logits - tf.math.log(-tf.math.log(u)), axis=-1).numpy()[0]
        return action

    def neglogpac(self, action):

        return tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                       labels=action)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)
        # return -tf.reduce_sum(self.pi * tf.math.log(self.pi))

    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=-1)