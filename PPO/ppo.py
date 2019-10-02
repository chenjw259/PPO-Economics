import tensorflow as tf
import numpy as np
from .shared import DenseBlock, DenseBatchNormBlock
from .distribution import Distribution


class PPO(tf.Module):

    def __init__(self, output_size):

        self.output_size = output_size
        self.init = tf.keras.initializers.glorot_normal
        self._create_model()
        self.clip_val = 0.2
        self.beta = 1e-2
        # self.opt = tf.keras.optimizers.Adam(lr=2.5e-4)
        self.opt = tf.keras.optimizers.Adam(lr=1e-4)
        self.grad_clip = 1

    def _create_model(self):

        activation_fcn = tf.keras.layers.LeakyReLU
        Block = DenseBlock
        reg = None

        self.layer1 = Block(128, self.init, activation_fcn, regularization=reg, name="shared_block_1")
        self.layer2 = Block(128, self.init, activation_fcn, regularization=reg, name="shared_block_2")
        self.layer3 = Block(128, self.init, activation_fcn, regularization=reg, name="shared_block_3")
        
        # policy_head  
        self.ph_1 = Block(128, self.init, activation_fcn, regularization=reg, name="ph_block_1")
        self.ph_2 = tf.keras.layers.Dense(self.output_size, name="ph_dense_out")
        self.ph_out = tf.keras.layers.Activation("softmax")

        # extrinsic head      
        self.ext_1 = Block(128, self.init, activation_fcn, regularization=reg, name="ext_block_1")
        self.ext_2 = Block(128, self.init, activation_fcn, regularization=reg, name="ext_block_1")
        self.ext_3 = tf.keras.layers.Dense(1, name="ext_dense_out")

        # intrinsic head
        self.int_1 = Block(128, self.init, activation_fcn, regularization=reg, name="int_block_1")
        self.int_2 = Block(128, self.init, activation_fcn, regularization=reg, name="int_block_1")
        self.int_3 = tf.keras.layers.Dense(1, name="int_dense_out")

    def call_ph(self, shared, return_logits=False):
        
        ph_logits = self.ph_1(shared)
        ph_logits = self.ph_2(ph_logits)
        if return_logits:
            return ph_logits
        else:
            ph_out = self.ph_out(ph_logits)
            return ph_out

    def call_ext(self, shared):
        ext_out = self.ext_3(self.ext_2(self.ext_1(shared)))
        return ext_out

    def call_int(self, shared):
        int_out = self.int_3(self.int_2(self.int_1(shared)))
        return int_out

    def call_shared(self, x):
        shared = self.layer3(self.layer2(self.layer1(x)))
        return shared

    def __call__(self, x, return_logits=False):

        shared = self.call_shared(x)
        ph_out = self.call_ph(shared, return_logits=return_logits)
        ext_out = self.call_ext(shared)
        int_out = self.call_int(shared)
        
        return ph_out, ext_out, int_out

    def get_distribution(self, state):
        shared = self.call_shared(state)
        ph_out = self.call_ph(shared, return_logits=True)
        return Distribution(ph_out)

    def get_extrinsic_pred(self, state):
        shared = self.call_shared(state)
        ext_out = self.call_ext(shared)
        return tf.squeeze(ext_out)
  
    def get_intrinsic_pred(self, state):
        shared = self.call_shared(state)
        int_out = self.call_int(shared)
        return tf.squeeze(int_out)

    def _value_loss(self, labels, predictions):
        predictions = tf.squeeze(predictions)
        rewards = labels[0]
        returns = labels[1]
        cliprange = 0.2
        # x = tf.keras.losses.MSE(labels, predictions)

        clipped = rewards + tf.clip_by_value(predictions - rewards, -cliprange, cliprange)
        loss1 = tf.square(predictions - returns)
        loss2 = tf.square(clipped - returns)
        x = 0.5 * tf.reduce_mean(tf.maximum(loss1, loss2))


        if tf.math.is_nan(x):
            for v in self.trainable_variables:
                if v.name == "ext_dense_out":
                    print (v)
            print (predictions)
            raise ValueError("Value loss is nan")

        return x

    def categroical_loss(self, labels, predictions):

        prev_logits = labels[0]
        actions = labels[1]
        advantages = labels[2]
        logits = predictions

        neglogpac = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=actions)

        old_neglogpac = tf.nn.softmax_cross_entropy_with_logits(logits=prev_logits,
                                                                labels=actions)

        ratio = tf.exp(old_neglogpac - neglogpac)
        clipped_ratio = tf.clip_by_value(ratio, 1-self.clip_val, 1+self.clip_val)

        p1 = ratio * -advantages
        p2 = clipped_ratio * -advantages

        p_loss = tf.reduce_mean(tf.maximum(p1, p2))

        entropy = 0
        approx_kl = 0

        dist = Distribution(predictions)
        entropy = tf.reduce_mean(dist.entropy())
        # print (dist.entropy())

        loss = p_loss - self.beta * entropy 

        if tf.math.is_nan(loss):
            print (neglogpac)
            print (old_neglogpac)
            print (ratio)
            print (advantages)
            for v in self.trainable_variables:
                if v.name == "ph_dense_out":
                    print (v)
            raise ValueError("Policy loss is nan")

        self.p_loss = p_loss
        self.entropy = entropy
        self.approx_kl = .5 * tf.reduce_mean(tf.square(neglogpac - old_neglogpac))
        self.avg_ratio = tf.reduce_mean(ratio)
        self.min_ratio = tf.reduce_min(ratio)
        self.max_ratio = tf.reduce_max(ratio)

        return loss

    def calculate_loss(self, predictions, labels):

        self.policy_loss = self.categroical_loss(labels[0], predictions[0])
        self.ext_loss = self._value_loss(labels[1], predictions[1])
        self.int_loss = self._value_loss(labels[2], predictions[2])

        return self.policy_loss + 0.5 * (self.ext_loss + self.int_loss)

    def fit_batch(self, x, labels):

        with tf.GradientTape() as tape:
            pred_with_logits = self(np.array(x).astype(np.float32), return_logits=True)

            loss = self.calculate_loss(pred_with_logits, labels)

        grads = tape.gradient(loss, self.trainable_variables)
        
        grads, norm = tf.clip_by_global_norm(grads, 0.5) 

        # grads = [tf.clip_by_value(grad, -100., 100.) for grad in grads]
        # _, norm = tf.clip_by_global_norm(grads, 0.5) 
        
        self.global_grad_norm = norm  

        self.opt.apply_gradients(zip(grads,
                                       self.trainable_variables))

        return loss

        