import tensorflow as tf 

class DenseBlock(tf.Module):

    def __init__(self, units, init, activation, freeze=False, regularization=None, name=""):

        self.units = units 
        self.init = init 
        self.activation = activation
        self.freeze = freeze
        self._name = name
        if regularization != None:
            self.regularization = tf.keras.regularizers.l2(0.01)
        else:
            self.regularization = None

        self.layer1 = tf.keras.layers.Dense(self.units, 
                                        kernel_initializer=self.init(), 
                                        kernel_regularizer=self.regularization,
                                        trainable=not self.freeze,
                                        name=self._name + "_dense") 
        self.layer2 = self.activation(name=self._name + "_act")

        super(DenseBlock, self).__init__()
        

        # super(DenseBlock, self).build(input_shape)

    def __call__(self, x):

        x = self.layer1(x)
        return self.layer2(x)

class DenseBatchNormBlock(tf.keras.layers.Layer):

    def __init__(self, units, init, activation, freeze=False, regularization=None):

        self.units = units 
        self.init = init 
        self.activation = activation
        self.freeze = freeze
        if regularization != None:
            self.regularization = tf.keras.regularizers.l2(regularization)
        else:
            self.regularization = None

        super(DenseBatchNormBlock, self).__init__()

    def build(self, input_shape):

        self.layer1 = tf.keras.layers.Dense(self.units,
                                            kernel_initializer=self.init(), 
                                            kernel_regularizer=self.regularization,
                                            trainable=not self.freeze) 
        self.layer2 = tf.keras.layers.BatchNormalization()
        self.layer3 = self.activation()


        self.layer1.build(input_shape)
        self.layer2.build(input_shape)
        self.layer3.build(input_shape)

        super(DenseBatchNormBlock, self).build(input_shape)

    def call(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


class EmptyLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        
        super(EmptyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        
        super(EmptyLayer, self).build(input_shape)

    def call(self, x):
        
        return x

    def compute_output_shape(self, input_shape):
        
        return input_shape