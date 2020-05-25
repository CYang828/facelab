import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):
    """FPN Blcok Conv Block"""

    def __init__(self, f, k, s, weight_decay, activation=None, name='ConvBN', **kwargs):
        super(ConvBlock, self).__init__(name = name, **kwargs)
        self.conv = tf.keras.layers.Conv2D(filters = f, kernel_size = k, strides = s, padding = 'same',
                                           kernel_initializer = tf.keras.initializers.he_normal(),
                                           kernel_regularizer = tf.keras.regularizers.l2(weight_decay),
                                           use_bias = False, name = 'ConvBlock')
        self.bn = tf.keras.layers.BatchNormalization(name = 'BatchNormalization')

        if activation is None:
            self.active_function = tf.identity
        elif activation == 'relu':
            self.active_function = tf.keras.layers.ReLU()
        elif activation == 'lrelu':
            self.active_function = tf.keras.layers.LeakyReLU(0.1)
        else:
            raise NotImplementedError(
                'Activation function type {} is not recognized.'.format(activation))

    def call(self, x, **kwargs):
        return self.active_function(self.bn(self.conv(x)))
