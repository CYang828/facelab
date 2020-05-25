import tensorflow as tf


class LandmarkHead(tf.keras.layers.Layer):
    """Landmark Head Block"""

    def __init__(self, num_anchor, weight_decay, name='LandmarkHead', **kwargs):
        super(LandmarkHead, self).__init__(name = name, **kwargs)
        self.num_anchor = num_anchor
        self.conv = tf.keras.layers.Conv2D(filters = num_anchor * 10, kernel_size = 1, strides = 1)

    def call(self, x, **kwargs):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = self.conv(x)

        return tf.reshape(x, [-1, h * w * self.num_anchor, 10])
