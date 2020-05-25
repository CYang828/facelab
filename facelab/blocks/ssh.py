import tensorflow as tf

import facelab.blocks.conv


class SSH(tf.keras.layers.Layer):
    """Single Stage Headless Block"""

    def __init__(self, out_channel, weight_decay, name='SSH', **kwargs):
        super(SSH, self).__init__(name = name, **kwargs)
        assert out_channel % 4 == 0
        act = 'relu'
        if out_channel <= 64:
            act = 'lrelu'

        self.conv_3x3 = facelab.blocks.conv.ConvBlock(f = out_channel // 2, k = 3, s = 1,
                                                      weight_decay = weight_decay, activation = None)

        self.conv_5x5_1 = facelab.blocks.conv.ConvBlock(f = out_channel // 4, k = 3, s = 1,
                                                        weight_decay = weight_decay, activation = act)
        self.conv_5x5_2 = facelab.blocks.conv.ConvBlock(f = out_channel // 4, k = 3, s = 1,
                                                        weight_decay = weight_decay, activation = None)

        self.conv_7x7_2 = facelab.blocks.conv.ConvBlock(f = out_channel // 4, k = 3, s = 1,
                                                        weight_decay = weight_decay, activation = act)
        self.conv_7x7_3 = facelab.blocks.conv.ConvBlock(f = out_channel // 4, k = 3, s = 1,
                                                        weight_decay = weight_decay, activation = None)

        self.relu = tf.keras.layers.ReLU()

    def call(self, x, **kwargs):
        conv_3x3 = self.conv_3x3(x)

        conv_5x5_1 = self.conv_5x5_1(x)
        conv_5x5 = self.conv_5x5_2(conv_5x5_1)

        conv_7x7_2 = self.conv_7x7_2(conv_5x5_1)
        conv_7x7 = self.conv_7x7_3(conv_7x7_2)

        output = tf.concat([conv_3x3, conv_5x5, conv_7x7], axis = 3)
        output = self.relu(output)

        return output
