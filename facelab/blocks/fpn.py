import facelab.blocks.conv

import tensorflow as tf


class FPN(tf.keras.layers.Layer):
    """Feature Pyramid Network Block"""

    def __init__(self, out_channel, weight_decay, name='FPN', **kwargs):
        super(FPN, self).__init__(name = name, **kwargs)
        activation = 'relu'
        if out_channel <= 64:
            activation = 'lrelu'

        self.output1 = facelab.blocks.conv.ConvBlock(f = out_channel, k = 1, s = 1,
                                                     weight_decay = weight_decay, activation = activation)
        self.output2 = facelab.blocks.conv.ConvBlock(f = out_channel, k = 1, s = 1,
                                                     weight_decay = weight_decay, activation = activation)
        self.output3 = facelab.blocks.conv.ConvBlock(f = out_channel, k = 1, s = 1,
                                                     weight_decay = weight_decay, activation = activation)
        self.merge1 = facelab.blocks.conv.ConvBlock(f = out_channel, k = 3, s = 1,
                                                    weight_decay = weight_decay, activation = activation)
        self.merge2 = facelab.blocks.conv.ConvBlock(f = out_channel, k = 3, s = 1,
                                                    weight_decay = weight_decay, activation = activation)

    def call(self, x, **kwargs):
        output1 = self.output1(x[0])  # [80, 80, out_channel]
        output2 = self.output2(x[1])  # [40, 40, out_channel]
        output3 = self.output3(x[2])  # [20, 20, out_channel]

        up_h, up_w = tf.shape(output2)[1], tf.shape(output2)[2]
        up3 = tf.image.resize(output3, [up_h, up_w], method = 'nearest')
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up_h, up_w = tf.shape(output1)[1], tf.shape(output1)[2]
        up2 = tf.image.resize(output2, [up_h, up_w], method = 'nearest')
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return output1, output2, output3
