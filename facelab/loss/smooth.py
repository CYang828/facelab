import tensorflow as tf


class SmothL1Loss(tf.keras.losses.Loss):
    def __init__(self):
        super(SmothL1Loss, self).__init__(reduction = tf.keras.losses.Reduction.NONE, name = 'SmothL1Loss')

    def call(self, y_true, y_pred):
        t = tf.abs(y_pred - y_true)
        return tf.where(t < 1, 0.5 * t ** 2, t - 0.5)
