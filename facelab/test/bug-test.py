import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter


class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.flat = tf.keras.layers.Flatten(input_shape = (28, 28))
        
    def call(self, inputs, training=False, **kwargs):
        x = self.flat(inputs)
        out = (x, x, x)
        return out

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training = True)
            loss0 = tf.reduce_sum(self.losses)
            loss1, loss2, loss3 = self.loss(y, y_pred)
            total_loss = tf.reduce_sum([loss0, loss1, loss2, loss3])
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


class MultiTaskLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(MultiTaskLoss, self).__init__(reduction = tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        tf.print(y_pred[0].shape, y_pred[1].shape, y_pred[2].shape)
        loss1 = tf.reduce_sum(y_pred[0])
        loss2 = tf.reduce_sum(y_pred[1])
        loss3 = tf.reduce_sum(y_pred[2])
        return tf.cast(loss1, tf.float32), tf.cast(loss2, tf.float32), tf.cast(loss3, tf.float32)


tf.config.experimental_run_functions_eagerly(True)

tfds.list_builders()
dataset = tfds.load('mnist', split='train')
dataset = dataset.map(lambda exa: (exa['image'], exa['label']))
dataset = dataset.batch(8)
model = CustomModel()
loss = MultiTaskLoss()
model.compile(loss = loss, optimizer = 'Adam')
model.fit(dataset, epochs=1)
