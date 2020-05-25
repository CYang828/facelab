import tensorflow as tf


def summary(dataset=None, model=None):
    if dataset:
        dataset.sample(n=1)
    if model:
        dummy_inputs = tf.random.uniform(shape = (1, 224, 224, 3))
        model(dummy_inputs, training = True)
        model.summary()
