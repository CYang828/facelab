import tensorflow as tf
from facelab.models import retinaface, summary


checkpoint_dir = 'retinaface-mbv2-ckpoint'
# define model from models and summary model

model = retinaface.RetinaFace.from_yaml('cfg.yaml')
checkpoint = tf.train.Checkpoint(model = model)
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint.restore(checkpoint_path)
# summary(model = model)
# https://github.com/tensorflow/tensorflow/issues/39843
# model.load_weights(checkpoint_path)
