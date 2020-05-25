import tensorflow as tf
from facelab.util import set_memory_growth
from facelab.models import summary
from facelab.models.retinaface import RetinaFace
from facelab.preprocess import FacePreprocess
from facelab.util.anchor import prior_box


with tf.device('/CPU:0'):
    model = RetinaFace.from_yaml('cfg.yaml')
    summary(model)
