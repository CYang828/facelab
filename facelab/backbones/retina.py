import tensorflow as tf

import facelab.backbones


class RetinaBackbone(tf.keras.layers.Layer):
    def __init__(self, input_shape, btype='MobileNetV2', pick_layers=None, **kwargs):
        _backbone = facelab.backbones.select(btype)(input_shape = input_shape,
                                                    include_top = False,
                                                    weights = 'imagenet')
        super(RetinaBackbone, self).__init__()
        self.btype = btype
        self.pick_layers = pick_layers if pick_layers else []
        if self.btype == 'MobileNetV2':
            self.pick_layers = [54, 116, 143]
            self.preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        elif self.btype == 'ResNet50':
            self.pick_layers = [80, 142, 174]
            self.preprocess = tf.keras.applications.resnet.preprocess_input

        outputs = list(_backbone.layers[layer].output for layer in self.pick_layers)
        self.backbone = tf.keras.Model(_backbone.input,
                                       outputs,
                                       name = 'RetinaBackbone')

    def call(self, inputs, **kwargs):
        return self.backbone(self.preprocess(inputs))

    def __str__(self):
        return '{} {} extractor'.format(type(self).__name__, self.btype)
