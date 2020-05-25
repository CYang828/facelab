import tensorflow as tf
import yaml
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter

import facelab.backbones
import facelab.blocks
import facelab.util.anchor


class RetinaFace(tf.keras.Model):
    def __init__(self, input_size, weights_decay, out_channel, num_anchor, backbone_type,
                 traing=False, iou_thredhold=0.4, score_thredhold=0.02, min_sizes=None,
                 steps=None, clip=None, variances=None, name="RetinaFace"):
        super(RetinaFace, self).__init__(name = name)
        input_size = input_size if traing else None
        weight_decay = weights_decay
        num_anchor = num_anchor
        backbone_type = backbone_type

        self.out_channel = out_channel
        self.iou_thredhold = iou_thredhold
        self.score_thredhold = score_thredhold
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip
        self.variances = variances

        self.backbone = facelab.backbones.select('RetinaBackbone')(input_shape = (input_size, input_size, 3),
                                                                   btype = backbone_type)
        self.fpn = facelab.blocks.select('FPN')(out_channel = out_channel, weight_decay = weight_decay, name = 'FPN')
        self.ssh1 = facelab.blocks.select('SSH')(out_channel = out_channel, weight_decay = weight_decay, name = 'SSH-1')
        self.ssh2 = facelab.blocks.select('SSH')(out_channel = out_channel, weight_decay = weight_decay, name = 'SSH-2')
        self.ssh3 = facelab.blocks.select('SSH')(out_channel = out_channel, weight_decay = weight_decay, name = 'SSH-3')

        self.bbox_head1 = facelab.blocks.select('BoundingBoxHead')(num_anchor = num_anchor,
                                                                   weight_decay = weight_decay, name = 'BBOX-HEAD-1')
        self.bbox_head2 = facelab.blocks.select('BoundingBoxHead')(num_anchor = num_anchor,
                                                                   weight_decay = weight_decay, name = 'BBOX-HEAD-2')
        self.bbox_head3 = facelab.blocks.select('BoundingBoxHead')(num_anchor = num_anchor,
                                                                   weight_decay = weight_decay, name = 'BBOX-HEAD-3')
        self.landmark_head1 = facelab.blocks.select('LandMarkHead')(num_anchor = num_anchor,
                                                                    weight_decay = weight_decay,
                                                                    name = 'LANDMARK-HEAD-1')
        self.landmark_head2 = facelab.blocks.select('LandMarkHead')(num_anchor = num_anchor,
                                                                    weight_decay = weight_decay,
                                                                    name = 'LANDMARK-HEAD-2')
        self.landmark_head3 = facelab.blocks.select('LandMarkHead')(num_anchor = num_anchor,
                                                                    weight_decay = weight_decay,
                                                                    name = 'LANDMARK-HEAD-3')
        self.claasify_head1 = facelab.blocks.select('ClassifyHead')(num_anchor = num_anchor,
                                                                    weight_decay = weight_decay,
                                                                    name = 'CLASSIFY-HEAD-1')
        self.claasify_head2 = facelab.blocks.select('ClassifyHead')(num_anchor = num_anchor,
                                                                    weight_decay = weight_decay,
                                                                    name = 'CLASSIFY-HEAD-2')
        self.claasify_head3 = facelab.blocks.select('ClassifyHead')(num_anchor = num_anchor,
                                                                    weight_decay = weight_decay,
                                                                    name = 'CLASSIFY-HEAD-3')

    def call(self, inputs, training=False, **kwargs):
        x = self.backbone(inputs)
        fpn1, fpn2, fpn3 = self.fpn(x)
        feature1 = self.ssh1(fpn1)
        feature2 = self.ssh2(fpn2)
        feature3 = self.ssh3(fpn3)
        bbox_regression1 = self.bbox_head1(feature1)
        bbox_regression2 = self.bbox_head2(feature2)
        bbox_regression3 = self.bbox_head3(feature3)
        bbox_regressions = tf.concat([bbox_regression1, bbox_regression2, bbox_regression3], axis = 1)
        landmark_regression1 = self.landmark_head1(feature1)
        landmark_regression2 = self.landmark_head2(feature2)
        landmark_regression3 = self.landmark_head3(feature3)
        landmark_regressions = tf.concat([landmark_regression1, landmark_regression2, landmark_regression3], axis = 1)
        classify1 = self.claasify_head1(feature1)
        classify2 = self.claasify_head2(feature2)
        classify3 = self.claasify_head3(feature3)
        classifications = tf.concat([classify1, classify2, classify3], axis = 1)
        classifications = tf.keras.layers.Softmax(axis = -1)(classifications)

        if training:
            tf.print('model out', bbox_regressions.shape, landmark_regressions.shape, classifications.shape)
            out = (bbox_regressions, landmark_regressions, classifications)
        else:
            # only for batch size 1
            preds = tf.concat(  # [bboxes, landmark, landmark_valid, conf]
                [bbox_regressions[0], landmark_regressions[0],
                 tf.ones_like(classifications[0, :, 0][..., tf.newaxis]),
                 classifications[0, :, 1][..., tf.newaxis]], 1)
            priors = facelab.util.anchor.prior_box_tf((tf.shape(inputs)[1], tf.shape(inputs)[2]),
                                                      self.min_sizes, self.steps, self.clip)
            decode_preds = facelab.util.anchor.decode_tf(preds, priors, self.variances)

            selected_indices = tf.image.non_max_suppression(
                boxes = decode_preds[:, :4],
                scores = decode_preds[:, -1],
                max_output_size = tf.shape(decode_preds)[0],
                iou_threshold = self.iou_thredhold,
                score_threshold = self.score_thredhold)

            out = tf.gather(decode_preds, selected_indices)
        return out

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, 'r') as f:
            cfg = yaml.load(f, Loader = yaml.Loader)
        model = cls(input_size = cfg['input_size'],
                    weights_decay = cfg['weights_decay'],
                    out_channel = cfg['out_channel'],
                    num_anchor = len(cfg['min_sizes'][0]),
                    backbone_type = cfg['backbone'],
                    min_sizes = cfg['min_sizes'],
                    steps = cfg['steps'],
                    clip = cfg['clip'],
                    variances = cfg['variances'])
        return model

    def train_step(self, data):
        """tf 2.2 issue: https://github.com/tensorflow/tensorflow/issues/39714"""
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training = True)
            reg_loss = tf.reduce_sum(self.losses)
            loc_loss, landm_loss, class_loss = self.compiled_loss(y, y_pred)
            total_loss = tf.add_n([reg_loss, loc_loss, landm_loss, class_loss])
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
