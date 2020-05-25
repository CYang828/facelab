import tensorflow as tf

import facelab.facedataset
import facelab.facefeature as facefeature
import facelab.facepreprocessor
import facelab.util.python


class WiderFaceWtihLandmark5Train(facelab.facedataset.FaceDataset):
    def __init__(self, name, tfdataset=None):
        super(WiderFaceWtihLandmark5Train, self).__init__(name, tfdataset)
        self.preprocessor = facelab.facepreprocessor.FacePreprocessor(dataset = self)

    @classmethod
    def tfrecord2example(cls, tfrecord):
        features = {'image/img_name': tf.io.FixedLenFeature([], tf.string),
                    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
                    'image/object/landmark0/x': tf.io.VarLenFeature(tf.float32),
                    'image/object/landmark0/y': tf.io.VarLenFeature(tf.float32),
                    'image/object/landmark1/x': tf.io.VarLenFeature(tf.float32),
                    'image/object/landmark1/y': tf.io.VarLenFeature(tf.float32),
                    'image/object/landmark2/x': tf.io.VarLenFeature(tf.float32),
                    'image/object/landmark2/y': tf.io.VarLenFeature(tf.float32),
                    'image/object/landmark3/x': tf.io.VarLenFeature(tf.float32),
                    'image/object/landmark3/y': tf.io.VarLenFeature(tf.float32),
                    'image/object/landmark4/x': tf.io.VarLenFeature(tf.float32),
                    'image/object/landmark4/y': tf.io.VarLenFeature(tf.float32),
                    'image/object/landmark/valid': tf.io.VarLenFeature(tf.float32),
                    'image/encoded': tf.io.FixedLenFeature([], tf.string)}
        x = tf.io.parse_single_example(tfrecord, features)
        img = tf.image.decode_jpeg(x['image/encoded'], channels = 3)

        labels = tf.stack(
            [tf.sparse.to_dense(x['image/object/bbox/xmin']),
             tf.sparse.to_dense(x['image/object/bbox/ymin']),
             tf.sparse.to_dense(x['image/object/bbox/xmax']),
             tf.sparse.to_dense(x['image/object/bbox/ymax']),
             tf.sparse.to_dense(x['image/object/landmark0/x']),
             tf.sparse.to_dense(x['image/object/landmark0/y']),
             tf.sparse.to_dense(x['image/object/landmark1/x']),
             tf.sparse.to_dense(x['image/object/landmark1/y']),
             tf.sparse.to_dense(x['image/object/landmark2/x']),
             tf.sparse.to_dense(x['image/object/landmark2/y']),
             tf.sparse.to_dense(x['image/object/landmark3/x']),
             tf.sparse.to_dense(x['image/object/landmark3/y']),
             tf.sparse.to_dense(x['image/object/landmark4/x']),
             tf.sparse.to_dense(x['image/object/landmark4/y']),
             tf.sparse.to_dense(x['image/object/landmark/valid'])], axis = 1)

        bboxes = tf.stack([
            tf.sparse.to_dense(x['image/object/bbox/xmin']),
            tf.sparse.to_dense(x['image/object/bbox/ymin']),
            tf.sparse.to_dense(x['image/object/bbox/xmax']) - tf.sparse.to_dense(x['image/object/bbox/xmin']),
            tf.sparse.to_dense(x['image/object/bbox/ymax']) - tf.sparse.to_dense(x['image/object/bbox/ymin'])],
            axis = 1)

        landmarks = tf.stack(
            [tf.sparse.to_dense(x['image/object/landmark0/x']),
             tf.sparse.to_dense(x['image/object/landmark0/y']),
             tf.sparse.to_dense(x['image/object/landmark1/x']),
             tf.sparse.to_dense(x['image/object/landmark1/y']),
             tf.sparse.to_dense(x['image/object/landmark2/x']),
             tf.sparse.to_dense(x['image/object/landmark2/y']),
             tf.sparse.to_dense(x['image/object/landmark3/x']),
             tf.sparse.to_dense(x['image/object/landmark3/y']),
             tf.sparse.to_dense(x['image/object/landmark4/x']),
             tf.sparse.to_dense(x['image/object/landmark4/y'])], axis = 1)
        return {facefeature.SEARCH_FEATURE: x['image/img_name'],
                facefeature.IMAGE_FEATURE: img,
                facefeature.LABEL_FEATURE: labels,
                facefeature.BBOX_FEATURE: bboxes,
                facefeature.LANDMARK_TEATURE: landmarks}

    @classmethod
    def xy2example(cls, x, y):
        pass

    @property
    def size(self):
        return 172304

    @classmethod
    def download_metadata(cls):
        return {'train': ('http://rv.okjiaoyu.cn/widerface_with_landmark5_train.tfrecord',
                          'widerface_with_landmark5_train.tfrecord'),
                'val': ('http://rv.okjiaoyu.cn/widerface_with_landmark5_val.tfrecord',
                        'widerface_with_landmark5_val.tfrecord')}
