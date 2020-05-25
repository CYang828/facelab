from abc import ABC

import facelab.facedataset
import facelab.facepreprocessor


class WiderFaceDataset(facelab.facedataset.FaceDataset, ABC):
    """features = {
        'image':
            tfds.features.Image(encoding_format='jpeg'),
        'image/filename':
            tfds.features.Text(),
        'faces':
            tfds.features.Sequence({
                'bbox': tfds.features.BBoxFeature(),
                'blur': tf.uint8,
                'expression': tf.bool,
                'illumination': tf.bool,
                'occlusion': tf.uint8,
                'pose': tf.bool,
                'invalid': tf.bool,
            }),
    }"""
    search_feature = 'image/filename'
    image_feature = 'image'

    def __init__(self, name, tfdataset=None):
        super(WiderFaceDataset, self).__init__(name, tfdataset)
        self.preprocessor = WiderFaceFacePreprocess(dataset=self)


class WiderFaceFacePreprocess(facelab.facepreprocessor.FacePreprocessor):

    def parse_and_make_example(self, example):
        return example['image'], example['faces']['bbox'] * 255

