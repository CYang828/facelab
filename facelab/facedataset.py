import abc
import pathlib
import random

import fuzzywuzzy.process
import tensorflow as tf

import facelab.facefeature as facefeature
import facelab.image
import facelab.imageset
import facelab.util.python


class FaceDataset(tf.data.Dataset, abc.ABC):
    """wrapper for `tf.data.Dataset`
    network -> disk -> tfrecord -> tfdataset -> imageset -> image -> numpy

    TODO: snapshot in tf v2.3"""

    def __init__(self, name, tfdataset=None):
        self.tfdataset = tfdataset
        self.name = name
        super(FaceDataset, self).__init__(self.tfdataset._variant_tensor)

    def __str__(self):
        return '<{}>'.format(type(self).__name__)

    @classmethod
    def from_tfdataset(cls, name, obj):
        return cls(name, tfdataset = obj)

    @classmethod
    def from_disk(cls, name, disk_dir):
        dataset = cls.load_from_disk(disk_dir)
        return cls.from_tfdataset(name, dataset)

    @classmethod
    def from_http(cls, name, url, untar=True):
        data_root_orig = tf.keras.utils.get_file(
            fname = name,
            origin = url,
            untar = untar)
        dataset = cls.load_from_disk(data_root_orig)
        return cls.from_tfdataset(name, dataset)

    @staticmethod
    def load_from_disk(disk_dir, labels):
        data_root = pathlib.Path(disk_dir)
        all_image_paths = list(data_root.glob('*/*'))
        all_image_paths = [str(path) for path in all_image_paths]
        random.shuffle(all_image_paths)

        def load_image(path, label):
            img_raw = tf.io.read_file(path)
            img_tensor = tf.image.decode_image(img_raw, channels = 3)
            return img_tensor, label

        path_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, labels))
        dataset = path_ds.map(load_image, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        return dataset

    @classmethod
    def load(cls, name, split=None, data_dir=None, with_info=False, **kwargs):
        print('load dataset {}'.format(name))
        print('WARN', '{} these arguments are not used'.format(kwargs))
        splits = facelab.util.python.to_iter(split) if split else ['train']
        ret_dataset = []
        for split_name in splits:
            download_path = cls.download_metadata().get(split_name)[1]
            if data_dir:
                download_path = pathlib.Path(data_dir, download_path)
            tfrecord_path = tf.keras.utils.get_file(download_path,
                                                    cls.download_metadata().get(split_name)[0])
            dataset = tf.data.TFRecordDataset(tfrecord_path)
            dataset = dataset.map(cls.tfrecord2example)
            ret_dataset.append(dataset)
        if with_info:
            return ret_dataset, None
        else:
            return ret_dataset

    def select(self, where, limit=10):
        # get result by search condition
        candicates = {item[self.search_feature]: item for item in
                      self.take(limit * 20).as_example_iterator()}
        result = fuzzywuzzy.process.extract(where, candicates.keys())
        if len(result) > limit:
            result = random.sample(result, limit)
            result = result[: limit]

        # construct return ImageSet object
        images = []
        for name, _ in result:
            image = facelab.image.Image.from_example(candicates[name], dataset = self)
            images.append(image)
        return facelab.imageset.ImageSet.from_images(images)

    def check(self, with_bbox=False, with_landmark=False):
        def _wait_for_command():
            val = input('please press `Enter` to the next image, exit with `q`:\n')
            if val == 'q':
                return False
            else:
                return True

        for example in self.as_example_iterator():
            # example[self.image_feature] = tf.cast(example[self.image_feature], tf.uint16)
            image = facelab.image.Image.from_example(example, self)
            image.show(with_bbox, with_landmark)
            if not _wait_for_command():
                break

    def sample(self, n=1):
        for example in self.take(n).as_example_iterator():
            image = facelab.image.Image.from_example(example, dataset = self)
            label = example[self.label_feature]
            image.show()
            tf.print('image: {}\nlabel: {}'.format(image, label.shape))

    def cache(self, filename=""):
        return self.__class__.from_tfdataset('cache {}'.format(self), super().cache(filename))

    def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None):
        return self.__class__.from_tfdataset('shuffle {}'.format(self),
                                             super().shuffle(buffer_size, seed, reshuffle_each_iteration))

    def prefetch(self, buffer_size):
        return self.__class__.from_tfdataset('sub {}'.format(self), super().prefetch(buffer_size))

    def batch(self, batch_size, drop_remainder=False):
        return self.__class__.from_tfdataset('batch {}'.format(self), super().batch(batch_size, drop_remainder))

    def take(self, count):
        return self.__class__.from_tfdataset('take {}'.format(self), super().take(count))

    def repeat(self, count=None):
        return self.__class__.from_tfdataset(name = self.name, obj = super().repeat(count))

    def map(self, map_func, num_parallel_calls=None):
        return self.__class__.from_tfdataset(name = self.name, obj = super().map(map_func, num_parallel_calls))

    def _inputs(self):
        return self.tfdataset._inputs()

    def _shape_invariant_to_type_spec(self, shape):
        pass

    @property
    def element_spec(self):
        return self.tfdataset.element_spec

    @property
    def search_feature(self):
        return facefeature.SEARCH_FEATURE

    @property
    def image_feature(self):
        return facefeature.IMAGE_FEATURE

    @property
    def label_feature(self):
        return facefeature.LABEL_FEATURE

    @property
    def bbox_feature(self):
        return facefeature.BBOX_FEATURE

    @property
    def landmark_feature(self):
        return facefeature.LANDMARK_TEATURE

    @property
    def classfication_feature(self):
        return facefeature.CLASSFICATION_FEATURE

    @classmethod
    @abc.abstractmethod
    def tfrecord2example(cls, tfrecord):
        raise NotImplementedError

    @classmethod
    def example2xy(cls, example):
        return tf.cast(example[facefeature.IMAGE_FEATURE], tf.float32), \
               tf.cast(example[facefeature.LABEL_FEATURE], tf.float32)

    @classmethod
    @abc.abstractmethod
    def xy2example(cls, x, y):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def size(self):
        raise NotImplementedError

    @classmethod
    def download_metadata(cls):
        return {}

    def as_example_iterator(self):
        """return example iterator of dataset"""
        return super().as_numpy_iterator()

    def for_train(self, batch_size):
        def _for_train(example):
            return tf.cast(example[self.image_feature], tf.float32), \
                   tf.cast(example[self.label_feature], tf.float32)

        dataset = self.map(_for_train, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
