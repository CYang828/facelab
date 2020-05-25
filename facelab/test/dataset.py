import tensorflow as tf
import tensorflow_datasets as tfds

from facelab.datasets import load

print(tfds.list_builders())
dataset, dataset_info = load('wider_face', split = 'train', shuffle_files = True, with_info = True)
print(dataset, dataset_info)


def _parse_function(example_proto):
    features = {"image": tf.io.FixedLenFeature((), tf.string, default_value = ""),
                "label": tf.io.FixedLenFeature((), tf.int32, default_value = 0)}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return parsed_features["image"], parsed_features["label"]


def g(x):
    return x['image'], x['faces']['bbox']
dataset = dataset.take(5).map(g)


print([i for i in dataset.as_numpy_iterator()])

# dataset = dataset.map(
#     normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# print(dataset_info.splits['train'].num_examples)
# print(dataset)
# dataset = dataset.cache().shuffle(128).prefetch(tf.data.experimental.AUTOTUNE)
# print(dataset.select(where='girl', limit = 5).show(column=1))

# dataset.shuffle(128)
# dataset = dataset.batch(128)
# dataset.prefetch(tf.data.experimental.AUTOTUNE)
# print(dataset.select(5))
# subset = dataset.take(5).as_numpy_iterator()
# for i in subset:
#     print(i['image/filename'])
#     print(i['image'].shape)
#     print(i['faces']['bbox'])
# print(subset)
# tfds.show_examples(dataset_info, subset)
# assert isinstance(dataset, tf.data.Dataset)
