from facelab.datasets import load
from facelab.util.anchor import prior_box


(dataset_train, dataset_val), dataset_info = load('wider_face_with_landmark5', split = ['train', 'val'],
                                                  with_info = True)
# dataset_train.select(where = 'girl', limit = 1).show(column=2, with_bbox = True, with_landmark = True)
# dataset_train.preprocessor.random_crop(with_landmark = 5)(shuffle = True, buffer_size = 1000).check(
#     with_landmark = True, with_bbox = True)
# dataset_train.preprocessor.resize(112, with_landmark=5)(shuffle=True, buffer_size=1000).sample()

# dataset_train.preprocessor.padding()().check(with_landmark = True, with_bbox = True)
priors = prior_box(image_sizes = (224, 224),
                   min_sizes = [[16, 32], [64, 128], [256, 512]],
                   steps = [8, 16, 32])
dataset = dataset_train.preprocessor.resize(112, with_landmark=5).label_encode(priors)()
dataset.sample()
dataset = dataset.for_train(batch_size = 8)
for i in dataset.as_numpy_iterator():
    print(i[0].shape, i[1].shape)
    break

