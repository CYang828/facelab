import tensorflow as tf

from facelab.datasets import load
from facelab.loss.multitask import RetinaMultiTaskLoss
from facelab.lr.warmup import multi_step_warmup_lr
from facelab.models import retinaface, summary
from facelab.util.anchor import prior_box


# tf.debugging.set_log_device_placement(True)
tf.config.experimental_run_functions_eagerly(True)

# define hparameter
input_size = 640
steps = [8, 16, 32]
min_sizes = [[16, 32], [64, 128], [256, 512]]
priors = prior_box((input_size, input_size), min_sizes, steps)
batch_size = 8
epoch = 100

# define dataset by load from datasets
(dataset_train, dataset_val), dataset_info = load('wider_face_with_landmark5', split = ['train', 'val'],
                                                  with_info = True)
# preprocess dataset with dataset inner attribute: `preprocessor`
dataset = dataset_train.preprocessor.resize(input_size, with_landmark = 5). \
    label_encode(priors)(shuffle = True, buffer_size = 1000)

# define model from models and summary model
model = retinaface.RetinaFace.from_yaml('cfg.yaml')
summary(dataset, model)

# define loss
loss = RetinaMultiTaskLoss()
# define optimizer
steps_per_epoch = dataset.size // batch_size
learning_rate = multi_step_warmup_lr(
    initial_learning_rate= 1e-2,
    lr_steps = [e * steps_per_epoch for e in [50, 68]],
    lr_rate = 0.1,
    warmup_steps = 5 * steps_per_epoch,
    min_lr = 1e-3)
optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = 0.9, nesterov = True)

model.compile(optimizer = optimizer, loss = loss, metrics = 'accuracy')
model.fit(dataset.for_train(batch_size = 16), epochs = 100)
# Trainer(checkpoint_dir = '.checkpoints/', save_model = 'model/', log_dir = '.log/').\
#     set_dataset(dataset, batch_size=batch_size).\
#     set_model(model).\
#     set_loss(loss).\
#     set_optimizer(optimizer).\
#     run(epoch = epoch)
