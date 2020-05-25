import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard


class CallbackManager(list):
    def __init__(self, checkpoint_path, log_path='./logs/summary/'):
        self.checkpoint(checkpoint_path)
        self.tensorboard()
        self.early_stopping()
        self.summary = tf.summary.create_file_writer(log_path)
        super(CallbackManager, self).__init__()

    def checkpoint(self, filepath, monitor='val_loss', verbose=0, save_best_only=False,
                   save_weights_only=False, mode='auto', save_freq='epoch', **kwargs):
        checkpoint = ModelCheckpoint(filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq,
                                     **kwargs)
        self[0] = checkpoint

    def tensorboard(self, log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False,
                    update_freq='epoch', profile_batch=2, embeddings_freq=0,
                    embeddings_metadata=None, **kwargs):
        tensorboard = TensorBoard(log_dir, histogram_freq, write_graph, write_images, update_freq, profile_batch,
                                  embeddings_freq, embeddings_metadata, **kwargs)
        self[1] = tensorboard

    def early_stopping(self, monitor='val_loss', min_delta=0, patience=0, verbose=0,
                       mode='auto', baseline=None, restore_best_weights=False):
        early_stopping = EarlyStopping(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights)
        self[2] = early_stopping


