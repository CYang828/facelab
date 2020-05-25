import tensorflow as tf

import facelab.loss.smooth


class RetinaMultiTaskLoss(tf.keras.losses.Loss):

    def __init__(self, num_class=2, neg_pos_ratio=3):
        super(RetinaMultiTaskLoss, self).__init__(reduction = tf.keras.losses.Reduction.NONE,
                                                  name = 'RetinaMultiTaskLoss')
        self.num_class = num_class
        self.neg_pos_ratio = neg_pos_ratio

    def call(self, y_true, y_pred):
        num_batch = tf.shape(y_true)[0]
        num_prior = tf.shape(y_true)[1]

        loc_pred = tf.reshape(y_pred[0], [num_batch * num_prior, 4])
        landm_pred = tf.reshape(y_pred[1], [num_batch * num_prior, 10])
        class_pred = tf.reshape(y_pred[2], [num_batch * num_prior, self.num_class])
        loc_true = tf.reshape(y_true[..., :4], [num_batch * num_prior, 4])
        landm_true = tf.reshape(y_true[..., 4:14], [num_batch * num_prior, 10])
        landm_valid = tf.reshape(y_true[..., 14], [num_batch * num_prior, 1])
        class_true = tf.reshape(y_true[..., 15], [num_batch * num_prior, 1])

        # define filter mask: class_true = 1 (pos), 0 (neg), -1 (ignore)
        #                     landm_valid = 1 (w landm), 0 (w/o landm)
        mask_pos = tf.equal(class_true, 1)
        mask_neg = tf.equal(class_true, 0)
        mask_landm = tf.logical_and(tf.equal(landm_valid, 1), mask_pos)

        # landm loss (smooth L1)
        mask_landm_b = tf.broadcast_to(mask_landm, tf.shape(landm_true))
        loss_landm = facelab.loss.smooth.SmothL1Loss()(tf.boolean_mask(landm_true, mask_landm_b),
                                                       tf.boolean_mask(landm_pred, mask_landm_b))
        loss_landm = tf.reduce_mean(loss_landm)

        # localization loss (smooth L1)
        mask_pos_b = tf.broadcast_to(mask_pos, tf.shape(loc_true))
        loss_loc = facelab.loss.smooth.SmothL1Loss()(tf.boolean_mask(loc_true, mask_pos_b),
                                                     tf.boolean_mask(loc_pred, mask_pos_b))
        loss_loc = tf.reduce_mean(loss_loc)

        # classification loss (crossentropy)
        # 1. compute max conf across batch for hard negative mining
        loss_class = tf.where(mask_neg,
                              1 - class_pred[:, 0][..., tf.newaxis], 0)

        # 2. hard negative mining
        loss_class = tf.reshape(loss_class, [num_batch, num_prior])
        loss_class_idx = tf.argsort(loss_class, axis = 1, direction = 'DESCENDING')
        loss_class_idx_rank = tf.argsort(loss_class_idx, axis = 1)
        mask_pos_per_batch = tf.reshape(mask_pos, [num_batch, num_prior])
        num_pos_per_batch = tf.reduce_sum(
            tf.cast(mask_pos_per_batch, tf.float32), 1, keepdims = True)
        num_pos_per_batch = tf.maximum(num_pos_per_batch, 1)
        num_neg_per_batch = tf.minimum(self.neg_pos_ratio * num_pos_per_batch,
                                       tf.cast(num_prior, tf.float32) - 1)
        mask_hard_neg = tf.reshape(
            tf.cast(loss_class_idx_rank, tf.float32) < num_neg_per_batch,
            [num_batch * num_prior, 1])

        # 3. classification loss including positive and negative examples
        loss_class_mask = tf.logical_or(mask_pos, mask_hard_neg)
        loss_class_mask_b = tf.broadcast_to(loss_class_mask,
                                            tf.shape(class_pred))
        filter_class_true = tf.boolean_mask(tf.cast(mask_pos, tf.float32),
                                            loss_class_mask)
        filter_class_pred = tf.boolean_mask(class_pred, loss_class_mask_b)
        filter_class_pred = tf.reshape(filter_class_pred, [-1, self.num_class])
        loss_class = tf.keras.losses.sparse_categorical_crossentropy(
            y_true = filter_class_true, y_pred = filter_class_pred)
        loss_class = tf.reduce_mean(loss_class)

        return loss_loc, loss_landm, loss_class
