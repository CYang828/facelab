import abc

import tensorflow as tf

import facelab.util.lazy
from facelab.util.anchor import point_form, encode_bbox, encode_landmark
from facelab.util.metrics import calc_iof, calc_jaccard


class FacePreprocessor(abc.ABC, facelab.util.lazy.LazyCall):
    """face preprocessor using lazy call, it will process untill the object is called.
    subclass must implement parse_and_make_example method, it will return image, labels."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __call__(self, shuffle=False, buffer_size=None):
        # self.dataset = self.dataset.repeat()
        if shuffle:
            buffer_size = buffer_size if buffer_size else self.dataset.size
            self.dataset = self.dataset.shuffle(buffer_size = buffer_size)
        self.dataset = self.dataset.map(self._preprocess)
        return self.dataset

    def _preprocess(self, example):
        """recieve an example and return it by doing some preprocessings"""
        for bridge in self.bridges:
            bridge.fn_args = (example,) + bridge.fn_args
            example = bridge.call()
        example[self.dataset.image_feature] = tf.cast(example[self.dataset.image_feature], tf.uint8)
        example[self.dataset.label_feature] = tf.cast(example[self.dataset.label_feature], tf.uint8)
        return example

    def lazy_random_crop(self, example, max_loop=250, with_landmark=0, with_classfication=0,
                         precale=(0.3, 0.45, 0.6, 0.8, 1.0)):
        """random crop using iof and center location"""
        img, labels = self.dataset.example2xy(example)
        shape = tf.shape(img)

        def search_valid_crop(_i, _l, _w, _h, _wh):
            """make sure at least one face in this crop"""
            valid_crop = tf.constant(1, tf.int32)
            pre_scale = tf.constant(precale, dtype = tf.float32)
            scale = pre_scale[tf.random.uniform([], 0, 5, dtype = tf.int32)]
            short_side = tf.cast(tf.minimum(shape[0], shape[1]), tf.float32)
            _wh = _h = _w = tf.cast(scale * short_side, tf.int32)
            h_offset = tf.random.uniform([], 0, shape[0] - _h + 1, dtype = tf.int32)
            w_offset = tf.random.uniform([], 0, shape[1] - _w + 1, dtype = tf.int32)
            roi = tf.stack([w_offset, h_offset, w_offset + _w, h_offset + _h])
            roi = tf.cast(roi, tf.float32)
            # check iof
            value = calc_iof(_l[:, :4], roi[tf.newaxis])
            valid_crop = tf.cond(tf.math.reduce_any(value >= 1),
                                 lambda: valid_crop, lambda: 0)
            # check center
            centers = (_l[:, :2] + _l[:, 2:4]) / 2
            mask_a = tf.reduce_all(
                tf.math.logical_and(roi[:2] < centers, centers < roi[2:]),
                axis = 1)
            labels_t = tf.boolean_mask(_l, mask_a)
            valid_crop = tf.cond(tf.reduce_any(mask_a),
                                 lambda: valid_crop, lambda: 0)
            return tf.cond(valid_crop == 1,
                           lambda: (max_loop, labels_t, w_offset, h_offset, _wh),
                           lambda: (_i + 1, _l, _w, _h, _wh))

        # calc valid crop and offset
        _, labels, woffset, hoffset, wh = tf.while_loop(
            lambda _i, _l, _w, _h, _wh: tf.less(_i, max_loop),
            search_valid_crop,
            [tf.constant(-1), labels, 0, 0, 0])
        # crop image
        img = img[hoffset:hoffset + wh, woffset:woffset + wh, :]
        woffset = tf.cast(woffset, tf.float32)
        hoffset = tf.cast(hoffset, tf.float32)
        concat_labels = []
        # bbox transform
        bbox_l = [labels[:, 0] - woffset, labels[:, 1] - hoffset,
                  labels[:, 2] - woffset, labels[:, 3] - hoffset]
        bbox = [labels[:, 0] - woffset, labels[:, 1] - hoffset,
                labels[:, 2] - labels[:, 0], labels[:, 3] - labels[:, 1]]
        bbox = tf.stack(bbox, axis = 1)
        bbox_l = tf.stack(bbox_l, axis = 1)
        concat_labels.append(bbox_l)
        loffset = 3
        # landmark tranform
        landmark = []
        if with_landmark:
            for i in range(0, with_landmark * 2, 2):
                landmark.append(labels[:, loffset + 1] - woffset)
                landmark.append(labels[:, loffset + 2] - hoffset)
                loffset += 2
            landmark = tf.stack(landmark, axis = 1)
            concat_labels.append(landmark)
        # classfication
        classfication = []
        if with_classfication:
            for i in range(0, with_classfication):
                classfication.append(labels[:, loffset + i])
            classfication = tf.stack(classfication, axis = 1)
            concat_labels.append(classfication)
        # example replace
        labels = tf.concat(concat_labels, axis = 1)

        example[self.dataset.image_feature] = img
        example[self.dataset.label_feature] = labels
        example[self.dataset.bbox_feature] = bbox
        example[self.dataset.landmark_feature] = landmark
        return example

    def lazy_padding(self, example):
        """padding to square"""
        img, labels = self.dataset.example2xy(example)
        height, width = tf.shape(img)[0], tf.shape(img)[1]

        def pad_h():
            img_pad_h = tf.ones([width - height, width, 3]) * tf.reduce_mean(img, axis = [0, 1],
                                                                             keepdims = True)
            return tf.concat([img, img_pad_h], axis = 0)

        def pad_w():
            img_pad_w = tf.ones([height, height - width, 3]) * tf.reduce_mean(img, axis = [0, 1],
                                                                              keepdims = True)
            return tf.concat([img, img_pad_w], axis = 1)

        img = tf.case([(tf.greater(height, width), pad_w),
                       (tf.less(height, width), pad_h)], default = lambda: img)

        example[self.dataset.image_feature] = img
        return example

    def lazy_resize(self, example, image_shape, with_landmark=0):
        img, labels = self.dataset.example2xy(example)
        w_f = tf.cast(tf.shape(img)[1], tf.float32)
        h_f = tf.cast(tf.shape(img)[0], tf.float32)
        bbox = tf.stack([labels[:, 0] / w_f, labels[:, 1] / h_f,
                         labels[:, 2] / w_f, labels[:, 3] / h_f], axis = 1)
        loffset = 3
        landmark = []
        if with_landmark:
            for i in range(0, with_landmark * 2, 2):
                landmark.append(labels[:, loffset + 1] / w_f)
                landmark.append(labels[:, loffset + 2] / h_f)
                loffset += 2
            landmark = tf.stack(landmark, axis = 1)
        bbox = tf.concat([bbox, landmark], axis = 1)
        bbox = tf.clip_by_value(bbox, 0, 1)
        labels = tf.concat([bbox, labels[:, 3 + with_landmark * 2][:, tf.newaxis]], axis = 1)
        resize_case = tf.random.uniform([], 0, 5, dtype = tf.int32)

        def resize(method):
            def _resize():
                return tf.image.resize(
                    img, [image_shape, image_shape], method = method, antialias = True)

            return _resize

        img = tf.case([(tf.equal(resize_case, 0), resize('bicubic')),
                       (tf.equal(resize_case, 1), resize('area')),
                       (tf.equal(resize_case, 2), resize('nearest')),
                       (tf.equal(resize_case, 3), resize('lanczos3'))],
                      default = resize('bilinear'))
        example[self.dataset.image_feature] = img
        example[self.dataset.label_feature] = labels
        return example

    def lazy_flip(self, example, with_landmark=0, with_classfication=0):
        """TODO: 带有landmark和classfication的翻转"""
        img, labels = self.dataset.example2xy(example)
        flip_case = tf.random.uniform([], 0, 2, dtype = tf.int32)

        def flip_func():
            flip_img = tf.image.flip_left_right(img)
            lbls_t = [1 - labels[:, 2], labels[:, 1],
                      1 - labels[:, 0], labels[:, 3]]
            flip_labels = tf.stack(lbls_t, axis = 1)
            return flip_img, flip_labels

        img, labels = tf.case([(tf.equal(flip_case, 0), flip_func)],
                              default = lambda: (img, labels))
        example[self.dataset.image_feature] = img
        example[self.dataset.label_feature] = labels
        return example

    def lazy_distort(self, example):
        img, labels = self.dataset.example2xy(example)
        img = tf.image.random_brightness(img, 0.4)
        img = tf.image.random_contrast(img, 0.5, 1.5)
        img = tf.image.random_saturation(img, 0.5, 1.5)
        img = tf.image.random_hue(img, 0.1)
        example[self.dataset.image_feature] = img
        return example

    def lazy_label_encode(self, example, priors, match_thresh=0.45, ignore_thresh=0.3, variances=(0.1, 0.2)):
        assert ignore_thresh <= match_thresh
        img, labels = self.dataset.example2xy(example)
        priors = tf.cast(priors, tf.float32)
        bbox = labels[:, :4]
        landmark = labels[:, 4:-1]
        landmark_valid = labels[:, -1]

        # search the best prior by jaccard
        # calc each bbox and prior jaccard
        overlaps = calc_jaccard(bbox, point_form(priors))
        # (Bipartite Matching)
        # search the best prior for each ground truth（bbox）
        best_prior_overlap, best_prior_idx = tf.math.top_k(overlaps, k = 1)
        best_prior_overlap = best_prior_overlap[:, 0]
        best_prior_idx = best_prior_idx[:, 0]
        # search the best ground truth(bbox) for each prior
        overlaps_t = tf.transpose(overlaps)
        best_truth_overlap, best_truth_idx = tf.math.top_k(overlaps_t, k = 1)
        best_truth_overlap = best_truth_overlap[:, 0]
        best_truth_idx = best_truth_idx[:, 0]

        # make sure by match threshold
        def search_valid_prior(i, bt_idx, bt_overlap):
            bp_mask = tf.one_hot(best_prior_idx[i], tf.shape(bt_idx)[0])
            bp_mask_int = tf.cast(bp_mask, tf.int32)
            new_bt_idx = bt_idx * (1 - bp_mask_int) + bp_mask_int * i
            bp_mask_float = tf.cast(bp_mask, tf.float32)
            new_bt_overlap = bt_overlap * (1 - bp_mask_float) + bp_mask_float * 2
            return tf.cond(best_prior_overlap[i] > match_thresh,
                           lambda: (i + 1, new_bt_idx, new_bt_overlap),
                           lambda: (i + 1, bt_idx, bt_overlap))

        _, best_truth_idx, best_truth_overlap = tf.while_loop(
            lambda i, bt_idx, bt_overlap: tf.less(i, tf.shape(best_prior_idx)[0]),
            search_valid_prior,
            [tf.constant(0), best_truth_idx, best_truth_overlap])

        matches_bbox = tf.gather(bbox, best_truth_idx)
        matches_landm = tf.gather(landmark, best_truth_idx)
        matches_landm_v = tf.gather(landmark_valid, best_truth_idx)

        loc_t = encode_bbox(matches_bbox, priors, variances)
        landm_t = encode_landmark(matches_landm, priors, variances)
        landm_valid_t = tf.cast(matches_landm_v > 0, tf.float32)
        conf_t = tf.cast(best_truth_overlap > match_thresh, tf.float32)
        conf_t = tf.where(
            tf.logical_and(best_truth_overlap < match_thresh,
                           best_truth_overlap > ignore_thresh),
            tf.ones_like(conf_t) * -1, conf_t)  # 1: pos, 0: neg, -1: ignore
        example[self.dataset.label_feature] = tf.concat([loc_t, landm_t, landm_valid_t[..., tf.newaxis],
                                                         conf_t[..., tf.newaxis]], axis = 1)
        return example
