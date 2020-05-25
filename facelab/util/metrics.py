import tensorflow as tf


def calc_iof(a, b):
    """calc intersection over foreground"""
    lt = tf.math.maximum(a[:, tf.newaxis, :2], b[:, :2])
    rb = tf.math.minimum(a[:, tf.newaxis, 2:], b[:, 2:])
    area_i = tf.math.reduce_prod(rb - lt, axis = 2) * tf.cast(tf.reduce_all(lt < rb, axis = 2), tf.float32)
    area_a = tf.math.reduce_prod(a[:, 2:] - a[:, :2], axis = 1)
    return area_i / tf.math.maximum(area_a[:, tf.newaxis], 1)


def calc_jaccard(box_a, box_b):
    """Compute the jaccard overlap of each bbox and prior.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = calc_intersect(box_a, box_b)
    area_a = tf.broadcast_to(
        tf.expand_dims(
            (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]), 1),
        tf.shape(inter))
    area_b = tf.broadcast_to(
        tf.expand_dims(
            (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]), 0),
        tf.shape(inter))
    union = area_a + area_b - inter
    return inter / union


def calc_intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2]:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    a = tf.shape(box_a)[0]
    b = tf.shape(box_b)[0]
    max_xy = tf.minimum(
        tf.broadcast_to(tf.expand_dims(box_a[:, 2:], 1), [a, b, 2]),
        tf.broadcast_to(tf.expand_dims(box_b[:, 2:], 0), [a, b, 2]))
    min_xy = tf.maximum(
        tf.broadcast_to(tf.expand_dims(box_a[:, :2], 1), [a, b, 2]),
        tf.broadcast_to(tf.expand_dims(box_b[:, :2], 0), [a, b, 2]))
    inter = tf.maximum((max_xy - min_xy), tf.zeros_like(max_xy - min_xy))
    return inter[:, :, 0] * inter[:, :, 1]