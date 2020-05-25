import six


def to_iter(e):
    """转换可迭代形式"""

    if isinstance(e, (six.string_types, six.string_types, six.class_types, six.text_type,
                      six.binary_type, six.class_types, six.integer_types, float)):
        return e,
    elif isinstance(e, list):
        return e
    else:
        return e
