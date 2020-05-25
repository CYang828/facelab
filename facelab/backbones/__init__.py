from facelab.backbones.retina import RetinaBackbone


import tensorflow

options = {
    'ResNet50': tensorflow.keras.applications.ResNet50,
    'MobileNet': tensorflow.keras.applications.MobileNet,
    'MobileNetV2': tensorflow.keras.applications.MobileNetV2,
    'RetinaBackbone': RetinaBackbone,
}


def select(tp):
    global options
    return options.get(tp)
