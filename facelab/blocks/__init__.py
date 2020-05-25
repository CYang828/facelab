from facelab.blocks.fpn import FPN
from facelab.blocks.ssh import SSH
from facelab.blocks.heads.bbox import BoundingBoxHead
from facelab.blocks.heads.landmark import LandmarkHead
from facelab.blocks.heads.classify import ClassifyHead


options = {
    'FPN': FPN,
    'SSH': SSH,
    'BoundingBoxHead': BoundingBoxHead,
    'LandMarkHead': LandmarkHead,
    'ClassifyHead': ClassifyHead,
}


def select(tp):
    global options
    return options.get(tp)
