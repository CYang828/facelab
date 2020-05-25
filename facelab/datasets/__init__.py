import tensorflow_datasets as tfds

import facelab.facedataset
from facelab.datasets.widerface import WiderFaceDataset
from facelab.datasets.widerface_with_landmark5 import WiderFaceWtihLandmark5Train

registered_dataset = {
    'lfw': (tfds.load, None),
    'wider_face': (tfds.load, WiderFaceDataset),
    'wider_face_with_landmark5': (WiderFaceWtihLandmark5Train.load, WiderFaceWtihLandmark5Train),
    'vgg_face2': (tfds.load, None),
}


def load(name,
         split=None,
         data_dir=None,
         download=True,
         as_supervised=False,
         decoders=None,
         read_config=None,
         with_info=False):
    global registered_dataset
    dataset_loader = registered_dataset.get(name)
    if dataset_loader:
        dataset_callable = dataset_loader[0]
        dataset_maker = dataset_loader[1] if dataset_loader[1] else facelab.dataset.FaceDataset

        datasets = dataset_callable(name,
                                    split = split,
                                    data_dir = data_dir,
                                    download = download,
                                    as_supervised = as_supervised,
                                    decoders = decoders,
                                    read_config = read_config,
                                    with_info = with_info)
        dataset_info = datasets[1]
        if with_info:
            if isinstance(datasets, list):
                datasets = (dataset_maker.from_tfdataset(name = name, obj = ds) for ds in datasets[0])
                return datasets, dataset_info
            else:
                datasets = [dataset_maker.from_tfdataset(name = name, obj = ds) for ds in datasets[0]]
                return datasets[0] if len(datasets) == 1 else datasets, dataset_info

        else:
            if isinstance(datasets, list):
                datasets = (dataset_maker.from_tfdataset(name = name, obj = ds) for ds in datasets)
            else:
                datasets = dataset_maker.from_tfdataset(name = name, obj = datasets)
            return datasets
