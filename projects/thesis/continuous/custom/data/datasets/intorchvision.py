import io
import logging
import contextlib
import os

import torch

from detectron2.data.catalog import MetadataCatalog, DatasetCatalog

import torchvision.datasets as datasets
"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_torchvision_json"]

class TorchvisionAPI:
    def __init__(self, data_root, data_name):
        self.data_path = os.path.join(data_root, data_name)

    @staticmethod
    def cvt_imgnet_to_detectron(class_idx: list):
        """

        :param (list) class_idices: an array of shape (N, 1) with class index
        :return: (numpy.ndarray) an array of shape (N, 1000) with one hot encoding
        """

        labels = torch.tensor(class_idx)
        return labels

def load_torchvision_json(data_root, dataset_name=None, train=False):

    # meta = MetadataCatalog.get(dataset_name)
    # meta.thing_classes = img_api.thing_classes
    if "cifar10" in dataset_name:
        with contextlib.redirect_stdout(io.StringIO()):
            img_api = TorchvisionAPI(data_root, 'cifar10')
            data_path = img_api.data_path

        dataset = datasets.CIFAR10(data_path, train=train, download=True)
    elif "cifar100" in dataset_name:
        with contextlib.redirect_stdout(io.StringIO()):
            img_api = TorchvisionAPI(data_root, 'cifar100')
            data_path = img_api.data_path

        dataset = datasets.CIFAR100(data_path, train=train, download=True)

    return dataset


def register_torchvision_instance(name, data_root):
    DatasetCatalog.register(name, lambda: load_torchvision_json(data_root, name, True))
def register_torchvision_test_instance(name, data_root):
    DatasetCatalog.register(name, lambda: load_torchvision_json(data_root, name, False))

register_torchvision_instance("cifar10", '/ws/data/open_datasets/classification')
register_torchvision_test_instance("cifar10_test", '/ws/data/open_datasets/classification')

register_torchvision_instance("cifar100", '/ws/data/open_datasets/classification')
register_torchvision_test_instance("cifar100_test", '/ws/data/open_datasets/classification')
