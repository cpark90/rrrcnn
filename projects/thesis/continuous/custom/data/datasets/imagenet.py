import io
import logging
import contextlib
import os

import torch

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from detectron2.data.catalog import MetadataCatalog, DatasetCatalog

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_imgnet_json", "load_imgnet_test_json"]

class ImageNetAPI:
    def __init__(self, data_root):
        self.train_path = os.path.join(data_root, 'train')
        self.test_path = os.path.join(data_root, 'test')

    @staticmethod
    def cvt_imgnet_to_detectron(class_idx: list):
        """

        :param (list) class_idices: an array of shape (N, 1) with class index
        :return: (numpy.ndarray) an array of shape (N, 1000) with one hot encoding
        """

        labels = torch.tensor(class_idx)
        return labels

def load_imgnet_test_json(data_root, dataset_name=None, extra_annotation_keys=None):
    with contextlib.redirect_stdout(io.StringIO()):
        img_api = ImageNetAPI(data_root)
        test_path = img_api.test_path

    # meta = MetadataCatalog.get(dataset_name)
    # meta.thing_classes = img_api.thing_classes

    dataset_dicts = []
    class_names = os.listdir(test_path)
    class_names.sort()
    for class_idx in range(len(class_names)):
        class_name = class_names[class_idx]
        class_path = os.path.join(test_path, class_name)
        file_names = os.listdir(class_path)

        for file_name in file_names:
            dataset_dicts.append({'file_name': os.path.join(class_path, file_name), 'class': img_api.cvt_imgnet_to_detectron([class_idx])})

    logger.info("Loaded {} images in imgnet format from {}".format(len(dataset_dicts), data_root))


    # val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(324),
    #         transforms.CenterCrop(299),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))

    return dataset_dicts


def load_imgnet_json(data_root, dataset_name=None, extra_annotation_keys=None):
    with contextlib.redirect_stdout(io.StringIO()):
        img_api = ImageNetAPI(data_root)
        train_path = img_api.train_path

    # meta = MetadataCatalog.get(dataset_name)
    # meta.thing_classes = img_api.thing_classes
    train_dataset = datasets.ImageFolder(
        train_path,
        transforms.Compose([
            transforms.RandomResizedCrop(size=299, scale=(0.08, 1), ratio=(0.75, 4/3)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-0.1, 0.1]),
            transforms.RandomRotation(degrees=(-45, 45)),
            transforms.ToTensor(),
        ]))

    logger.info("Loaded {} images in imgnet format from {}".format(len(train_dataset), data_root))
    return train_dataset

def register_imgnet_instance(name, data_root):
    DatasetCatalog.register(name, lambda: load_imgnet_json(data_root, name))

def register_imgnet_test_instance(name, data_root):
    DatasetCatalog.register(name, lambda: load_imgnet_test_json(data_root, name))

register_imgnet_instance("imgnet", '/ws/data/imagenet')
register_imgnet_test_instance("imgnet_test", '/ws/data/imagenet')
