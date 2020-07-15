# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import pickle
import random
import torch.utils.data as data

from detectron2.utils.serialize import PicklableWrapper

__all__ = ["ClassificationMapDataset"]


class ClassificationMapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.

    Args:
        dataset: a dataset where map function is applied.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        cur_idx = int(idx)

        return self._map_func(self._dataset[cur_idx])

