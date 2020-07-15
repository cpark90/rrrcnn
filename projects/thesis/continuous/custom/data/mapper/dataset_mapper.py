from detectron2.data import DatasetMapper

from .build import DATASET_MAPPER_REGISTRY

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["GeneralDatasetMapper"]

@DATASET_MAPPER_REGISTRY.register()
class GeneralDatasetMapper(DatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
