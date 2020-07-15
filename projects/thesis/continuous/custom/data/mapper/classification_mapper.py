import torchvision.transforms as transforms
from .build import DATASET_MAPPER_REGISTRY

__all__ = ["ClassificationDatasetMapper"]

@DATASET_MAPPER_REGISTRY.register()
class ClassificationDatasetMapper:
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
        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT

        input_size     = cfg.MODEL.CUSTOM.INPUT_SIZE
        rotation_train = cfg.DATASETS.CUSTOM.ROTATION_TRAIN
        rotation_test  = cfg.DATASETS.CUSTOM.ROTATION_TEST
        # fmt: on

        self.transform_train = transforms.Compose([
            # transforms.Grayscale(3),
            transforms.RandomRotation(rotation_train),
            # transforms.Grayscale(1),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])
        self.transform_test = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.RandomRotation(rotation_test),
                # transforms.Grayscale(1),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
        ])

        self.is_train = is_train

    def __call__(self, datasets):
        # USER: Remove if you don't use pre-computed proposals.
        image = datasets[0]
        label = datasets[1]
        if not self.is_train:
            image = self.transform_test(image)
            return (image, label)
        image = self.transform_train(image)
        return (image, label)
