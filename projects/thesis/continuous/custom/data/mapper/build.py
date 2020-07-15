from detectron2.utils.registry import Registry

__all__ = ["DATASET_MAPPER_REGISTRY"]

DATASET_MAPPER_REGISTRY = Registry("DATASET_MAPPER")
DATASET_MAPPER_REGISTRY.__doc__ = """
Registry for dataset mapper, which extract feature maps from images
"""

def build_dataset_mapper(cfg, is_train):

    dataset_mapper_name = cfg.DATASETS.CUSTOM.DATASET_MAPPER
    dataset_mapper = DATASET_MAPPER_REGISTRY.get(dataset_mapper_name)(cfg, is_train)

    return dataset_mapper
