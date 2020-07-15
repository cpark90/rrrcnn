# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone import Backbone

BACKBONE_PRETRAIN_REGISTRY = Registry("BACKBONE_PRETRAIN")
BACKBONE_PRETRAIN_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

It must returns an instance of :class:`Backbone`.
"""


def build_backbone_pretrain(cfg, input_shape=None, num_classes=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    if num_classes is None:
        num_classes = cfg.CUSTOM.CLASSES

    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_PRETRAIN_REGISTRY.get(backbone_name)(cfg, input_shape, num_classes)
    assert isinstance(backbone, Backbone)
    return backbone
