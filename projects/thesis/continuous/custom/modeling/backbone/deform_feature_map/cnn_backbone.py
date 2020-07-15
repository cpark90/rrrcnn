# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from continuous.custom.modeling.backbone import BACKBONE_PRETRAIN_REGISTRY

from .vanilla_cnn import CNNNet
from .skeleton_cnn import SkeletonNet
from .skeleton_group_equivariant_cnn import R8Net
from .group_equivariant_cnn import GENet
from .psr_cnn import PSRNet
from .ps_cnn import PSNet

__all__ = ["build_vanilla_cnn_backbone",
           "build_skeleton_backbone",
           "build_skeleton_group_equivariant_backbone",
           "build_group_equivariant_backbone",
           "build_psr_backbone",
           "build_ps_backbone"
           ]


@BACKBONE_PRETRAIN_REGISTRY.register()
def build_vanilla_cnn_backbone(cfg, input_shape, num_classes):
    channels        = cfg.MODEL.CUSTOM.CHANNELS

    return CNNNet(input_shape=input_shape, standard_channels=channels, num_classes=num_classes)


@BACKBONE_PRETRAIN_REGISTRY.register()
def build_skeleton_backbone(cfg, input_shape, num_classes):
    channels        = cfg.MODEL.CUSTOM.CHANNELS
    # num_kernel      = cfg.MODEL.CUSTOM.NUM_KERNEL
    num_kernel      = 5

    return SkeletonNet(input_shape=input_shape, num_kernel=num_kernel, standard_channels=channels, num_classes=num_classes)


@BACKBONE_PRETRAIN_REGISTRY.register()
def build_skeleton_group_equivariant_backbone(cfg, input_shape, num_classes):
    channels        = cfg.MODEL.CUSTOM.CHANNELS
    # num_kernel      = cfg.MODEL.CUSTOM.NUM_KERNEL
    num_kernel      = 5

    return R8Net(input_shape=input_shape, num_kernel=num_kernel, standard_channels=channels, num_classes=num_classes)


@BACKBONE_PRETRAIN_REGISTRY.register()
def build_group_equivariant_backbone(cfg, input_shape, num_classes):
    channels        = cfg.MODEL.CUSTOM.CHANNELS
    with_1x1        = cfg.MODEL.CUSTOM.PSR.ROT_1x1
    norm            = cfg.MODEL.CUSTOM.NORM
    # num_kernel      = cfg.MODEL.CUSTOM.NUM_KERNEL
    num_kernel      = 3

    return GENet(input_shape=input_shape, num_kernel=num_kernel, standard_channels=channels, num_classes=num_classes, norm=norm, with_1x1=with_1x1)


@BACKBONE_PRETRAIN_REGISTRY.register()
def build_psr_backbone(cfg, input_shape, num_classes):
    convf_name      = cfg.MODEL.CUSTOM.PSR.CONVF_NAME
    conv_name       = cfg.MODEL.CUSTOM.PSR.CONV_NAME
    channels        = cfg.MODEL.CUSTOM.CHANNELS
    rot_1x1         = cfg.MODEL.CUSTOM.PSR.ROT_1x1
    noise_var       = cfg.MODEL.CUSTOM.PSR.NOISE_VAR
    norm            = cfg.MODEL.CUSTOM.NORM

    return PSRNet(convf_name, conv_name, input_shape, num_classes, channels, rot_1x1, noise_var, norm)


@BACKBONE_PRETRAIN_REGISTRY.register()
def build_ps_backbone(cfg, input_shape, num_classes):
    channels        = cfg.MODEL.CUSTOM.CHANNELS
    kernel_type     = cfg.MODEL.CUSTOM.PS.KERNEL_TYPE
    spatial         = cfg.MODEL.CUSTOM.PS.SPATIAL
    spatial_1x1     = cfg.MODEL.CUSTOM.PS.SPATIAL_1x1
    noise_var       = cfg.MODEL.CUSTOM.PS.NOISE_VAR
    norm            = cfg.MODEL.CUSTOM.NORM

    return PSNet(input_shape, num_classes, channels, kernel_type, spatial, spatial_1x1, noise_var, norm)
