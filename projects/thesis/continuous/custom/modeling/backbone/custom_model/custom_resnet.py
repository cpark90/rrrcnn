# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch

from torch import nn
from torch.nn import functional as F

from detectron2.layers import FrozenBatchNorm2d, get_norm, ShapeSpec

from .resnet import *

from continuous.custom.modeling.backbone import BACKBONE_PRETRAIN_REGISTRY

# from .custom_fpn import CustomFPN
from .resnet_core import ResNet

__all__ = ["build_custom_backbone"]

def c2_xavier_fill(weight: nn.Parameter) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(weight, a=1)  # pyre-ignore

def c2_msra_fill(weight: nn.Parameter) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    # pyre-ignore
    nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")


def make_custom_stage(block_class, num_blocks, first_stride, *, in_channels, out_channels, **kwargs):
    """
    Create a resnet stage by creating many blocks for TridentNet.
    """
    assert "stride" not in kwargs, "Stride of blocks in make_stage cannot be changed."
    blocks = []
    for i in range(num_blocks):
        blocks.append(
            block_class(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=first_stride if i == 0 else 1,
                **kwargs,
            )
        )
        in_channels = out_channels
    return blocks


@BACKBONE_PRETRAIN_REGISTRY.register()
def build_custom_backbone(cfg, input_shape, num_classes=None):
    """
    Create a ResNet instance from config for TridentNet.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.CUSTOM.NORM
    stem = PSRBasicStem(
        in_channels=input_shape,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
        c7x7=cfg.MODEL.CUSTOM.RESNETS.STEM.CONVF_7x7,
        convf_name=cfg.MODEL.CUSTOM.RESNETS.STEM.CONVF_NAME,
        rot_1x1_out=cfg.MODEL.CUSTOM.RESNETS.ROT_1x1,
        noise_var=cfg.MODEL.CUSTOM.RESNETS.NOISE_VAR,
        stride_psr=cfg.MODEL.CUSTOM.RESNETS.STEM.STRIDE_PSR
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    out_features         = cfg.MODEL.RESNETS.OUT_FEATURES
    depth                = cfg.MODEL.RESNETS.DEPTH
    width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels  = width_per_group
    in_channels          = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    conv_name            = cfg.MODEL.CUSTOM.RESNETS.BLOCK.CONV_NAME
    conv_1x1_rot         = cfg.MODEL.CUSTOM.RESNETS.BLOCK.CONV_1x1_ROT
    rot_1x1_out          = cfg.MODEL.CUSTOM.RESNETS.ROT_1x1
    noise_var            = cfg.MODEL.CUSTOM.RESNETS.NOISE_VAR

    # fmt: on
    num_blocks_per_stage = {10: [1, 1, 1, 1], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    stages = []

    res_stage_idx = {"res2": 2, "res3": 3, "res4": 4, "res5": 5}
    out_stage_idx = [res_stage_idx[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        first_stride = 1 if idx == 0 else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "bottleneck_channels": bottleneck_channels,
            "out_channels": out_channels,
            "conv_name": conv_name,
            "conv_1x1_rot": conv_1x1_rot,
            "rot_1x1_out": rot_1x1_out,
            "noise_var": noise_var,
            "norm": norm
        }
        stage_kargs["block_class"] = PSRBottleneckBlock
        blocks = (
            make_custom_stage(**stage_kargs)
        )
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features, num_classes=num_classes)

# class LastLevelMaxPool(nn.Module):
#     """
#     This module is used in the original FPN to generate a downsampled
#     P6 feature from P5.
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.num_levels = 1
#         self.in_feature = "p5"
#
#     def forward(self, x):
#         return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]
#
# class R4LastLevelP6P7(nn.Module):
#     """
#     This module is used in RetinaNet to generate extra layers, P6 and P7 from
#     C5 feature.
#     """
#
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.num_levels = 2
#         self.in_feature = "res5"
#         self.p6 = R4ConvL(in_channels, out_channels, (3, 3), 2, 1)
#         self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
#         for module in [self.p6, self.p7]:
#             weight_init.c2_xavier_fill(module)
#
#     def forward(self, c5):
#         p6 = self.p6(c5)
#         p7 = self.p7(F.relu(p6))
#         return [p6, p7]

