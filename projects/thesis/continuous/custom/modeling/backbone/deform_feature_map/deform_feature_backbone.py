# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    Conv2d,
    FrozenBatchNorm2d,
    ShapeSpec,
    get_norm,
)

from detectron2.layers import FrozenBatchNorm2d, get_norm, ShapeSpec
from detectron2.modeling import ResNetBlockBase, make_stage, ResNet
from detectron2.modeling.backbone.resnet import BasicStem, BottleneckBlock

from .deform_feature_conv import DefeConv
from .deform_feature_net import DefeNet

__all__ = ["DefeBottleneckBlock", "build_defenet_backbone_pretrain"]

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


class DefeBottleneckBlock(ResNetBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
    ):
        """
        Similar to :class:`BottleneckBlock`, but with deformable conv in the 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        deform_conv_op = DefeConv
        offset_channels = 2 * bottleneck_channels

        self.L = 1

        self.conv2_offset_a = Conv2d(
            bottleneck_channels,
            offset_channels,
            kernel_size=3,
            stride=1,
            padding=1 * dilation,
            dilation=dilation,
        )
        self.conv2_offset_b = Conv2d(
            bottleneck_channels,
            offset_channels,
            kernel_size=3,
            stride=1,
            padding=1 * dilation,
            dilation=dilation,
        )

        self.conv2 = deform_conv_op(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        nn.init.constant_(self.conv2_offset_a.weight, 0)
        nn.init.constant_(self.conv2_offset_a.bias, 0)
        nn.init.constant_(self.conv2_offset_b.weight, 0)
        nn.init.constant_(self.conv2_offset_b.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        # offset = torch.stack([self.conv2_offset_a(out), self.conv2_offset_b(out)], dim=4)
        # offset = torch.log_softmax(offset, dim=4)
        # offset = torch.exp(offset)

        # out = self.conv2(out, (offset[:,:,:,:,0] + 1 - offset[:,:,:,:,1]) / 2)
        # out, _ = torch.max(torch.stack([self.conv2(out, offset[:,:,:,:,0]), self.conv2(out, offset[:,:,:,:,1])], dim=4), dim=4)

        offset = self.conv2_offset_a(out)
        # offset_rate = torch.sigmoid(self.conv2_offset_b(out)) * self.L
        #
        # offset = torch.tanh(torch.mul(offset, offset_rate))
        # offset = torch.tanh(offset)
        print(offset)
        out = self.conv2(out, offset)


        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out

def build_defenet_backbone_pretrain(cfg, input_channels, num_classes):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "bottleneck_channels": bottleneck_channels,
            "out_channels": out_channels,
            "num_groups": num_groups,
            "norm": norm,
            "stride_in_1x1": stride_in_1x1,
            "dilation": dilation,
        }
        if deform_on_per_stage[idx]:
            stage_kargs["block_class"] = DefeBottleneckBlock
        else:
            stage_kargs["block_class"] = BottleneckBlock
        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return RearrNet(stem, stages, num_classes=num_classes, out_features=out_features)
