# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch

from detectron2.layers import FrozenBatchNorm2d, get_norm, ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY

from .oriented_conv import *
from .rsnet import RsNet, RsNetBlockBase, make_rs_stage

__all__ = ["GTBottleneckBlock", "make_grouptrident_stage", "build_gtnet_backbone", "build_gtnet_backbone_pretrain"]

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

class GroupBasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()
        self.out_channels_var = out_channels
        self.conv1_weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 7, 7))
        self.conv1_stride = 2
        self.conv1_padding = 3
        self.bn1 = get_norm(norm, out_channels)
        self.relu = F.relu_
        c2_msra_fill(self.conv1_weight)

    def forward(self, x):
        x = oriented_conv_first(x, self.conv1_weight, stride=self.conv1_stride, padding=self.conv1_padding,
                                bn=[self.bn1], act=self.relu)
        x_tmp = []
        for idx in range(x.shape[4]):
            x_tmp.append(F.max_pool2d(x[:, :, :, :, idx], kernel_size=3, stride=2, padding=1))
        x = torch.stack(x_tmp, dim=4)

        return x

    @property
    def out_channels(self):
        return self.out_channels_var

    @property
    def stride(self):
        return 4  # = stride 2 conv -> stride 2 max pool

class R4BottleneckBlock(RsNetBlockBase):
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
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
            stride_in_1x1 (bool): when stride==2, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = R4Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=stride,
                padding=0,
                norm=get_norm(norm, out_channels),
                activation=None)
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = R4Conv(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=(1, 1),
            stride=stride_1x1,
            padding=0,
            norm=get_norm(norm, bottleneck_channels),
            activation=F.relu_
        )

        self.conv2 = R4Conv(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=(3, 3),
            stride=stride_3x3,
            padding=1,
            norm=get_norm(norm, bottleneck_channels),
            activation=F.relu_
        )

        self.conv3 = R4Conv(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=stride_1x1,
            padding=0,
            norm=get_norm(norm, out_channels),
            activation=None
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out

class GTBottleneckBlock(RsNetBlockBase):
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
        num_branch=3,
        dilations=(1, 2, 3),
        concat_output=False,
        test_branch_idx=-1,
    ):
        """
        Args:
            num_branch (int): the number of branches in TridentNet.
            dilations (tuple): the dilations of multiple branches in TridentNet.
            concat_output (bool): if concatenate outputs of multiple branches in TridentNet.
                Use 'True' for the last trident block.
        """
        assert num_branch == len(dilations)

        self.num_branch = num_branch
        self.concat_output = concat_output
        self.test_branch_idx = test_branch_idx
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = R4Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=stride,
                padding=0,
                norm=get_norm(norm, out_channels),
                activation=None)
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = R4Conv(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=(1, 1),
            stride=stride_1x1,
            padding=0,
            norm=get_norm(norm, bottleneck_channels),
            activation=F.relu_)

        self.conv2 = R4TConv(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=(3, 3),
            stride=stride_3x3,
            paddings=dilations,
            dilations=dilations,
            num_branch=num_branch,
            test_branch_idx=test_branch_idx,
            norm=get_norm(norm, bottleneck_channels),
            activation=F.relu_
        )

        self.conv3 = R4Conv(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=stride_1x1,
            padding=0,
            norm=get_norm(norm, out_channels),
            activation=None
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
        if not isinstance(x, list):
            x = [x] * num_branch

        out = [self.conv1(b) for b in x]
        out = self.conv2(out)
        out = [self.conv3(b) for b in out]

        if self.shortcut is not None:
            shortcut = [self.shortcut(b) for b in x]
        else:
            shortcut = x

        out = [out_b + shortcut_b for out_b, shortcut_b in zip(out, shortcut)]
        out = [F.relu_(b) for b in out]
        if self.concat_output:
            out = torch.cat(out)
        return out

def make_grouptrident_stage(block_class, num_blocks, first_stride, **kwargs):
    """
    Create a resnet stage by creating many blocks for TridentNet.
    """
    blocks = []
    for i in range(num_blocks - 1):
        blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    blocks.append(block_class(stride=1, concat_output=True, **kwargs))
    return blocks


@BACKBONE_REGISTRY.register()
def build_gtnet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config for TridentNet.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = GroupBasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    out_features         = cfg.MODEL.RESNETS.OUT_FEATURES
    depth                = cfg.MODEL.RESNETS.DEPTH
    num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels  = num_groups * width_per_group
    in_channels          = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation        = cfg.MODEL.RESNETS.RES5_DILATION
    num_branch           = cfg.MODEL.TRIDENT.NUM_BRANCH
    branch_dilations     = cfg.MODEL.TRIDENT.BRANCH_DILATIONS
    trident_stage        = cfg.MODEL.TRIDENT.TRIDENT_STAGE
    test_branch_idx      = cfg.MODEL.TRIDENT.TEST_BRANCH_IDX
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    stages = []

    res_stage_idx = {"res2": 2, "res3": 3, "res4": 4, "res5": 5}
    out_stage_idx = [res_stage_idx[f] for f in out_features]
    trident_stage_idx = res_stage_idx[trident_stage]
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
        if stage_idx == trident_stage_idx:
            stage_kargs["block_class"] = GTBottleneckBlock
            stage_kargs["num_branch"] = num_branch
            stage_kargs["dilations"] = branch_dilations
            stage_kargs["test_branch_idx"] = test_branch_idx
            stage_kargs.pop("dilation")
        else:
            stage_kargs["block_class"] = R4BottleneckBlock
        blocks = (
            make_grouptrident_stage(**stage_kargs)
            if stage_idx == trident_stage_idx
            else make_rs_stage(**stage_kargs)
        )
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return RsNet(stem, stages, out_features=out_features)


def build_gtnet_backbone_pretrain(cfg, input_channel, num_classes):
    """
    Create a ResNet instance from config for TridentNet.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = GroupBasicStem(
        in_channels=input_channel,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    out_features         = cfg.MODEL.RESNETS.OUT_FEATURES
    depth                = cfg.MODEL.RESNETS.DEPTH
    num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels  = num_groups * width_per_group
    in_channels          = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation        = cfg.MODEL.RESNETS.RES5_DILATION
    num_branch           = cfg.MODEL.TRIDENT.NUM_BRANCH
    branch_dilations     = cfg.MODEL.TRIDENT.BRANCH_DILATIONS
    trident_stage        = cfg.MODEL.TRIDENT.TRIDENT_STAGE
    test_branch_idx      = cfg.MODEL.TRIDENT.TEST_BRANCH_IDX
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    stages = []

    res_stage_idx = {"res2": 2, "res3": 3, "res4": 4, "res5": 5}
    out_stage_idx = [res_stage_idx[f] for f in out_features]
    trident_stage_idx = res_stage_idx[trident_stage]
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
        if stage_idx == trident_stage_idx:
            stage_kargs["block_class"] = GTBottleneckBlock
            stage_kargs["num_branch"] = num_branch
            stage_kargs["dilations"] = branch_dilations
            stage_kargs["test_branch_idx"] = test_branch_idx
            stage_kargs.pop("dilation")
        else:
            stage_kargs["block_class"] = R4BottleneckBlock
        blocks = (
            make_grouptrident_stage(**stage_kargs)
            if stage_idx == trident_stage_idx
            else make_rs_stage(**stage_kargs)
        )
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return RsNet(stem, stages, num_classes=num_classes, out_features=out_features)
