import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init

from detectron2.modeling.backbone.backbone import Backbone
from detectron2.layers import get_norm, Conv2d

from continuous.custom.continuous import SkeletonConv, DefemLayer



__all__ = [
    "SkeletonNet",
]

dtype = torch.FloatTensor
bn_seperate = False

class SkeletonNet(Backbone):
    def __init__(self, input_shape, num_kernel, standard_channels, num_classes=None, norm="BN"):
        super(SkeletonNet, self).__init__()
        self.num_classes = num_classes

        skel_conv_op = SkeletonConv
        defem_layer_op = DefemLayer

        kernel_size = num_kernel
        dilation = 1

        in_channels = standard_channels
        out_channels = standard_channels
        batch_channels = standard_channels

        self.conv0 = skel_conv_op(
            in_channels=input_shape,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            step=0,
            norm=get_norm(norm, batch_channels)
        )
        self.conv1 = skel_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            step=1,
            norm=get_norm(norm, batch_channels)
        )
        self.conv2 = skel_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            step=0,
            norm=get_norm(norm, batch_channels)
        )
        self.conv3 = skel_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            step=1,
            norm=get_norm(norm, batch_channels)
        )
        self.conv4 = skel_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            step=0,
            norm=get_norm(norm, batch_channels)
        )
        self.conv5 = skel_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            step=1,
            norm=get_norm(norm, batch_channels)
        )
        self.conv6 = skel_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            step=0,
            norm=get_norm(norm, batch_channels)
        )
        self.conv7 = skel_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            step=1,
            norm=get_norm(norm, batch_channels)
        )

        self.defem_layer1 = defem_layer_op(
            out_channels,
            shift=0,
            # norm=get_norm(norm, batch_channels),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(out_channels, num_classes)

        weight_init.c2_msra_fill(self.conv0)
        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)
        weight_init.c2_msra_fill(self.conv3)
        weight_init.c2_msra_fill(self.conv4)
        weight_init.c2_msra_fill(self.conv5)
        weight_init.c2_msra_fill(self.conv6)
        weight_init.c2_msra_fill(self.conv7)

    def forward(self, x):

        outputs = {}
        x = self.conv0(x)
        x = F.relu_(x)
        x = self.conv1(x)
        x = F.relu_(x)
        F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.conv2(x)
        x = F.relu_(x)
        x = self.conv3(x)
        x = F.relu_(x)
        F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.conv4(x)
        x = F.relu_(x)
        x = self.conv5(x)
        x = F.relu_(x)
        F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.conv6(x)
        x = F.relu_(x)
        x = self.conv7(x)
        x = F.relu_(x)
        F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.linear(x)
        outputs["linear"] = x

        return outputs

