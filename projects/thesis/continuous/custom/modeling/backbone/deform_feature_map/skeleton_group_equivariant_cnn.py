import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init

from detectron2.modeling.backbone.backbone import Backbone
from detectron2.layers import get_norm, Conv2d

from continuous.custom.continuous import R8ConvF, R8Conv, DefemLayer



__all__ = [
    "R8Net",
]

dtype = torch.FloatTensor
bn_seperate = False

class R8Net(Backbone):
    def __init__(self, input_shape, num_kernel, standard_channels, num_classes=None, norm="BN"):
        super(R8Net, self).__init__()
        self.num_classes = num_classes

        r8_convf_op = R8ConvF
        r8_conv_op = R8Conv
        defem_layer_op = DefemLayer

        kernel_size = num_kernel
        dilation = 1

        in_channels = standard_channels
        out_channels = standard_channels
        batch_channels = standard_channels

        self.conv0 = r8_convf_op(
            in_channels=input_shape,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            norm=get_norm(norm, batch_channels)
        )
        self.conv1 = r8_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            norm=get_norm(norm, batch_channels)
        )
        self.conv2 = r8_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            norm=get_norm(norm, batch_channels)
        )
        self.conv3 = r8_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            norm=get_norm(norm, batch_channels)
        )
        self.conv4 = r8_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            norm=get_norm(norm, batch_channels)
        )
        self.conv5 = r8_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            norm=get_norm(norm, batch_channels)
        )
        self.conv6 = r8_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            norm=get_norm(norm, batch_channels)
        )
        self.conv7 = r8_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
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
        out = []
        for idx in range(8):
            out.append(F.max_pool2d(x[:, :, idx, :, :], kernel_size=3, stride=2, padding=1))
        x = torch.stack(out, dim=2)

        x = self.conv2(x)
        x = F.relu_(x)
        x = self.conv3(x)
        x = F.relu_(x)
        out = []
        for idx in range(8):
            out.append(F.max_pool2d(x[:, :, idx, :, :], kernel_size=3, stride=2, padding=1))
        x = torch.stack(out, dim=2)

        x = self.conv4(x)
        x = F.relu_(x)
        x = self.conv5(x)
        x = F.relu_(x)
        out = []
        for idx in range(8):
            out.append(F.max_pool2d(x[:, :, idx, :, :], kernel_size=3, stride=2, padding=1))
        x = torch.stack(out, dim=2)

        x = self.conv6(x)
        x = F.relu_(x)
        x = self.conv7(x)
        x = F.relu_(x)
        out = []
        for idx in range(8):
            out.append(F.max_pool2d(x[:, :, idx, :, :], kernel_size=3, stride=2, padding=1))
        x = torch.stack(out, dim=2)

        x = torch.mean(x, 2)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.linear(x)
        outputs["linear"] = x

        return outputs

