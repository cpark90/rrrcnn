import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init

from detectron2.modeling.backbone.backbone import Backbone
from detectron2.layers import get_norm, Conv2d

from continuous.custom.continuous import R4ConvF, R4Conv, DefemLayer



__all__ = [
    "GENet",
]

dtype = torch.FloatTensor
bn_seperate = False

class GENet(Backbone):
    def __init__(self, input_shape, num_kernel, standard_channels, num_classes=None, norm="BN", with_1x1=False):
        super(GENet, self).__init__()
        self.num_classes = num_classes
        self.with_1x1 = with_1x1

        gef_conv_op = R4ConvF
        ge_conv_op = R4Conv
        defem_layer_op = DefemLayer

        in_channels = standard_channels
        out_channels = standard_channels
        batch_channels = standard_channels

        self.conv1_gef = gef_conv_op(
            in_channels=input_shape,
            out_channels=out_channels,
            padding=1,
            dilation=1,
            bias=False,
            norm=get_norm(norm, batch_channels),
        )

        self.conv1_ge = ge_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=1,
            dilation=1,
            bias=False,
            norm=get_norm(norm, batch_channels),
            with_1x1=with_1x1
        )
        self.conv2_ge = ge_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=1,
            dilation=1,
            bias=False,
            norm=get_norm(norm, batch_channels),
            with_1x1=with_1x1
        )
        self.conv3_ge = ge_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=1,
            dilation=1,
            bias=False,
            norm=get_norm(norm, batch_channels),
            with_1x1=with_1x1
        )
        self.conv4_ge = ge_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=1,
            dilation=1,
            bias=False,
            norm=get_norm(norm, batch_channels),
            with_1x1=with_1x1
        )
        self.conv5_ge = ge_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=1,
            dilation=1,
            bias=False,
            norm=get_norm(norm, batch_channels),
            with_1x1=with_1x1
        )
        self.conv6_ge = ge_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=1,
            dilation=1,
            bias=False,
            norm=get_norm(norm, batch_channels),
            with_1x1=with_1x1
        )
        self.conv7_ge = ge_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=1,
            dilation=1,
            bias=False,
            norm=get_norm(norm, batch_channels),
            with_1x1=with_1x1
        )
        self.conv8_ge = ge_conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=1,
            dilation=1,
            bias=False,
            norm=get_norm(norm, batch_channels),
            with_1x1=with_1x1
        )

        if self.with_1x1:
            in_channels = out_channels * 4
            self.conv1_1x1 = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                norm=get_norm(norm, out_channels)
                # activation=F.relu_
            )
            self.conv2_1x1 = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                norm=get_norm(norm, out_channels)
                # activation=F.relu_
            )
            self.conv3_1x1 = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                norm=get_norm(norm, out_channels)
                # activation=F.relu_
            )
            self.conv4_1x1 = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                norm=get_norm(norm, out_channels)
                # activation=F.relu_
            )
            self.conv5_1x1 = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                norm=get_norm(norm, out_channels)
                # activation=F.relu_
            )
            self.conv6_1x1 = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                norm=get_norm(norm, out_channels)
                # activation=F.relu_
            )
            self.conv7_1x1 = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                norm=get_norm(norm, out_channels)
                # activation=F.relu_
            )
            self.conv8_1x1 = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                norm=get_norm(norm, out_channels)
                # activation=F.relu_
            )

        self.defem_layer1 = defem_layer_op(
            shift=0,
            # norm=get_norm(norm, out_channels),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(out_channels, num_classes)

        weight_init.c2_msra_fill(self.conv1_gef)
        weight_init.c2_msra_fill(self.conv1_ge)
        weight_init.c2_msra_fill(self.conv2_ge)
        weight_init.c2_msra_fill(self.conv3_ge)
        weight_init.c2_msra_fill(self.conv4_ge)
        weight_init.c2_msra_fill(self.conv5_ge)
        weight_init.c2_msra_fill(self.conv6_ge)
        weight_init.c2_msra_fill(self.conv7_ge)
        weight_init.c2_msra_fill(self.conv8_ge)

        if self.with_1x1:
            weight_init.c2_msra_fill(self.conv1_1x1)
            weight_init.c2_msra_fill(self.conv2_1x1)
            weight_init.c2_msra_fill(self.conv3_1x1)
            weight_init.c2_msra_fill(self.conv4_1x1)
            weight_init.c2_msra_fill(self.conv5_1x1)
            weight_init.c2_msra_fill(self.conv6_1x1)
            weight_init.c2_msra_fill(self.conv7_1x1)
            weight_init.c2_msra_fill(self.conv8_1x1)


    def forward(self, x):
        outputs = {}

        x = self.conv1_gef(x)
        if self.with_1x1:
            x_in = [x[:, :, :, :, 0], x[:, :, :, :, 1], x[:, :, :, :, 2], x[:, :, :, :, 3]]
            x_1 = torch.cat([x_in[0], x_in[1], x_in[2], x_in[3]], dim=1)
            x_2 = torch.cat([x_in[1], x_in[2], x_in[3], x_in[0]], dim=1)
            x_3 = torch.cat([x_in[2], x_in[3], x_in[0], x_in[1]], dim=1)
            x_4 = torch.cat([x_in[3], x_in[0], x_in[1], x_in[2]], dim=1)
            x_1 = self.conv1_1x1(x_1)
            x_2 = self.conv1_1x1(x_2)
            x_3 = self.conv1_1x1(x_3)
            x_4 = self.conv1_1x1(x_4)
            x = torch.stack([x_1, x_2, x_3, x_4], dim=4)
        x = F.relu_(x)
        x = self.conv2_ge(x)
        if self.with_1x1:
            x_in = [x[:, :, :, :, 0], x[:, :, :, :, 1], x[:, :, :, :, 2], x[:, :, :, :, 3]]
            x_1 = torch.cat([x_in[0], x_in[1], x_in[2], x_in[3]], dim=1)
            x_2 = torch.cat([x_in[1], x_in[2], x_in[3], x_in[0]], dim=1)
            x_3 = torch.cat([x_in[2], x_in[3], x_in[0], x_in[1]], dim=1)
            x_4 = torch.cat([x_in[3], x_in[0], x_in[1], x_in[2]], dim=1)
            x_1 = self.conv2_1x1(x_1)
            x_2 = self.conv2_1x1(x_2)
            x_3 = self.conv2_1x1(x_3)
            x_4 = self.conv2_1x1(x_4)
            x = torch.stack([x_1, x_2, x_3, x_4], dim=4)
        x = F.relu_(x)
        out = []
        for idx in range(4):
            out.append(F.max_pool2d(x[:, :, :, :, idx], kernel_size=3, stride=2, padding=1))
        x = torch.stack(out, dim=4)
        x = F.relu_(x)

        x = self.conv3_ge(x)
        if self.with_1x1:
            x_in = [x[:, :, :, :, 0], x[:, :, :, :, 1], x[:, :, :, :, 2], x[:, :, :, :, 3]]
            x_1 = torch.cat([x_in[0], x_in[1], x_in[2], x_in[3]], dim=1)
            x_2 = torch.cat([x_in[1], x_in[2], x_in[3], x_in[0]], dim=1)
            x_3 = torch.cat([x_in[2], x_in[3], x_in[0], x_in[1]], dim=1)
            x_4 = torch.cat([x_in[3], x_in[0], x_in[1], x_in[2]], dim=1)
            x_1 = self.conv3_1x1(x_1)
            x_2 = self.conv3_1x1(x_2)
            x_3 = self.conv3_1x1(x_3)
            x_4 = self.conv3_1x1(x_4)
            x = torch.stack([x_1, x_2, x_3, x_4], dim=4)
        x = F.relu_(x)
        x = self.conv4_ge(x)
        if self.with_1x1:
            x_in = [x[:, :, :, :, 0], x[:, :, :, :, 1], x[:, :, :, :, 2], x[:, :, :, :, 3]]
            x_1 = torch.cat([x_in[0], x_in[1], x_in[2], x_in[3]], dim=1)
            x_2 = torch.cat([x_in[1], x_in[2], x_in[3], x_in[0]], dim=1)
            x_3 = torch.cat([x_in[2], x_in[3], x_in[0], x_in[1]], dim=1)
            x_4 = torch.cat([x_in[3], x_in[0], x_in[1], x_in[2]], dim=1)
            x_1 = self.conv4_1x1(x_1)
            x_2 = self.conv4_1x1(x_2)
            x_3 = self.conv4_1x1(x_3)
            x_4 = self.conv4_1x1(x_4)
            x = torch.stack([x_1, x_2, x_3, x_4], dim=4)
        x = F.relu_(x)
        out = []
        for idx in range(4):
            out.append(F.max_pool2d(x[:, :, :, :, idx], kernel_size=3, stride=2, padding=1))
        x = torch.stack(out, dim=4)
        x = F.relu_(x)

        x = self.conv5_ge(x)
        if self.with_1x1:
            x_in = [x[:, :, :, :, 0], x[:, :, :, :, 1], x[:, :, :, :, 2], x[:, :, :, :, 3]]
            x_1 = torch.cat([x_in[0], x_in[1], x_in[2], x_in[3]], dim=1)
            x_2 = torch.cat([x_in[1], x_in[2], x_in[3], x_in[0]], dim=1)
            x_3 = torch.cat([x_in[2], x_in[3], x_in[0], x_in[1]], dim=1)
            x_4 = torch.cat([x_in[3], x_in[0], x_in[1], x_in[2]], dim=1)
            x_1 = self.conv5_1x1(x_1)
            x_2 = self.conv5_1x1(x_2)
            x_3 = self.conv5_1x1(x_3)
            x_4 = self.conv5_1x1(x_4)
            x = torch.stack([x_1, x_2, x_3, x_4], dim=4)
        x = F.relu_(x)
        x = self.conv6_ge(x)
        if self.with_1x1:
            x_in = [x[:, :, :, :, 0], x[:, :, :, :, 1], x[:, :, :, :, 2], x[:, :, :, :, 3]]
            x_1 = torch.cat([x_in[0], x_in[1], x_in[2], x_in[3]], dim=1)
            x_2 = torch.cat([x_in[1], x_in[2], x_in[3], x_in[0]], dim=1)
            x_3 = torch.cat([x_in[2], x_in[3], x_in[0], x_in[1]], dim=1)
            x_4 = torch.cat([x_in[3], x_in[0], x_in[1], x_in[2]], dim=1)
            x_1 = self.conv6_1x1(x_1)
            x_2 = self.conv6_1x1(x_2)
            x_3 = self.conv6_1x1(x_3)
            x_4 = self.conv6_1x1(x_4)
            x = torch.stack([x_1, x_2, x_3, x_4], dim=4)
        x = F.relu_(x)
        out = []
        for idx in range(4):
            out.append(F.max_pool2d(x[:, :, :, :, idx], kernel_size=3, stride=2, padding=1))
        x = torch.stack(out, dim=4)
        x = F.relu_(x)

        x = self.conv7_ge(x)
        if self.with_1x1:
            x_in = [x[:, :, :, :, 0], x[:, :, :, :, 1], x[:, :, :, :, 2], x[:, :, :, :, 3]]
            x_1 = torch.cat([x_in[0], x_in[1], x_in[2], x_in[3]], dim=1)
            x_2 = torch.cat([x_in[1], x_in[2], x_in[3], x_in[0]], dim=1)
            x_3 = torch.cat([x_in[2], x_in[3], x_in[0], x_in[1]], dim=1)
            x_4 = torch.cat([x_in[3], x_in[0], x_in[1], x_in[2]], dim=1)
            x_1 = self.conv7_1x1(x_1)
            x_2 = self.conv7_1x1(x_2)
            x_3 = self.conv7_1x1(x_3)
            x_4 = self.conv7_1x1(x_4)
            x = torch.stack([x_1, x_2, x_3, x_4], dim=4)
        x = F.relu_(x)
        x = self.conv8_ge(x)
        if self.with_1x1:
            x_in = [x[:, :, :, :, 0], x[:, :, :, :, 1], x[:, :, :, :, 2], x[:, :, :, :, 3]]
            x_1 = torch.cat([x_in[0], x_in[1], x_in[2], x_in[3]], dim=1)
            x_2 = torch.cat([x_in[1], x_in[2], x_in[3], x_in[0]], dim=1)
            x_3 = torch.cat([x_in[2], x_in[3], x_in[0], x_in[1]], dim=1)
            x_4 = torch.cat([x_in[3], x_in[0], x_in[1], x_in[2]], dim=1)
            x_1 = self.conv8_1x1(x_1)
            x_2 = self.conv8_1x1(x_2)
            x_3 = self.conv8_1x1(x_3)
            x_4 = self.conv8_1x1(x_4)
            x = torch.stack([x_1, x_2, x_3, x_4], dim=4)
        x = F.relu_(x)
        out = []
        for idx in range(4):
            out.append(F.max_pool2d(x[:, :, :, :, idx], kernel_size=3, stride=2, padding=1))
        x = torch.stack(out, dim=4)
        x = F.relu_(x)

        x = torch.mean(x, dim=4)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.linear(x)
        outputs["linear"] = x

        return outputs

