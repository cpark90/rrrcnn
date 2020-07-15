import torch
import torch.nn as nn
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init
from detectron2.layers import (
    CNNBlockBase,
    get_norm,
    Conv2d
)

from continuous.custom.continuous import PR4Conv, PR8Conv, PSR4Conv, PSR8Conv, DefemLayer, GConv1x1

CONV_DICT = {
    "PR4Conv"   : PR4Conv,
    "PR8Conv"   : PR8Conv,
    "PSR4Conv"  : PSR4Conv,
    "PSR8Conv"  : PSR8Conv
}

__all__ = ["PSRBottleneckBlock"]

class PSRBottleneckBlock(CNNBlockBase):
    def __init__(
            self,
            in_channels,
            out_channels,
            *,
            bottleneck_channels,
            conv_name,
            conv_1x1_rot,
            rot_1x1_out,
            noise_var=0.0,
            stride=1,
            norm="BN"
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

        GConv = CONV_DICT[conv_name]
        self.conv2 = GConv(
            bottleneck_channels,
            bottleneck_channels,
            rot_1x1_in=True,
            rot_1x1_out=rot_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            norm=norm
        )

        self.kernel_rot = self.conv2.kernel_rot
        self.noise_var = noise_var
        self.defem_layer = DefemLayer()

        if in_channels != out_channels:
            self.shortcut = GConv1x1(
                rot_1x1=conv_1x1_rot,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                kernel_rot=self.kernel_rot,
                stride=1,
                padding=0,
                dilation=1,
                norm=get_norm(norm, out_channels))
            weight_init.c2_msra_fill(self.shortcut)
        else:
            self.shortcut = None

        self.conv1 = GConv1x1(
            rot_1x1=conv_1x1_rot,
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            kernel_rot=self.kernel_rot,
            stride=1,
            padding=0,
            dilation=1,
            norm=get_norm(norm, bottleneck_channels)
        )

        self.conv3 = GConv1x1(
            rot_1x1=conv_1x1_rot,
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            kernel_rot=self.kernel_rot,
            stride=1,
            padding=0,
            dilation=1,
            norm=get_norm(norm, out_channels)
        )

        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv3)

    def forward(self, x):
        x_out = []
        for rot in range(self.kernel_rot):
            grid_size = (int(x.size(3) / self.stride), int(x.size(4) / self.stride))
            offset = torch.randn(x.size(0), x.size(1) * 2, grid_size[0], grid_size[1], device=x.device) * self.noise_var
            x_out.append(self.defem_layer(x[:, :, rot, :, :], offset, grid_size, 0.5 if self.stride == 2 else 0))
        x_in = torch.stack(x_out, dim=2)

        out = self.conv1(x_in)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x_in)
        else:
            shortcut = x_in

        out += shortcut
        out = F.relu_(out)
        return out
