import torch
import torch.nn as nn
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, get_norm

from continuous.custom.continuous import PR4ConvF, PR4Conv, PR8ConvF, PR8Conv, PSR4ConvF, PSR4Conv, PSR8ConvF, PSR8Conv, DefemLayer

CONV_DICT = {
    "PR4ConvF"  : PR4ConvF,
    "PR4Conv"   : PR4Conv,
    "PR8ConvF"  : PR8ConvF,
    "PR8Conv"   : PR8Conv,
    "PSR4ConvF" : PSR4ConvF,
    "PSR4Conv"  : PSR4Conv,
    "PSR8ConvF" : PSR8ConvF,
    "PSR8Conv"  : PSR8Conv
}

__all__ = ["PSRBasicStem"]

class PSRBasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, norm="BN", c7x7=True, convf_name=None, rot_1x1_out=True, noise_var=0.0, stride_psr=1):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()

        self.conv_7x7 = None
        self.conv_psr = None
        self.noise_var = noise_var

        self.out_channels = out_channels
        self.defem_layer = DefemLayer()

        if c7x7:
            self.conv_7x7 = Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
            weight_init.c2_msra_fill(self.conv_7x7)
            in_channels = out_channels

        if convf_name != "":
            GConvF = CONV_DICT[convf_name]
            self.conv_psr = GConvF(
                in_channels,
                out_channels,
                rot_1x1_out=rot_1x1_out,
                noise_var=noise_var,
                stride=stride_psr,
                padding=1,
                dilation=1,
                norm=norm)

    def forward(self, x):
        if self.conv_7x7 != None:
            if self.noise_var > 0:
                grid_size = (x.size(2), x.size(3))
                offset = torch.randn(x.size(0), x.size(1) * 2, grid_size[0], grid_size[1], device=x.device) * self.noise_var
                x = self.defem_layer(x, offset, grid_size, 0)
            x = self.conv_7x7(x)
            x = F.relu_(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        if self.conv_psr != None:
            x = self.conv_psr(x)
            x = F.relu_(x)

            if self.conv_7x7 is None:
                x = F.max_pool3d(x, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        return x

    @property
    def stride(self):
        return 4  # = stride 2 conv -> stride 2 max pool
