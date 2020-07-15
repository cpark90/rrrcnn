import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

import fvcore.nn.weight_init as weight_init

from detectron2.layers.wrappers import _NewEmptyTensorOp
from detectron2.layers import get_norm, Conv2d

from continuous.custom.continuous import DefemLayer
from continuous.custom.continuous.spatial_conv import _SpatialConv


__all__ = [
    "PSConvBlock"
]

spatial_conv = _SpatialConv.apply

class _SConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_type,
            spatial,
            spatial_1x1_in,
            stride=1,
            padding=1,
            dilation=1,
            norm=None,
            activation=None
    ):
        """
            kernel_type (int): 0=skel, 1=3x3, 2=bar
            spatial (bool): True=concat spatial dim, False=sum spatial dim
            is_first (bool): True=input element is non group equivaraint feature
            rot_1x1_in (bool): check previous layer has 1x1 compression
        """
        super(_SConv, self).__init__()

        if kernel_type == 0:
            self.num_kernel = 5
        elif kernel_type == 1:
            self.num_kernel = 9
        else:
            self.num_kernel = 3

        if spatial:
            in_channels = in_channels if spatial_1x1_in else self.num_kernel * in_channels

        self.kernel_type = kernel_type
        self.in_channels = in_channels
        self.out_channels = out_channels * self.num_kernel if spatial else out_channels
        self.spatial = spatial
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.activation = activation

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.num_kernel), requires_grad=True
        )
        if norm != None:
            if spatial:
                self.norm = get_norm(norm, out_channels * self.num_kernel)
            else:
                self.norm = get_norm(norm, out_channels)
        else:
            self.norm = norm

    def forward(self, x):
        # x is zero, return empty
        if x.numel() == 0:
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, (3, 3), self.stride
                )
            ]
            if self.spatial:
                output_shape = [x.shape[0], self.weight.shape[0] * self.num_kernel] + output_shape
            else:
                output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)

        x = spatial_conv(
            x,
            self.weight,
            self.kernel_type,
            self.num_kernel,
            0,
            self.stride,
            self.padding,
            self.dilation)

        # concat or sum.
        if self.spatial:
            spatial_out = []
            for idx in range(x.size(0)):
                spatial_out.append(x[idx, :, :, :, :])
            x = torch.cat(spatial_out, dim=1)
        else:
            x = torch.sum(x, dim=0)

        # batch norm is same for all rotation.
        if self.norm != None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_type=" + str(self.kernel_type)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", bias=False"
        return tmpstr

class PSConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_type,
            spatial,
            spatial_1x1_in,
            spatial_1x1_out,
            noise_var=0,
            stride=1,
            padding=1,
            dilation=1,
            norm=None,
            activation=None
    ):
        super(PSConvBlock, self).__init__()

        self.conv = _SConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_type=kernel_type,
            spatial=spatial,
            spatial_1x1_in=spatial_1x1_in,
            stride=stride,
            padding=padding,
            dilation=dilation,
            norm=norm,
            activation=activation
        )

        self.spatial = spatial
        self.spatial_1x1_out = spatial_1x1_out
        self.noise_var = noise_var
        self.num_kernel = self.conv.num_kernel
        self.activation = activation

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

        if self.noise_var > 0:
            self.positional_noise = DefemLayer()

        if self.spatial_1x1_out:
            self.conv_spatial_1x1 = Conv2d(
                in_channels=out_channels * self.num_kernel,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1
            )
            self.conv_spatial_norm = get_norm(norm, out_channels)
            weight_init.c2_msra_fill(self.conv_spatial_1x1)

    def forward(self, x):
        if self.noise_var > 0:
            offset = torch.randn(x.size(0), x.size(1) * 2, x.size(2), x.size(3), device=x.device) * self.noise_var
            grid_size = (x.size(2), x.size(3))
            x = self.positional_noise(x, offset, grid_size)

        x = self.conv(x)

        if self.spatial_1x1_out:
            x = self.conv_spatial_1x1(x)
            x = self.conv_spatial_norm(x)

        if self.activation != None:
            x = self.activation(x)

        return x
