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
    "PSR4ConvF", "PSR4Conv", "PSR8ConvF", "PSR8Conv", "GConv1x1"
]

spatial_conv = _SpatialConv.apply

class GConv1x1(nn.Module):
    def __init__(
        self,
        rot_1x1,
        in_channels,
        out_channels,
        kernel_size,
        kernel_rot,
        stride,
        padding,
        dilation,
        norm=None
    ):
        super(GConv1x1, self).__init__()
        self.rot_1x1 = rot_1x1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.kernel_rot = kernel_rot
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.diltation = _pair(dilation)
        self.norm = norm

        if rot_1x1:
            self.weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels, kernel_rot, *self.kernel_size), requires_grad=True
            )
        else:
            self.weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels, 1, *self.kernel_size), requires_grad=True
            )
        self.bias = None

    def forward(self, x):
        if self.rot_1x1:
            x_out = []
            for rot in range(self.kernel_rot):
                rot_idx = [int((idx - rot) % self.kernel_rot) for idx in range(self.kernel_rot)]
                x_rot = F.conv3d(x, self.weight[:, :, rot_idx, :, :], stride=(1, *self.stride), padding=(0, *self.padding), dilation=(1, *self.diltation))
                x_rot = x_rot.view(x_rot.size(0), x_rot.size(1), x_rot.size(3), x_rot.size(4))
                if self.norm != None:
                    x_rot = self.norm(x_rot)
                x_out.append(x_rot)
            x = torch.stack(x_out, dim=2)
        else:
            x = F.conv3d(
                x,
                self.weight,
                stride=(1, *self.stride), padding=(0, *self.padding), dilation=(1, *self.diltation))
            if self.norm != None:
                for rot in range(self.kernel_rot):
                    x[:, :, rot, :, :] = self.norm(x[:, :, rot, :, :])
        return x

class GConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_type,
        is_first,
        rot_1x1_in,
        stride=1,
        padding=1,
        dilation=1,
        norm=None,
        activation=None,
    ):
        """
            kernel_type (int): 0=skel, 1=3x3, 2=bar
            spatial (bool): True=concat spatial dim, False=sum spatial dim
            is_first (bool): True=input element is non group equivaraint feature
            rot_1x1_in (bool): check previous layer has 1x1 compression
        """
        super(GConv, self).__init__()

        if kernel_type == 0:
            self.num_kernel = 5
            self.kernel_rot = 8
        elif kernel_type == 1:
            self.num_kernel = 9
            self.kernel_rot = 4
        else:
            self.num_kernel = 3
            self.kernel_rot = 8

        self.kernel_type = kernel_type
        self.in_channels = in_channels
        self.out_channels = out_channels * self.num_kernel
        self.is_first = is_first
        self.rot_1x1_in = rot_1x1_in
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.norm = norm
        self.activation = activation

        if is_first or rot_1x1_in:
            self.weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels, self.num_kernel), requires_grad=True
            )
        else:
            self.weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels, self.kernel_rot, self.num_kernel), requires_grad=True
            )

    def forward(self, x_tensor):
        # x_tensor is zero, return empty
        if x_tensor.numel() == 0:
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x_tensor.shape[-2:], self.padding, self.dilation, (3, 3), self.stride
                )
            ]
            if self.spatial:
                output_shape = [x_tensor.shape[0], self.weight.shape[0] * self.num_kernel, self.kernel_rot] + output_shape
            else:
                output_shape = [x_tensor.shape[0], self.weight.shape[0], self.kernel_rot] + output_shape
            return _NewEmptyTensorOp.apply(x_tensor, output_shape)

        step = int(8 / self.kernel_rot)

        if not self.rot_1x1_in and not self.is_first:
            x_tensor = torch.cat(
                [x_tensor[:, :, idx, :, :] for idx in range(self.kernel_rot)]
                , dim=1)

        rot_out = []
        for rot in range(self.kernel_rot):
            if self.is_first:
                x_rot = x_tensor
                weight = self.weight
            elif self.rot_1x1_in:
                x_rot = x_tensor[:, :, rot, :, :]
                weight = self.weight
            else:
                x_rot = x_tensor
                weight = torch.cat(
                    [self.weight[:, :, int((idx - rot) % self.kernel_rot), :] for idx in range(self.kernel_rot)],
                    dim=1)

            x_rot = spatial_conv(
                x_rot,
                weight,
                self.kernel_type,
                self.num_kernel,
                step * rot,
                self.stride,
                self.padding,
                self.dilation)

            # concat or sum.
            spatial_out = []
            for idx in range(x_rot.size(0)):
                spatial_out.append(x_rot[idx, :, :, :, :])
            x_rot = torch.cat(spatial_out, dim=1)

            # batch norm is same for all rotation.
            if self.norm != None:
                x_rot = self.norm(x_rot)

            rot_out.append(x_rot)

        x_out = torch.stack(rot_out, dim=2)

        if self.activation is not None:
            x_out = self.activation(x_out)
        return x_out

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_type=" + str(self.kernel_type)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", bias=False"
        return tmpstr


class _SR4Conv(GConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        is_first,
        rot_1x1_in,
        stride=1,
        padding=1,
        dilation=1,
        norm=None,
        activation=None,
    ):
        if norm != None:
            norm = get_norm(norm, out_channels * 9)

        super(_SR4Conv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_type=1,
            is_first=is_first,
            rot_1x1_in=rot_1x1_in,
            stride=stride,
            padding=padding,
            dilation=dilation,
            norm=norm,
            activation=activation)

class _SR8Conv(GConv):
    def __init__(
            self,
            in_channels,
            out_channels,
            is_first,
            rot_1x1_in,
            stride=1,
            padding=0,
            dilation=1,
            norm=None,
            activation=None,
    ):
        if norm != None:
            norm = get_norm(norm, out_channels * 5)

        super(_SR8Conv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_type=0,
            is_first=is_first,
            rot_1x1_in=rot_1x1_in,
            stride=stride,
            padding=padding,
            dilation=dilation,
            norm=norm,
            activation=activation)

class PSRConvBlock(nn.Module):
    def __init__(
            self,
            _SRConv,
            in_channels,
            out_channels,
            is_first,
            rot_1x1_in,
            rot_1x1_out,
            noise_var=0,
            stride=1,
            padding=1,
            dilation=1,
            norm=None,
            activation=None,
    ):
        super(PSRConvBlock, self).__init__()

        self.conv = _SRConv(
            in_channels=in_channels,
            out_channels=out_channels,
            is_first=is_first,
            rot_1x1_in=rot_1x1_in,
            stride=stride,
            padding=padding,
            dilation=dilation,
            norm=None,#norm
            activation=F.relu
        )

        self.is_first = is_first
        self.rot_1x1_out = rot_1x1_out
        self.noise_var = noise_var
        self.num_kernel = self.conv.num_kernel
        self.kernel_rot = self.conv.kernel_rot
        self.activation = activation

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

        if self.noise_var > 0:
            self.positional_noise = DefemLayer()

        self.conv_spatial_1x1 = GConv1x1(
            rot_1x1=False,
            in_channels=out_channels * self.num_kernel,
            out_channels=out_channels,
            kernel_size=1,
            kernel_rot=self.kernel_rot,
            stride=1,
            padding=0,
            dilation=1,
            norm=None#get_norm(norm, out_channels)
        )
        weight_init.c2_msra_fill(self.conv_spatial_1x1)

        if self.rot_1x1_out:
            self.conv_rot_1x1 = GConv1x1(
                rot_1x1=True,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                kernel_rot=self.kernel_rot,
                stride=1,
                padding=0,
                dilation=1,
                norm=None#get_norm(norm, out_channels)
            )
            weight_init.c2_msra_fill(self.conv_rot_1x1)

        self.norm = None
        if norm != None:
            self.norm = get_norm(norm, out_channels)


    def forward(self, x):
        if self.noise_var > 0:
            if self.is_first:
                grid_size = (x.size(2), x.size(3))
                offset = torch.randn(x.size(0), x.size(1) * 2, grid_size[0], grid_size[1], device=x.device) * self.noise_var
                x = self.positional_noise(x, offset, grid_size, 0)
            else:
                x_out = []
                for rot in range(self.kernel_rot):
                    x_rot = x[:, :, rot, :, :]
                    grid_size = (x_rot.size(2), x_rot.size(3))
                    offset = torch.randn(x_rot.size(0), x_rot.size(1) * 2, grid_size[0], grid_size[1], device=x_rot.device) * self.noise_var
                    x_out.append(self.positional_noise(x_rot, offset, grid_size, 0))
                x = torch.stack(x_out, dim=2)

        x = self.conv(x)

        x = self.conv_spatial_1x1(x)

        if self.rot_1x1_out:
            x = self.conv_rot_1x1(x)

        if self.norm != None:
            for rot in range(self.kernel_rot):
                x[:, :, rot, :, :] = self.norm(x[:, :, rot, :, :])

        if self.activation != None:
            x = self.activation(x)

        return x

class PSR4ConvF(PSRConvBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        rot_1x1_out=False,
        noise_var=0,
        stride=1,
        padding=1,
        dilation=1,
        norm=None,
        activation=None
    ):
        super(PSR4ConvF, self).__init__(
            _SRConv=_SR4Conv,
            in_channels=in_channels,
            out_channels=out_channels,
            is_first=True,
            rot_1x1_in=False,
            rot_1x1_out=rot_1x1_out,
            noise_var=noise_var,
            stride=stride,
            padding=padding,
            dilation=dilation,
            norm=norm,
            activation=activation
        )

class PSR8ConvF(PSRConvBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        rot_1x1_out=False,
        noise_var=0,
        stride=1,
        padding=1,
        dilation=1,
        norm=None,
        activation=None
    ):
        super(PSR8ConvF, self).__init__(
            _SRConv=_SR8Conv,
            in_channels=in_channels,
            out_channels=out_channels,
            is_first=True,
            rot_1x1_in=False,
            rot_1x1_out=rot_1x1_out,
            noise_var=noise_var,
            stride=stride,
            padding=padding,
            dilation=dilation,
            norm=norm,
            activation=activation
        )

class PSR4Conv(PSRConvBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        rot_1x1_in=False,
        rot_1x1_out=False,
        noise_var=0,
        stride=1,
        padding=1,
        dilation=1,
        norm=None,
        activation=None
    ):
        super(PSR4Conv, self).__init__(
            _SRConv=_SR4Conv,
            in_channels=in_channels,
            out_channels=out_channels,
            is_first=False,
            rot_1x1_in=rot_1x1_in,
            rot_1x1_out=rot_1x1_out,
            noise_var=noise_var,
            stride=stride,
            padding=padding,
            dilation=dilation,
            norm=norm,
            activation=activation
        )

class PSR8Conv(PSRConvBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        rot_1x1_in=False,
        rot_1x1_out=False,
        noise_var=0,
        stride=1,
        padding=1,
        dilation=1,
        norm=None,
        activation=None
    ):
        super(PSR8Conv, self).__init__(
            _SRConv=_SR8Conv,
            in_channels=in_channels,
            out_channels=out_channels,
            is_first=False,
            rot_1x1_in=rot_1x1_in,
            rot_1x1_out=rot_1x1_out,
            noise_var=noise_var,
            stride=stride,
            padding=padding,
            dilation=dilation,
            norm=norm,
            activation=activation
        )
