import math
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

import fvcore.nn.weight_init as weight_init

from detectron2.layers.wrappers import _NewEmptyTensorOp
from detectron2.layers import get_norm, Conv2d

from continuous import _C
from continuous.custom.continuous import DefemLayer, GConv1x1


__all__ = [
    "PR4ConvF", "PR4Conv", "PR8ConvF", "PR8Conv", "_GroupConv", "RConv"
]

class _GroupConv(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        kernel_type,
        num_kernel,
        step=0,
        stride=1,
        padding=0,
        dilation=1,
    ):
        if input is not None and input.dim() != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(input.dim())
            )
        ctx.kernel_type = kernel_type
        ctx.num_kernel = num_kernel
        ctx.step = step
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)

        ctx.save_for_backward(input, weight)

        output = input.new_empty(
            _GroupConv._output_size(input, weight, (3, 3), ctx.padding, ctx.dilation, ctx.stride)
        )

        ctx.bufs_ = [input.new_empty(0)]  # columns, ones

        if not input.is_cuda:
            raise NotImplementedError
        else:
            _C.group_conv_forward(
                input,
                weight,
                output,
                ctx.bufs_[0],
                ctx.kernel_type,
                ctx.num_kernel,
                ctx.stride[1],
                ctx.stride[0],
                ctx.padding[1],
                ctx.padding[0],
                ctx.dilation[1],
                ctx.dilation[0],
                ctx.step,
            )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        grad_input = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:

            if ctx.needs_input_grad[0]:
                grad_input = torch.zeros_like(input)
                _C.group_conv_backward_input(
                    input,
                    grad_output,
                    grad_input,
                    weight,
                    ctx.bufs_[0],
                    ctx.kernel_type,
                    ctx.num_kernel,
                    ctx.stride[1],
                    ctx.stride[0],
                    ctx.padding[1],
                    ctx.padding[0],
                    ctx.dilation[1],
                    ctx.dilation[0],
                    ctx.step,
                )

            if ctx.needs_input_grad[1]:
                grad_weight = torch.zeros_like(weight)
                _C.group_conv_backward_filter(
                    input,
                    grad_output,
                    grad_weight,
                    ctx.bufs_[0],
                    ctx.kernel_type,
                    ctx.num_kernel,
                    ctx.stride[1],
                    ctx.stride[0],
                    ctx.padding[1],
                    ctx.padding[0],
                    ctx.dilation[1],
                    ctx.dilation[0],
                    ctx.step,
                    1,
                )

        return grad_input, grad_weight, None, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, kernel_size, padding, dilation, stride):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (kernel_size[d] - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    "x".join(map(str, output_size))
                )
            )
        return output_size


pr_conv = _GroupConv.apply

class RConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_type,
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
        super(RConv, self).__init__()

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
        self.out_channels = out_channels
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.norm = norm
        self.activation = activation

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.num_kernel), requires_grad=True
        )

    def forward(self, x_tensor, rot):
        # x_tensor is zero, return empty
        if x_tensor.numel() == 0:
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x_tensor.shape[-2:], self.padding, self.dilation, (3, 3), self.stride
                )
            ]
            output_shape = [x_tensor.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x_tensor, output_shape)

        step = int(8 / self.kernel_rot)

        x_rot = pr_conv(
            x_tensor,
            self.weight,
            self.kernel_type,
            self.num_kernel,
            step * rot,
            self.stride,
            self.padding,
            self.dilation)

        # batch norm is same for all rotation.
        if self.norm != None:
            x_rot = self.norm(x_rot)
        if self.activation is not None:
            x_rot = self.activation(x_rot)
        return x_rot

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_type=" + str(self.kernel_type)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", bias=False"
        return tmpstr

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
        self.out_channels = out_channels
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

            x_rot = pr_conv(
                x_rot,
                weight,
                self.kernel_type,
                self.num_kernel,
                step * rot,
                self.stride,
                self.padding,
                self.dilation)

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

class _R4Conv(GConv):
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
            norm = get_norm(norm, out_channels)

        super(_R4Conv, self).__init__(
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

class _R8Conv(GConv):
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
            norm = get_norm(norm, out_channels)

        super(_R8Conv, self).__init__(
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


class PRConvBlock(nn.Module):
    def __init__(
            self,
            _RConv,
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
        super(PRConvBlock, self).__init__()

        self.conv = _RConv(
            in_channels=in_channels,
            out_channels=out_channels,
            is_first=is_first,
            rot_1x1_in=rot_1x1_in,
            stride=stride,
            padding=padding,
            dilation=dilation,
            norm=None,#norm,
            activation=F.relu
        )

        self.is_first = is_first
        self.rot_1x1_out = rot_1x1_out
        self.noise_var = noise_var
        self.kernel_rot = self.conv.kernel_rot
        self.activation = activation

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

        if self.noise_var > 0:
            self.positional_noise = DefemLayer()

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
                # norm=get_norm(norm, out_channels)
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

        if self.rot_1x1_out:
            x = self.conv_rot_1x1(x)

        if self.norm != None:
            for rot in range(self.kernel_rot):
                x[:, :, rot, :, :] = self.norm(x[:, :, rot, :, :])

        if self.activation != None:
            x = self.activation(x)

        return x

class PR4ConvF(PRConvBlock):
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
        super(PR4ConvF, self).__init__(
            _RConv=_R4Conv,
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

class PR8ConvF(PRConvBlock):
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
        super(PR8ConvF, self).__init__(
            _RConv=_R8Conv,
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

class PR4Conv(PRConvBlock):
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
        super(PR4Conv, self).__init__(
            _RConv=_R4Conv,
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

class PR8Conv(PRConvBlock):
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
        super(PR8Conv, self).__init__(
            _RConv=_R8Conv,
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
