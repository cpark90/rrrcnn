# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from functools import lru_cache
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from continuous import _C
from detectron2.layers import get_norm
from detectron2.layers.wrappers import _NewEmptyTensorOp

__all__ = [
    "SpatialConv", "_SpatialConv"
]

class _SpatialLayer(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
    ):
        if input is not None and input.dim() != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(input.dim())
            )
        ctx.kernel_size = _pair(kernel_size)
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)

        ctx.save_for_backward(input)

        output = input.new_empty(
            _SpatialConv._output_size(input, ctx.kernel_size, ctx.padding, ctx.dilation, ctx.stride)
        )

        if not input.is_cuda:
            raise NotImplementedError
        else:
            _C.hadamard_layer_forward(
                input,
                output,
                ctx.kernel_size[1],
                ctx.kernel_size[0],
                ctx.stride[1],
                ctx.stride[0],
                ctx.padding[1],
                ctx.padding[0],
                ctx.dilation[1],
                ctx.dilation[0],
            )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:

            if ctx.needs_input_grad[0]:
                grad_input = torch.zeros_like(input)
                _C.hadamard_layer_backward_input(
                    input,
                    grad_output,
                    grad_input,
                    ctx.kernel_size[1],
                    ctx.kernel_size[0],
                    ctx.stride[1],
                    ctx.stride[0],
                    ctx.padding[1],
                    ctx.padding[0],
                    ctx.dilation[1],
                    ctx.dilation[0],
                )

        return grad_input, None, None, None, None

    @staticmethod
    def _output_size(input, kernel_size, padding, dilation, stride):
        channels = input.size(1)
        output_size = (kernel_size[0] * kernel_size[1], input.size(0), channels)
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

class _SpatialConv(Function):
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
            _SpatialConv._output_size(input, weight, (3, 3), ctx.padding, ctx.dilation, ctx.stride)
        )

        ctx.bufs_ = [input.new_empty(0)]  # columns, ones

        if not input.is_cuda:
            raise NotImplementedError
        else:
            _C.hadamard_conv_forward(
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
                _C.hadamard_conv_backward_input(
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
                _C.hadamard_conv_backward_filter(
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
        output_size = (weight.size(2), input.size(0), channels)
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

spatial_conv = _SpatialConv.apply

class SpatialConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_type,
        spatial,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        norm=None,
        activation=None,
    ):
        """
        Deformable convolution.

        Arguments are similar to :class:`Conv2D`. Extra arguments:

        Args:
            kernel_type (int): skel, ex-skel, 3x3
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        """
        super(SpatialConv, self).__init__()

        assert not bias

        self.in_channels = in_channels
        self.out_channels = out_channels * self.num_kernel if spatial else out_channels
        self.spatial = spatial
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        if norm is not None:
            self.norm = get_norm(norm, self.out_channels)
        else:
            self.norm = None
        self.activation = activation

        if kernel_type == 0:
            self.num_kernel = 5
        elif kernel_type == 1:
            self.num_kernel = 9
        else:
            self.num_kernel = 3

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.num_kernel)
        )
        self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x, step=0):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, (3, 3), self.stride
                )
            ]
            out_channels = self.weight.shape[0]
            if self.spatial:
                out_channels = out_channels * self.num_kernel
            output_shape = [x.shape[0], out_channels] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)

        x = spatial_conv(
            x,
            self.weight,
            self.kernel_type,
            self.num_kernel,
            step,
            self.stride,
            self.padding,
            self.dilation
        )
        if self.spatial:
            out = []
            for idx in range(x.size(0)):
                out.append(x[idx, :, :, :, :])
            x = torch.cat(out, dim=1)
        else:
            x = torch.sum(x, dim=0)

        if self.norm is not None:
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

