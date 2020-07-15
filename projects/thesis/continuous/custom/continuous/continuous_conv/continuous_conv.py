# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from functools import lru_cache
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from continuous import _C
from detectron2.layers.wrappers import _NewEmptyTensorOp

import scipy.ndimage
import numpy as np
import torch.nn.functional as F

__all__ = [
    "ContinuousConv",
]

class GaussianLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 sigma):
        super(GaussianLayer, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.weights_init(kernel_size, sigma)

        o = torch.zeros(in_channels, 1, 1, 1)
        self.weight = nn.Parameter(o + self.weight.reshape(1, 1, self.weight.size(0), self.weight.size(1)), requires_grad=False)

    def forward(self, x):
        return F.conv2d(x, self.weight, stride=1, padding=int((self.kernel_size - 1) / 2), groups=self.in_channels)

    def weights_init(self, kernel_size, sigma):
        n = np.zeros((kernel_size, kernel_size))
        m = int((kernel_size - 1) / 2)
        n[m, m] = 1
        k = scipy.ndimage.gaussian_filter(n, sigma=sigma)
        self.weight = torch.from_numpy(k).type(torch.FloatTensor)

class _ContinuousConv(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        offset,
        grid_size,
        shift=1,
        dilation=1,
        groups=1,
        im2col_step=256,
    ):
        if input is not None and input.dim() != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(input.dim())
            )
        ctx.grid_size = grid_size
        ctx.shift = _pair(shift)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(input, weight, offset)

        output = input.new_empty(
            _ContinuousConv._output_size(input, weight, ctx.grid_size)
        )

        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones

        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = _ContinuousConv._cal_im2col_step(input.shape[0], ctx.im2col_step)
            assert (input.shape[0] % cur_im2col_step) == 0, "im2col step must divide batchsize"

            _C.continuous_conv_forward(
                input,
                weight,
                offset,
                output,
                ctx.bufs_[0],
                ctx.bufs_[1],
                ctx.grid_size[1],
                ctx.grid_size[0],
                weight.size(3),
                weight.size(2),
                ctx.shift[1],
                ctx.shift[0],
                ctx.dilation[1],
                ctx.dilation[0],
                ctx.groups,
                cur_im2col_step,
            )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, weight, offset = ctx.saved_tensors

        grad_input = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = _ContinuousConv._cal_im2col_step(input.shape[0], ctx.im2col_step)
            assert (input.shape[0] % cur_im2col_step) == 0, "im2col step must divide batchsize"

            if ctx.needs_input_grad[0]:
                grad_input = torch.zeros_like(input)
                _C.continuous_conv_backward_input(
                    input,
                    offset,
                    grad_output,
                    grad_input,
                    weight,
                    ctx.bufs_[0],
                    ctx.grid_size[1],
                    ctx.grid_size[0],
                    weight.size(3),
                    weight.size(2),
                    ctx.shift[1],
                    ctx.shift[0],
                    ctx.dilation[1],
                    ctx.dilation[0],
                    ctx.groups,
                    cur_im2col_step,
                )

            if ctx.needs_input_grad[1]:
                grad_weight = torch.zeros_like(weight)
                _C.continuous_conv_backward_filter(
                    input,
                    offset,
                    grad_output,
                    grad_weight,
                    ctx.bufs_[0],
                    ctx.bufs_[1],
                    ctx.grid_size[1],
                    ctx.grid_size[0],
                    weight.size(3),
                    weight.size(2),
                    ctx.shift[1],
                    ctx.shift[0],
                    ctx.dilation[1],
                    ctx.dilation[0],
                    ctx.groups,
                    1,
                    cur_im2col_step,
                )

        return grad_input, grad_weight, None, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, grid_size):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = grid_size[d]
            output_size += (in_size,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    "x".join(map(str, output_size))
                )
            )
        return output_size

    @staticmethod
    @lru_cache(maxsize=128)
    def _cal_im2col_step(input_size, default_size):
        """
        Calculate proper im2col step size, which should be divisible by input_size and not larger
        than prefer_size. Meanwhile the step size should be as large as possible to be more
        efficient. So we choose the largest one among all divisors of input_size which are smaller
        than prefer_size.
        :param input_size: input batch size .
        :param default_size: default preferred im2col step size.
        :return: the largest proper step size.
        """
        if input_size <= default_size:
            return input_size
        best_step = 1
        for step in range(2, min(int(math.sqrt(input_size)) + 1, default_size)):
            if input_size % step == 0:
                if input_size // step <= default_size:
                    return input_size // step
                best_step = step

        return best_step

continuous_conv = _ContinuousConv.apply

class ContinuousConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        offset_scale=0.1,
        offset_std=0,
        shift=1,
        dilation=1,
        groups=1,
        bias=False,
        norm=None,
        activation=None,
    ):
        """
        Deformable convolution.

        Arguments are similar to :class:`Conv2D`. Extra arguments:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        """
        super(ContinuousConv, self).__init__()

        assert not bias
        assert in_channels % groups == 0, "in_channels {} cannot be divisible by groups {}".format(
            in_channels, groups
        )
        assert (
            out_channels % groups == 0
        ), "out_channels {} cannot be divisible by groups {}".format(out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.shift = _pair(shift)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.norm = norm
        self.activation = activation
        self.offset_scale = offset_scale
        self.offset_std = offset_std

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size)
        )
        self.bias = None

        # self.gradient_w = nn.Parameter(torch.Tensor([[-1, 0, 1]]), requires_grad=False)
        # self.gradient_h = nn.Parameter(torch.Tensor([[-1], [0], [1]]), requires_grad=False)
        self.gaussian = GaussianLayer(self.in_channels, kernel_size=9, sigma=1)

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x, grid_size):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [x.shape[0], self.weight.shape[0], grid_size[0], grid_size[1]]
            return _NewEmptyTensorOp.apply(x, output_shape)

        if not self.offset_std is None:
            offset = (self.offset_std * torch.randn([x.shape[0], 2 * x.shape[1], grid_size[1], grid_size[0]], device=x.device)).clamp(min=-0.5, max=0.5)
        else:
            offset = torch.zeros([x.shape[0], 2 * x.shape[1], grid_size[1], grid_size[0]], device=x.device)

        x = continuous_conv(
            x,
            self.weight,
            offset,
            grid_size,
            self.shift,
            self.dilation,
            self.groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", offset_scale=" + str(self.offset_scale)
        tmpstr += ", offset_std=" + str(self.offset_std)
        tmpstr += ", shift=" + str(self.shift)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", bias=False"
        return tmpstr
