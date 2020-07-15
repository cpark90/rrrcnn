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

__all__ = [
    "SkeletonConv", "R8ConvF", "R8Conv"
]

class _SkeletonConv(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        dilation=1,
        step=0,
        im2col_step=256,
    ):
        if input is not None and input.dim() != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(input.dim())
            )
        ctx.dilation = dilation
        ctx.step = step
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(input, weight)

        output = input.new_empty(
            _SkeletonConv._output_size(input, weight)
        )

        ctx.bufs_ = [input.new_empty(0)]  # columns, ones

        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = _SkeletonConv._cal_im2col_step(input.shape[0], ctx.im2col_step)
            assert (input.shape[0] % cur_im2col_step) == 0, "im2col step must divide batchsize"

            _C.skeleton_conv_forward(
                input,
                weight,
                output,
                ctx.bufs_[0],
                weight.size(2),
                ctx.dilation,
                ctx.step,
                cur_im2col_step,
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
            cur_im2col_step = _SkeletonConv._cal_im2col_step(input.shape[0], ctx.im2col_step)
            assert (input.shape[0] % cur_im2col_step) == 0, "im2col step must divide batchsize"

            if ctx.needs_input_grad[0]:
                grad_input = torch.zeros_like(input)
                _C.skeleton_conv_backward_input(
                    input,
                    grad_output,
                    grad_output,
                    grad_input,
                    weight,
                    ctx.bufs_[0],
                    weight.size(2),
                    ctx.dilation,
                    ctx.step,
                    cur_im2col_step,
                )

            if ctx.needs_input_grad[1]:
                grad_weight = torch.zeros_like(weight)
                _C.skeleton_conv_backward_filter(
                    input,
                    grad_output,
                    grad_output,
                    grad_weight,
                    ctx.bufs_[0],
                    weight.size(2),
                    ctx.dilation,
                    ctx.step,
                    1,
                    cur_im2col_step,
                )

        return grad_input, grad_weight, None, None, None

    @staticmethod
    def _output_size(input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels, input.size(2), input.size(3))
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

skeleton_conv = _SkeletonConv.apply

class SkeletonConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        step=0,
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
        super(SkeletonConv, self).__init__()

        assert not bias

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.step = step
        self.norm = norm
        self.activation = activation

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.kernel_size)
        )
        self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [x.shape[0], self.weight.shape[0], x.shape[2], x.shape[3]]
            return _NewEmptyTensorOp.apply(x, output_shape)

        x = skeleton_conv(
            x,
            self.weight,
            self.dilation,
            self.step
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
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", bias=False"
        return tmpstr


class R8ConvF(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dilation=1,
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
        super(R8ConvF, self).__init__()

        assert not bias

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.norm = norm
        self.activation = activation

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.kernel_size)
        )
        self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [x.shape[0], self.weight.shape[0], x.shape[2], x.shape[3], 8]
            return _NewEmptyTensorOp.apply(x, output_shape)

        out = []
        for idx in range(8):
            x_tmp = skeleton_conv(x, self.weight, self.dilation, idx)
            if self.norm is not None:
                x_tmp = self.norm(x_tmp)
            out.append(x_tmp)

        x_out = torch.stack(out, dim=2)
        if self.activation is not None:
            x_out = self.activation(x_out)

        return x_out

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", bias=False"
        return tmpstr

class R8Conv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dilation=1,
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
        super(R8Conv, self).__init__()

        assert not bias

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.norm = norm
        self.activation = activation

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, 8 * in_channels, self.kernel_size)
        )
        self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [x.shape[0], self.weight.shape[0], 8, x.shape[2], x.shape[3]]
            return _NewEmptyTensorOp.apply(x, output_shape)

        out = []
        for idx in range(8):
            x_tmp = torch.cat([x[:, :, (i + idx) % 8, :, :] for i in range(8)], dim=1)
            x_tmp = skeleton_conv(x_tmp, self.weight, self.dilation, idx)
            if self.norm is not None:
                x_tmp = self.norm(x_tmp)
            out.append(x_tmp)

        x_out = torch.stack(out, dim=2)
        if self.activation is not None:
            x_out = self.activation(x_out)

        return x_out

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", bias=False"
        return tmpstr

