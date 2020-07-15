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
    "DeoLayer",
]

class _DeoLayer(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        groups=1,
        osize=4,
        gradient_type='normal',
        im2col_step=256,
    ):
        if input is not None and input.dim() != 5:
            raise ValueError(
                "Expected 5D tensor as input, got {}D tensor instead.".format(input.dim())
            )
        ctx.groups = groups
        ctx.osize = osize
        ctx.im2col_step = im2col_step
        ctx.gradient_type = gradient_type

        ctx.save_for_backward(input, offset)

        output = input.new_empty(
            _DeoLayer._output_size(input)
        )

        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = _DeoLayer._cal_im2col_step(input.shape[0], ctx.im2col_step)
            assert (input.shape[0] % cur_im2col_step) == 0, "im2col step must divide batchsize"

            _C.deform_orientation_forward(
                input,
                offset,
                output,
                ctx.groups,
                ctx.osize,
                cur_im2col_step,
            )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset = ctx.saved_tensors

        if ctx.gradient_type == 'big':
            gtype = 1
        elif ctx.gradient_type == 'small':
            gtype = 2
        else:
            gtype = 0

        grad_input = grad_offset = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = _DeoLayer._cal_im2col_step(input.shape[0], ctx.im2col_step)
            assert (input.shape[0] % cur_im2col_step) == 0, "im2col step must divide batchsize"

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                _C.deform_orientation_backward_input(
                    input,
                    offset,
                    grad_output,
                    grad_input,
                    grad_offset,
                    ctx.groups,
                    ctx.osize,
                    cur_im2col_step,
                    gtype,
                )

        return grad_input, grad_offset, None, None, None, None

    @staticmethod
    def _output_size(input):
        output_size = (input.size(0), input.size(1), input.size(2), input.size(3), input.size(4))
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

deo_layer = _DeoLayer.apply

class DeoLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        groups=1,
        osize=4,
        gradient_type='normal',
        norm=None,
        activation=None,
    ):
        """
        Deformable convolution.

        Arguments are similar to :class:`Conv2D`. Extra arguments:

        Args:
            deformable_groups (int): number of groups used in deformable convolution.
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        """
        super(DeoLayer, self).__init__()

        assert in_channels % groups == 0, "in_channels {} cannot be divisible by groups {}".format(
            in_channels, groups
        )

        self.in_channels = in_channels
        self.groups = groups
        self.osize = osize
        self.gradient_type = gradient_type
        self.norm = norm
        self.activation = activation

    def forward(self, x, offset):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]]
            return _NewEmptyTensorOp.apply(x, output_shape)

        x = deo_layer(
            x,
            offset,
            self.groups,
            self.osize,
            self.gradient_type,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", osize=" + str(self.osize)
        tmpstr += ", gradient_type=" + str(self.gradient_type)
        tmpstr += ", bias=False"
        return tmpstr

