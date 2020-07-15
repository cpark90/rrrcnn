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
    "DefemLayer",
]

class _DefemLayer(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        grid_size,
        shift=0,
        im2col_step=256,
    ):
        if input is not None and input.dim() != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(input.dim())
            )
        ctx.grid_size = _pair(grid_size)
        ctx.shift = _pair(shift)
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(input, offset)

        output = input.new_empty(
            _DefemLayer._output_size(input, ctx.grid_size)
        )

        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = _DefemLayer._cal_im2col_step(input.shape[0], ctx.im2col_step)
            assert (input.shape[0] % cur_im2col_step) == 0, "im2col step must divide batchsize"

            _C.deform_feature_map_forward(
                input,
                offset,
                output,
                ctx.grid_size[1],
                ctx.grid_size[0],
                ctx.shift[1],
                ctx.shift[0],
                1,
                cur_im2col_step,
            )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset = ctx.saved_tensors

        grad_input = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = _DefemLayer._cal_im2col_step(input.shape[0], ctx.im2col_step)
            assert (input.shape[0] % cur_im2col_step) == 0, "im2col step must divide batchsize"

            if ctx.needs_input_grad[0]:
                grad_input = torch.zeros_like(input)
                _C.deform_feature_map_backward_input(
                    input,
                    offset,
                    grad_output,
                    grad_input,
                    ctx.grid_size[1],
                    ctx.grid_size[0],
                    ctx.shift[1],
                    ctx.shift[0],
                    1,
                    cur_im2col_step
                )

        return grad_input, None, None, None, None

    @staticmethod
    def _output_size(input, grid_size):
        channels = input.size(1)
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

defem_layer = _DefemLayer.apply

class DefemLayer(nn.Module):
    def __init__(
        self,
        norm=None,
        activation=None,
    ):
        super(DefemLayer, self).__init__()

        self.norm = norm
        self.activation = activation

    def forward(self, x, offset, grid_size, shift=0):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [x.shape[0], x.shape[1], grid_size[1], grid_size[0]]
            return _NewEmptyTensorOp.apply(x, output_shape)

        x = defem_layer(
            x,
            offset,
            grid_size,
            _pair(shift)
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        tmpstr = ", bias=False"
        return tmpstr

