import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from detectron2.layers.wrappers import _NewEmptyTensorOp

from ..spatial_conv import _SpatialConv

__all__ = [
    "R4ConvF", "R4Conv",#"R4ConvL", "R4TConv", "ginterpolate"
]

spatial_conv = _SpatialConv.apply

class GConvF(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_type="3x3",
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        norm=None,
        activation=None,
    ):
        """
            kernel_type (int): skel, 3x3
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        """
        super(GConvF, self).__init__()

        assert not bias

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.norm = norm
        self.activation = activation

        if kernel_type == "skel":
            self.kernel_type = 0
            self.num_kernel = 5
        elif kernel_type == "3x3":
            self.kernel_type = 1
            self.num_kernel = 9
        else:
            self.kernel_type = 2
            self.num_kernel = 3

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.num_kernel), requires_grad=True
        )
        self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x):
        if x.numel() == 0:
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, (3, 3), self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape + [4]
            return _NewEmptyTensorOp.apply(x, output_shape)

        x_1 = spatial_conv(x, self.weight, self.kernel_type, self.num_kernel, 0, self.stride, self.padding, self.dilation).sum(0)
        x_2 = spatial_conv(x, self.weight, self.kernel_type, self.num_kernel, 2, self.stride, self.padding, self.dilation).sum(0)
        x_3 = spatial_conv(x, self.weight, self.kernel_type, self.num_kernel, 4, self.stride, self.padding, self.dilation).sum(0)
        x_4 = spatial_conv(x, self.weight, self.kernel_type, self.num_kernel, 6, self.stride, self.padding, self.dilation).sum(0)

        if self.norm != None:
            x_1 = self.norm(x_1)
            x_2 = self.norm(x_2)
            x_3 = self.norm(x_3)
            x_4 = self.norm(x_4)

        x_out = torch.stack([x_1, x_2, x_3, x_4], dim=4)

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


class GConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_type="3x3",
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            norm=None,
            activation=None,
            with_1x1=False,
    ):
        """
            kernel_type (int): skel, ex-skel, 3x3
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        """
        super(GConv, self).__init__()

        assert not bias

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.norm = norm
        self.activation = activation
        self.with_1x1 = with_1x1

        in_channels = in_channels if with_1x1 else 4 * in_channels

        if kernel_type == "skel":
            self.kernel_type = 0
            self.num_kernel = 5
        elif kernel_type == "3x3":
            self.kernel_type = 1
            self.num_kernel = 9
        else:
            self.kernel_type = 2
            self.num_kernel = 3

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.num_kernel), requires_grad=True
        )
        self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x_tensor):
        if x_tensor.numel() == 0:
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x_tensor.shape[-2:], self.padding, self.dilation, (3, 3), self.stride
                )
            ]
            output_shape = [x_tensor.shape[0], self.weight.shape[0]] + output_shape + [4]
            return _NewEmptyTensorOp.apply(x_tensor, output_shape)

        if self.with_1x1:
            x_1 = x_tensor[:, :, :, :, 0]
            x_2 = x_tensor[:, :, :, :, 1]
            x_3 = x_tensor[:, :, :, :, 2]
            x_4 = x_tensor[:, :, :, :, 3]
        else:
            x_in = [x_tensor[:, :, :, :, 0], x_tensor[:, :, :, :, 1], x_tensor[:, :, :, :, 2], x_tensor[:, :, :, :, 3]]
            x_1 = torch.cat([x_in[0], x_in[1], x_in[2], x_in[3]], dim=1)
            x_2 = torch.cat([x_in[1], x_in[2], x_in[3], x_in[0]], dim=1)
            x_3 = torch.cat([x_in[2], x_in[3], x_in[0], x_in[1]], dim=1)
            x_4 = torch.cat([x_in[3], x_in[0], x_in[1], x_in[2]], dim=1)

        x_1 = spatial_conv(x_1, self.weight, self.kernel_type, self.num_kernel, 0, self.stride, self.padding, self.dilation).sum(0)
        x_2 = spatial_conv(x_2, self.weight, self.kernel_type, self.num_kernel, 2, self.stride, self.padding, self.dilation).sum(0)
        x_3 = spatial_conv(x_3, self.weight, self.kernel_type, self.num_kernel, 4, self.stride, self.padding, self.dilation).sum(0)
        x_4 = spatial_conv(x_4, self.weight, self.kernel_type, self.num_kernel, 6, self.stride, self.padding, self.dilation).sum(0)

        if self.norm != None:
            x_1 = self.norm(x_1)
            x_2 = self.norm(x_2)
            x_3 = self.norm(x_3)
            x_4 = self.norm(x_4)

        x_out = torch.stack([x_1, x_2, x_3, x_4], dim=4)

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


class GConvL(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        groups,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        norm=None,
        activation=None,
    ):
        super(GConvL, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.groups = groups
        self.with_bias = bias
        self.padding = padding
        self.dilations = dilation
        self.norm = norm
        self.activation = activation

        self.weight = nn.Parameter(torch.Tensor(out_channels, groups * in_channels, *self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x_tensor):
        x_in = [x_tensor[:, :, :, :, 0], x_tensor[:, :, :, :, 1], x_tensor[:, :, :, :, 2], x_tensor[:, :, :, :, 3]]

        x = torch.cat([x_in[0], x_in[1], x_in[2], x_in[3]], dim=1)

        x = F.conv2d(x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilations)

        if self.norm != None:
            x = self.norm(x)

        if self.activation != None:
            x = self.activation(x)

        return x

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilations)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", bias=" + str(self.with_bias)
        return tmpstr

class R4ConvF(GConvF):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_type="3x3",
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        norm=None,
        activation=None,
    ):
        super(R4ConvF, self).__init__(
            in_channels,
            out_channels,
            kernel_type,
            stride,
            padding,
            dilation,
            bias,
            norm,
            activation)

class R4Conv(GConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_type="3x3",
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        norm=None,
        activation=None,
        with_1x1=False,
    ):
        super(R4Conv, self).__init__(
            in_channels,
            out_channels,
            kernel_type,
            stride,
            padding,
            dilation,
            bias,
            norm,
            activation,
            with_1x1)

# class R4ConvL(GConvL):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride=1,
#         padding=0,
#         dilation=1,
#         groups=4,
#         bias=False,
#         norm=None,
#         activation=None,
#     ):
#         super(R4ConvL, self).__init__(
#             in_channels,
#             out_channels,
#             kernel_size,
#             groups,
#             stride,
#             padding,
#             dilation,
#             bias,
#             norm,
#             activation)
#
# class GTConv(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         groups,
#         stride=1,
#         paddings=0,
#         dilations=1,
#         num_branch=1,
#         test_branch_idx=-1,
#         bias=False,
#         norm=None,
#         activation=None,
#     ):
#         super(GTConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _pair(kernel_size)
#         self.num_branch = num_branch
#         self.stride = _pair(stride)
#         self.groups = groups
#         self.with_bias = bias
#         if isinstance(paddings, int):
#             paddings = [paddings] * self.num_branch
#         if isinstance(dilations, int):
#             dilations = [dilations] * self.num_branch
#         self.paddings = [_pair(padding) for padding in paddings]
#         self.dilations = [_pair(dilation) for dilation in dilations]
#         self.test_branch_idx = test_branch_idx
#         self.norm = norm
#         self.activation = activation
#
#         assert len({self.num_branch, len(self.paddings), len(self.dilations)}) == 1
#
#         self.weight = nn.Parameter(torch.Tensor(out_channels, groups * in_channels, *self.kernel_size))
#
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.bias = None
#
#         nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
#         if self.bias is not None:
#             nn.init.constant_(self.bias, 0)
#
#     def forward(self, inputs):
#         num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
#         assert len(inputs) == num_branch
#
#         if inputs[0].numel() == 0:
#             output_shape = [
#                 (i + 2 * p - (di * (k - 1) + 1)) // s + 1
#                 for i, p, di, k, s in zip(
#                     inputs[0].shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
#                 )
#             ]
#             output_shape = [inputs[0].shape[0], self.weight.shape[0]] + output_shape
#             return [_NewEmptyTensorOp.apply(input, output_shape) for input in inputs]
#
#         if self.training or self.test_branch_idx == -1:
#             outputs = []
#             for x_tensor, dilation, padding in zip(inputs, self.dilations, self.paddings):
#                 x_in = [x_tensor[:, :, :, :, 0], x_tensor[:, :, :, :, 1], x_tensor[:, :, :, :, 2],
#                         x_tensor[:, :, :, :, 3]]
#
#                 weight_1 = self.weight
#                 weight_2 = self.weight.rot90(1, [2, 3])
#                 weight_3 = self.weight.rot90(2, [2, 3])
#                 weight_4 = self.weight.rot90(3, [2, 3])
#
#                 x_1 = torch.cat([x_in[0], x_in[1], x_in[2], x_in[3]], dim=1)
#                 x_2 = torch.cat([x_in[1], x_in[2], x_in[3], x_in[0]], dim=1)
#                 x_3 = torch.cat([x_in[2], x_in[3], x_in[0], x_in[1]], dim=1)
#                 x_4 = torch.cat([x_in[3], x_in[0], x_in[1], x_in[2]], dim=1)
#
#                 x_1 = F.conv2d(x_1, weight=weight_1, bias=self.bias, stride=self.stride, padding=padding,
#                                dilation=dilation)
#                 x_2 = F.conv2d(x_2, weight=weight_2, bias=self.bias, stride=self.stride, padding=padding,
#                                dilation=dilation)
#                 x_3 = F.conv2d(x_3, weight=weight_3, bias=self.bias, stride=self.stride, padding=padding,
#                                dilation=dilation)
#                 x_4 = F.conv2d(x_4, weight=weight_4, bias=self.bias, stride=self.stride, padding=padding,
#                                dilation=dilation)
#
#                 if self.norm != None:
#                     x_1 = self.norm(x_1)
#                     x_2 = self.norm(x_2)
#                     x_3 = self.norm(x_3)
#                     x_4 = self.norm(x_4)
#
#                 x_out = torch.stack([x_1, x_2, x_3, x_4], dim=4)
#
#                 if self.activation != None:
#                     x_out = self.activation(x_out)
#                 outputs.append(x_out)
#
#         else:
#             outputs = []
#             x_tensor = inputs[0]
#             x_in = [x_tensor[:, :, :, :, 0], x_tensor[:, :, :, :, 1], x_tensor[:, :, :, :, 2],
#                     x_tensor[:, :, :, :, 3]]
#
#             weight_1 = self.weight
#             weight_2 = self.weight.rot90(1, [2, 3])
#             weight_3 = self.weight.rot90(2, [2, 3])
#             weight_4 = self.weight.rot90(3, [2, 3])
#
#             x_1 = torch.cat([x_in[0], x_in[1], x_in[2], x_in[3]], dim=1)
#             x_2 = torch.cat([x_in[1], x_in[2], x_in[3], x_in[0]], dim=1)
#             x_3 = torch.cat([x_in[2], x_in[3], x_in[0], x_in[1]], dim=1)
#             x_4 = torch.cat([x_in[3], x_in[0], x_in[1], x_in[2]], dim=1)
#
#             x_1 = F.conv2d(x_1, weight=weight_1, bias=self.bias, stride=self.stride, padding=self.paddings[self.test_branch_idx],
#                            dilation=self.dilations[self.test_branch_idx])
#             x_2 = F.conv2d(x_2, weight=weight_2, bias=self.bias, stride=self.stride, padding=self.paddings[self.test_branch_idx],
#                            dilation=self.dilations[self.test_branch_idx])
#             x_3 = F.conv2d(x_3, weight=weight_3, bias=self.bias, stride=self.stride, padding=self.paddings[self.test_branch_idx],
#                            dilation=self.dilations[self.test_branch_idx])
#             x_4 = F.conv2d(x_4, weight=weight_4, bias=self.bias, stride=self.stride, padding=self.paddings[self.test_branch_idx],
#                            dilation=self.dilations[self.test_branch_idx])
#
#             if self.norm != None:
#                 x_1 = self.norm(x_1)
#                 x_2 = self.norm(x_2)
#                 x_3 = self.norm(x_3)
#                 x_4 = self.norm(x_4)
#
#             x_out = torch.stack([x_1, x_2, x_3, x_4], dim=4)
#
#             if self.activation != None:
#                 x_out = self.activation(x_out)
#             outputs.append(x_out)
#
#         return outputs
#
#     def extra_repr(self):
#         tmpstr = "in_channels=" + str(self.in_channels)
#         tmpstr += ", out_channels=" + str(self.out_channels)
#         tmpstr += ", kernel_size=" + str(self.kernel_size)
#         tmpstr += ", num_branch=" + str(self.num_branch)
#         tmpstr += ", test_branch_idx=" + str(self.test_branch_idx)
#         tmpstr += ", stride=" + str(self.stride)
#         tmpstr += ", paddings=" + str(self.paddings)
#         tmpstr += ", dilations=" + str(self.dilations)
#         tmpstr += ", groups=" + str(self.groups)
#         tmpstr += ", bias=" + str(self.with_bias)
#         return tmpstr
#
# class R4TConv(GTConv):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride=1,
#         paddings=0,
#         dilations=1,
#         num_branch = 1,
#         test_branch_idx = -1,
#         groups=4,
#         bias=False,
#         norm=None,
#         activation=None,
#     ):
#         super(R4TConv, self).__init__(
#             in_channels,
#             out_channels,
#             kernel_size,
#             groups,
#             stride,
#             paddings,
#             dilations,
#             num_branch,
#             test_branch_idx,
#             bias,
#             norm,
#             activation)
#
# def ginterpolate(gfeatures, scale_factor=2, mode="nearest"):
#     gfeatures = [gfeatures[:, :, :, :, idx] for idx in range(gfeatures.shape[4])]
#
#     outputs = []
#     for gfeature in gfeatures:
#         outputs.append(F.interpolate(gfeature, scale_factor=scale_factor, mode=mode))
#
#     return torch.stack(tuple(outputs), dim=4)