import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from detectron2.layers.wrappers import _NewEmptyTensorOp


def oriented_conv_first(x_in, weight, bias=None, stride=1, padding=0, dilation=1, bn=None, act=None):
    weight_1 = weight
    weight_2 = weight.rot90(1, [2, 3])
    weight_3 = weight.rot90(2, [2, 3])
    weight_4 = weight.rot90(3, [2, 3])

    x_1 = F.conv2d(x_in, weight=weight_1, bias=bias, stride=stride, padding=padding, dilation=dilation)
    x_2 = F.conv2d(x_in, weight=weight_2, bias=bias, stride=stride, padding=padding, dilation=dilation)
    x_3 = F.conv2d(x_in, weight=weight_3, bias=bias, stride=stride, padding=padding, dilation=dilation)
    x_4 = F.conv2d(x_in, weight=weight_4, bias=bias, stride=stride, padding=padding, dilation=dilation)

    if bn[0] != None:
        x_1 = bn[0](x_1)
        x_2 = bn[0](x_2)
        x_3 = bn[0](x_3)
        x_4 = bn[0](x_4)

    x_out = torch.stack([x_1, x_2, x_3, x_4], dim=4)

    if act != None:
        x_out = act(x_out)

    return x_out

def oriented_conv(x_tensor, weight, bias=None, stride=1, padding=0, dilation=1, bn=None, act=None):
    x_in = [x_tensor[:, :, :, :, 0], x_tensor[:, :, :, :, 1], x_tensor[:, :, :, :, 2], x_tensor[:, :, :, :, 3]]

    weight_1 = weight
    weight_2 = weight.rot90(1, [2, 3])
    weight_3 = weight.rot90(2, [2, 3])
    weight_4 = weight.rot90(3, [2, 3])

    x_1 = torch.cat([x_in[0], x_in[1], x_in[2], x_in[3]], dim=1)
    x_2 = torch.cat([x_in[1], x_in[2], x_in[3], x_in[0]], dim=1)
    x_3 = torch.cat([x_in[2], x_in[3], x_in[0], x_in[1]], dim=1)
    x_4 = torch.cat([x_in[3], x_in[0], x_in[1], x_in[2]], dim=1)

    x_1 = F.conv2d(x_1, weight=weight_1, bias=bias, stride=stride, padding=padding, dilation=dilation)
    x_2 = F.conv2d(x_2, weight=weight_2, bias=bias, stride=stride, padding=padding, dilation=dilation)
    x_3 = F.conv2d(x_3, weight=weight_3, bias=bias, stride=stride, padding=padding, dilation=dilation)
    x_4 = F.conv2d(x_4, weight=weight_4, bias=bias, stride=stride, padding=padding, dilation=dilation)

    if bn[0] != None:
        x_1 = bn[0](x_1)
        x_2 = bn[0](x_2)
        x_3 = bn[0](x_3)
        x_4 = bn[0](x_4)

    x_out = torch.stack([x_1, x_2, x_3, x_4], dim=4)

    if act != None:
        x_out = act(x_out)

    return x_out

def oriented_conv_last(x_tensor, weight, bias=None, stride=1, padding=0, dilation=1, bn=None, act=None):
    x_in = [x_tensor[:, :, :, :, 0], x_tensor[:, :, :, :, 1], x_tensor[:, :, :, :, 2], x_tensor[:, :, :, :, 3]]

    x = torch.cat([x_in[0], x_in[1], x_in[2], x_in[3]], dim=1)

    x = F.conv2d(x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)

    if bn[0] != None:
        x = bn[0](x)

    if act != None:
        x = act(x)

    return x

def oriented_conv_first_w_d(x_in, weight, bias=None, stride=1, padding=0, bn=None, act=None):
    weight_1 = weight
    weight_2 = weight.rot90(1, [2, 3])
    weight_3 = weight.rot90(2, [2, 3])
    weight_4 = weight.rot90(3, [2, 3])

    x_1 = []
    x_2 = []
    x_3 = []
    x_4 = []

    dilations=[1, 2, 3, 4]
    for dilation in dilations:
        x_1_tmp = F.conv2d(x_in, weight=weight_1, bias=bias, stride=stride, padding=padding+dilation-1, dilation=dilation)
        x_2_tmp = F.conv2d(x_in, weight=weight_2, bias=bias, stride=stride, padding=padding+dilation-1, dilation=dilation)
        x_3_tmp = F.conv2d(x_in, weight=weight_3, bias=bias, stride=stride, padding=padding+dilation-1, dilation=dilation)
        x_4_tmp = F.conv2d(x_in, weight=weight_4, bias=bias, stride=stride, padding=padding+dilation-1, dilation=dilation)

        if bn[0] != None:
            x_1_tmp = bn[0](x_1_tmp)
            x_2_tmp = bn[0](x_2_tmp)
            x_3_tmp = bn[0](x_3_tmp)
            x_4_tmp = bn[0](x_4_tmp)
        x_1.append(x_1_tmp)
        x_2.append(x_2_tmp)
        x_3.append(x_3_tmp)
        x_4.append(x_4_tmp)

    x_1 = torch.mean(torch.stack(tuple(x_1), dim=4), dim=4)
    x_2 = torch.mean(torch.stack(tuple(x_2), dim=4), dim=4)
    x_3 = torch.mean(torch.stack(tuple(x_3), dim=4), dim=4)
    x_4 = torch.mean(torch.stack(tuple(x_4), dim=4), dim=4)

    x_out = torch.stack([x_1, x_2, x_3, x_4], dim=4)

    if act != None:
        x_out = act(x_out)

    return x_out

def oriented_conv_w_d(x_tensor, weight, bias=None, stride=1, padding=0, bn=None, act=None):
    x_in = [x_tensor[:, :, :, :, 0], x_tensor[:, :, :, :, 1], x_tensor[:, :, :, :, 2], x_tensor[:, :, :, :, 3]]

    weight_1 = weight
    weight_2 = weight.rot90(1, [2, 3])
    weight_3 = weight.rot90(2, [2, 3])
    weight_4 = weight.rot90(3, [2, 3])

    x_in_1 = torch.cat([x_in[0], x_in[1], x_in[2], x_in[3]], dim=1)
    x_in_2 = torch.cat([x_in[1], x_in[2], x_in[3], x_in[0]], dim=1)
    x_in_3 = torch.cat([x_in[2], x_in[3], x_in[0], x_in[1]], dim=1)
    x_in_4 = torch.cat([x_in[3], x_in[0], x_in[1], x_in[2]], dim=1)

    x_1 = []
    x_2 = []
    x_3 = []
    x_4 = []

    dilations=[1, 2, 3, 4]
    for dilation in dilations:
        x_1_tmp = F.conv2d(x_in_1, weight=weight_1, bias=bias, stride=stride, padding=padding+dilation-1, dilation=dilation)
        x_2_tmp = F.conv2d(x_in_2, weight=weight_2, bias=bias, stride=stride, padding=padding+dilation-1, dilation=dilation)
        x_3_tmp = F.conv2d(x_in_3, weight=weight_3, bias=bias, stride=stride, padding=padding+dilation-1, dilation=dilation)
        x_4_tmp = F.conv2d(x_in_4, weight=weight_4, bias=bias, stride=stride, padding=padding+dilation-1, dilation=dilation)

        if bn[0] != None:
            x_1_tmp = bn[0](x_1_tmp)
            x_2_tmp = bn[0](x_2_tmp)
            x_3_tmp = bn[0](x_3_tmp)
            x_4_tmp = bn[0](x_4_tmp)
        x_1.append(x_1_tmp)
        x_2.append(x_2_tmp)
        x_3.append(x_3_tmp)
        x_4.append(x_4_tmp)

    x_1 = torch.mean(torch.stack(tuple(x_1), dim=4), dim=4)
    x_2 = torch.mean(torch.stack(tuple(x_2), dim=4), dim=4)
    x_3 = torch.mean(torch.stack(tuple(x_3), dim=4), dim=4)
    x_4 = torch.mean(torch.stack(tuple(x_4), dim=4), dim=4)

    x_out = torch.stack([x_1, x_2, x_3, x_4], dim=4)

    if act != None:
        x_out = act(x_out)

    return x_out

class GConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        groups,
        gconv,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        norm=None,
        activation=None,
    ):
        super(GConv, self).__init__()
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
        self.gconv=gconv

        self.weight = nn.Parameter(torch.Tensor(out_channels, groups * in_channels, *self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):

        output = self.gconv(x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                      bn=[self.norm], act=self.activation)

        return output

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

class R4Conv(GConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=4,
        bias=False,
        norm=None,
        activation=None,
    ):
        super(R4Conv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            groups,
            oriented_conv,
            stride,
            padding,
            dilation,
            bias,
            norm,
            activation)

class R4ConvL(GConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=4,
        bias=False,
        norm=None,
        activation=None,
    ):
        super(R4ConvL, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            groups,
            oriented_conv_last,
            stride,
            padding,
            dilation,
            bias,
            norm,
            activation)

class GTConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        groups,
        gconv,
        stride=1,
        paddings=0,
        dilations=1,
        num_branch=1,
        test_branch_idx=-1,
        bias=False,
        norm=None,
        activation=None,
    ):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.num_branch = num_branch
        self.stride = _pair(stride)
        self.groups = groups
        self.with_bias = bias
        if isinstance(paddings, int):
            paddings = [paddings] * self.num_branch
        if isinstance(dilations, int):
            dilations = [dilations] * self.num_branch
        self.paddings = [_pair(padding) for padding in paddings]
        self.dilations = [_pair(dilation) for dilation in dilations]
        self.test_branch_idx = test_branch_idx
        self.norm = norm
        self.activation = activation
        self.gconv = gconv

        assert len({self.num_branch, len(self.paddings), len(self.dilations)}) == 1

        self.weight = nn.Parameter(torch.Tensor(out_channels, groups * in_channels, *self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, inputs):
        num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
        assert len(inputs) == num_branch

        if inputs[0].numel() == 0:
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    inputs[0].shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [inputs[0].shape[0], self.weight.shape[0]] + output_shape
            return [_NewEmptyTensorOp.apply(input, output_shape) for input in inputs]

        if self.training or self.test_branch_idx == -1:
            outputs = [
                self.gconv(input,
                           weight=self.weight,
                           bias=self.bias,
                           stride=self.stride,
                           padding=padding,
                           dilation=dilation,
                           bn=[self.norm],
                           act=self.activation)
                for input, dilation, padding in zip(inputs, self.dilations, self.paddings)
            ]
        else:
            outputs = [
                self.gconv(inputs[0],
                           weight=self.weight,
                           bias=self.bias,
                           stride=self.stride,
                           padding=self.paddings[self.test_branch_idx],
                           dilation=self.dilations[self.test_branch_idx],
                           bn=[self.norm],
                           act=self.activation)
            ]
        return outputs

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", num_branch=" + str(self.num_branch)
        tmpstr += ", test_branch_idx=" + str(self.test_branch_idx)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", paddings=" + str(self.paddings)
        tmpstr += ", dilations=" + str(self.dilations)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", bias=" + str(self.with_bias)
        return tmpstr

class R4TConv(GTConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        paddings=0,
        dilations=1,
        num_branch = 1,
        test_branch_idx = -1,
        groups=4,
        bias=False,
        norm=None,
        activation=None,
    ):
        super(R4TConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            groups,
            oriented_conv,
            stride,
            paddings,
            dilations,
            num_branch,
            test_branch_idx,
            bias,
            norm,
            activation)

def ginterpolate(gfeatures, scale_factor=2, mode="nearest"):
    gfeatures = [gfeatures[:, :, :, :, idx] for idx in range(gfeatures.shape[4])]

    outputs = []
    for gfeature in gfeatures:
        outputs.append(F.interpolate(gfeature, scale_factor=scale_factor, mode=mode))

    return torch.stack(tuple(outputs), dim=4)