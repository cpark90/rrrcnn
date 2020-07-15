import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.backbone.backbone import Backbone

from continuous.custom.continuous import PR4ConvF, PR4Conv, PR8ConvF, PR8Conv, PSR4ConvF, PSR4Conv, PSR8ConvF, PSR8Conv

__all__ = [
    "PSRNet",
]

CONV_DICT = {
    "PR4ConvF"  : PR4ConvF,
    "PR4Conv"   : PR4Conv,
    "PR8ConvF"  : PR8ConvF,
    "PR8Conv"   : PR8Conv,
    "PSR4ConvF" : PSR4ConvF,
    "PSR4Conv"  : PSR4Conv,
    "PSR8ConvF" : PSR8ConvF,
    "PSR8Conv"  : PSR8Conv
}

class _PSRNet(Backbone):
    def __init__(
        self,
        _ConvF,
        _Conv,
        input_shape,
        num_classes,
        standard_channels,
        rot_1x1_out,
        noise_var,
        norm="BN"
    ):
        super(_PSRNet, self).__init__()

        in_channels = standard_channels
        out_channels = standard_channels

        self.convf = _ConvF(
            input_shape,
            out_channels,
            rot_1x1_out=rot_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            dilation=1,
            norm=norm
        )
        self.conv1 = _Conv(
            in_channels,
            out_channels,
            rot_1x1_in=rot_1x1_out,
            rot_1x1_out=rot_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            dilation=1,
            norm=norm
        )
        self.conv2 = _Conv(
            in_channels,
            out_channels,
            rot_1x1_in=rot_1x1_out,
            rot_1x1_out=rot_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            dilation=1,
            norm=norm
        )
        self.conv3 = _Conv(
            in_channels,
            out_channels,
            rot_1x1_in=rot_1x1_out,
            rot_1x1_out=rot_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            dilation=1,
            norm=norm
        )
        self.conv4 = _Conv(
            in_channels,
            out_channels,
            rot_1x1_in=rot_1x1_out,
            rot_1x1_out=rot_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            dilation=1,
            norm=norm
        )
        self.conv5 = _Conv(
            in_channels,
            out_channels,
            rot_1x1_in=rot_1x1_out,
            rot_1x1_out=rot_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            dilation=1,
            norm=norm
        )
        self.conv6 = _Conv(
            in_channels,
            out_channels,
            rot_1x1_in=rot_1x1_out,
            rot_1x1_out=rot_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            dilation=1,
            norm=norm
        )
        self.conv7 = _Conv(
            in_channels,
            out_channels,
            rot_1x1_in=rot_1x1_out,
            rot_1x1_out=rot_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            dilation=1,
            norm=norm
        )

        self.kernel_rot = self.convf.kernel_rot
        self.rot_1x1_out = rot_1x1_out

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        outputs = {}

        x = self.convf(x)
        x = F.relu_(x)
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool3d(x, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        x = self.conv2(x)
        x = F.relu_(x)
        x = self.conv3(x)
        x = F.relu_(x)
        x = F.max_pool3d(x, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        x = self.conv4(x)
        x = F.relu_(x)
        x = self.conv5(x)
        x = F.relu_(x)
        x = F.max_pool3d(x, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        x = self.conv6(x)
        x = F.relu_(x)
        x = self.conv7(x)
        x = F.relu_(x)
        x = F.max_pool3d(x, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # x = torch.mean(x, 2)
        # x = x[:, :, 0, :, :]

        x = self.avgpool(x[:, :, 0, :, :])
        x = x.flatten(1)
        x = self.linear(x)
        outputs["linear"] = x

        return outputs


class PSRNet(_PSRNet):
    def __init__(
        self,
        convf_name,
        conv_name,
        input_shape,
        num_classes,
        standard_channels,
        rot_1x1_out,
        noise_var,
        norm
    ):
        super(PSRNet, self).__init__(
            CONV_DICT[convf_name],
            CONV_DICT[conv_name],
            input_shape,
            num_classes,
            standard_channels,
            rot_1x1_out,
            noise_var,
            norm
        )
