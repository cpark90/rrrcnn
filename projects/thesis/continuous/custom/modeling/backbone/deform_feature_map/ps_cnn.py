import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.backbone.backbone import Backbone

from continuous.custom.continuous import PSConvBlock

__all__ = [
    "PSNet",
]

class PSNet(Backbone):
    def __init__(
        self,
        input_shape,
        num_classes,
        standard_channels,
        kernel_type,
        spatial,
        spatial_1x1_out,
        noise_var,
        norm="BN"
    ):
        super(PSNet, self).__init__()

        in_channels = standard_channels
        out_channels = standard_channels

        self.conv1 = PSConvBlock(
            in_channels=input_shape,
            out_channels=out_channels,
            kernel_type=kernel_type,
            spatial=spatial,
            spatial_1x1_in=True,
            spatial_1x1_out=spatial_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            dilation=1,
            norm=norm
        )
        self.conv2 = PSConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_type=kernel_type,
            spatial=spatial,
            spatial_1x1_in=spatial_1x1_out,
            spatial_1x1_out=spatial_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            dilation=1,
            norm=norm
        )
        self.conv3 = PSConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_type=kernel_type,
            spatial=spatial,
            spatial_1x1_in=spatial_1x1_out,
            spatial_1x1_out=spatial_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            dilation=1,
            norm=norm
        )
        self.conv4 = PSConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_type=kernel_type,
            spatial=spatial,
            spatial_1x1_in=spatial_1x1_out,
            spatial_1x1_out=spatial_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            dilation=1,
            norm=norm
        )
        self.conv5 = PSConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_type=kernel_type,
            spatial=spatial,
            spatial_1x1_in=spatial_1x1_out,
            spatial_1x1_out=spatial_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            dilation=1,
            norm=norm
        )
        self.conv6 = PSConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_type=kernel_type,
            spatial=spatial,
            spatial_1x1_in=spatial_1x1_out,
            spatial_1x1_out=spatial_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            dilation=1,
            norm=norm
        )
        self.conv7 = PSConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_type=kernel_type,
            spatial=spatial,
            spatial_1x1_in=spatial_1x1_out,
            spatial_1x1_out=spatial_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            dilation=1,
            norm=norm
        )
        self.conv8 = PSConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_type=kernel_type,
            spatial=spatial,
            spatial_1x1_in=spatial_1x1_out,
            spatial_1x1_out=spatial_1x1_out,
            noise_var=noise_var,
            stride=1,
            padding=1,
            dilation=1,
            norm=norm
        )

        self.num_kernel = self.conv1.num_kernel
        self.spatial = spatial
        self.spatial_1x1_out = spatial_1x1_out

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        outputs = {}

        x = self.conv1(x)
        x = F.relu_(x)
        x = self.conv2(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.conv3(x)
        x = F.relu_(x)
        x = self.conv4(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.conv5(x)
        x = F.relu_(x)
        x = self.conv6(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.conv7(x)
        x = F.relu_(x)
        x = self.conv8(x)
        x = F.relu_(x)

        if self.spatial and not self.spatial_1x1_out:
            x = torch.sum(torch.stack(torch.chunk(x, self.num_kernel, dim=1), dim=4), dim=4)

        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.linear(x)
        outputs["linear"] = x

        return outputs

