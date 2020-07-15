import torch.nn as nn
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init

from detectron2.modeling.backbone.backbone import Backbone
from detectron2.layers import get_norm, Conv2d

__all__ = [
    "CNNNet",
]

class CNNNet(Backbone):
    def __init__(self, input_shape, standard_channels, num_classes=None, norm="BN"):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(CNNNet, self).__init__()
        self.num_classes = num_classes

        in_channels = standard_channels
        out_channels = standard_channels

        self.conv1 = Conv2d(
            in_channels=input_shape,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=get_norm(norm, out_channels)
        )
        self.conv2 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=get_norm(norm, out_channels)
        )
        self.conv3 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=get_norm(norm, out_channels)
        )
        self.conv4 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=get_norm(norm, out_channels)
        )
        self.conv5 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=get_norm(norm, out_channels)
        )
        self.conv6 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=get_norm(norm, out_channels)
        )
        self.conv7 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=get_norm(norm, out_channels)
        )
        self.conv8 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=get_norm(norm, out_channels)
        )


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(out_channels, num_classes)

        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)
        weight_init.c2_msra_fill(self.conv3)
        weight_init.c2_msra_fill(self.conv4)
        weight_init.c2_msra_fill(self.conv5)
        weight_init.c2_msra_fill(self.conv6)
        weight_init.c2_msra_fill(self.conv7)
        weight_init.c2_msra_fill(self.conv8)


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
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.linear(x)
        outputs["linear"] = x

        return outputs

