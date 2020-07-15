import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.tensorboard import SummaryWriter

from imagenet_pretrain import OIConv, RealConv, OIPMConv, GConvF, GConv, DefemLayer, DeoLayer, ATTGLayer, R8OIConvF, R8OIConv, OidConv

from detectron2.layers import FrozenBatchNorm2d, get_norm, ShapeSpec, Conv2d, DeformConv
import fvcore.nn.weight_init as weight_init

MODEL_TYPE = 'oid'
TRAIN_ROTATION = 0 #0 15 45 180
TEST_ROTATION = 0 #0 180
MODEL_NAME = MODEL_TYPE + '_train_' + str(TRAIN_ROTATION) + '_test_' + str(TEST_ROTATION)
NUM_KERNEL = 5
CHANNELS = 20

class Model(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, norm="BN"):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()

        oi_conv_op = OIConv
        real_conv_op = RealConv
        deform_conv_op = DeformConv
        deo_layer_op = DeoLayer
        att_layer_op = ATTGLayer

        self.model_type = MODEL_TYPE

        standard_channels = CHANNELS
        out_channels = standard_channels

        self.conv1_o_real = real_conv_op(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            # offset_scale=0.1,
            offset_std=0.01,
            offset_scale=None,
            # offset_std=None,
            shift=0.5,
            dilation=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.conv2_o_real = real_conv_op(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            # offset_scale=0.1,
            offset_std=0.01,
            offset_scale=None,
            # offset_std=None,
            shift=0.5,
            dilation=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.conv3_o_real = real_conv_op(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            # offset_scale=0.1,
            offset_std=0.01,
            offset_scale=None,
            # offset_std=None,
            shift=0.5,
            dilation=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.conv4_o_real = real_conv_op(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            # offset_scale=0.1,
            offset_std=0.01,
            offset_scale=None,
            # offset_std=None,
            shift=0.5,
            dilation=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )


        in_channels = standard_channels
        out_channels = standard_channels
        self.conv1 = Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.conv2 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        out_channels = 9
        self.conv3 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.conv4 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.conv5 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.conv6 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.conv7 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.conv8 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.conv9 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.conv10 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.conv11 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.conv12 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.conv13 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        self.conv14 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )
        in_channels = 1

        self.conv1_ge_offset = ge_conv_op(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=False,
            # norm=get_norm(norm, out_channels)
        )
        self.conv2_ge_offset = ge_conv_op(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=False,
            # norm=get_norm(norm, out_channels)
        )

        offset_channels = 2 * out_channels
        self.conv1_offset = Conv2d(
            out_channels,
            offset_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2_offset = Conv2d(
            out_channels,
            offset_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3_offset = Conv2d(
            out_channels,
            offset_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv4_offset = Conv2d(
            out_channels,
            offset_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv1_deform = deform_conv_op(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        self.conv2_deform = deform_conv_op(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )


        out_channels = int(standard_channels / 8)

        self.deo_layer = deo_layer_op(
            out_channels,
            osize=4
            # norm=get_norm(norm, out_channels),
        )
        self.att_layer = att_layer_op(
            out_channels
        )

        offset_channels = 2 * out_channels
        self.conv1_defe_offset = real_conv_op(
            in_channels=out_channels,
            out_channels=offset_channels,
            kernel_size=3,
            # offset_scale=0.1,
            offset_std=0.01,
            offset_scale=None,
            # offset_std=None,
            shift=0,
            dilation=1,
            bias=False,
            # norm=get_norm(norm, out_channels)
        )
        self.conv2_defe_offset = real_conv_op(
            in_channels=out_channels,
            out_channels=offset_channels,
            kernel_size=3,
            # offset_scale=0.1,
            offset_std=0.01,
            offset_scale=None,
            # offset_std=None,
            shift=0,
            dilation=1,
            bias=False,
            # norm=get_norm(norm, out_channels)
        )
        self.conv3_defe_offset = real_conv_op(
            in_channels=out_channels,
            out_channels=offset_channels,
            kernel_size=3,
            # offset_scale=0.1,
            offset_std=0.01,
            offset_scale=None,
            # offset_std=None,
            shift=0,
            dilation=1,
            bias=False,
            # norm=get_norm(norm, out_channels)
        )
        self.conv4_defe_offset = real_conv_op(
            in_channels=out_channels,
            out_channels=offset_channels,
            kernel_size=3,
            # offset_scale=0.1,
            offset_std=0.01,
            offset_scale=None,
            # offset_std=None,
            shift=1,
            dilation=1,
            bias=False,
            # norm=get_norm(norm, out_channels)
        )
        self.conv5_defe_offset = real_conv_op(
            in_channels=out_channels,
            out_channels=offset_channels,
            kernel_size=3,
            # offset_scale=0.1,
            offset_std=0.01,
            offset_scale=None,
            # offset_std=None,
            shift=0,
            dilation=1,
            bias=False,
            # norm=get_norm(norm, out_channels)
        )
        self.conv6_defe_offset = real_conv_op(
            in_channels=out_channels,
            out_channels=offset_channels,
            kernel_size=3,
            # offset_scale=0.1,
            offset_std=0.01,
            offset_scale=None,
            # offset_std=None,
            shift=0,
            dilation=1,
            bias=False,
            # norm=get_norm(norm, out_channels)
        )
        self.conv7_defe_offset = real_conv_op(
            in_channels=out_channels,
            out_channels=offset_channels,
            kernel_size=3,
            # offset_scale=0.1,
            offset_std=0.01,
            offset_scale=None,
            # offset_std=None,
            shift=0,
            dilation=1,
            bias=False,
            # norm=get_norm(norm, out_channels)
        )
        self.conv8_defe_offset = real_conv_op(
            in_channels=out_channels,
            out_channels=offset_channels,
            kernel_size=3,
            # offset_scale=0.1,
            offset_std=0.01,
            offset_scale=None,
            # offset_std=None,
            shift=0,
            dilation=1,
            bias=False,
            # norm=get_norm(norm, out_channels)
        )
        self.conv9_defe_offset = real_conv_op(
            in_channels=out_channels,
            out_channels=offset_channels * 2,
            kernel_size=3,
            # offset_scale=0.1,
            offset_std=0.01,
            offset_scale=None,
            # offset_std=None,
            shift=0,
            dilation=1,
            bias=False,
            # norm=get_norm(norm, out_channels)
        )

        kernel_size = NUM_KERNEL
        dilation = 1
        in_channels = standard_channels
        out_channels = standard_channels


        kernel_size = NUM_KERNEL
        in_channels = standard_channels
        out_channels = standard_channels






        nn.init.constant_(self.conv1_defe_offset.weight, 0)
        nn.init.constant_(self.conv2_defe_offset.weight, 0)
        nn.init.constant_(self.conv3_defe_offset.weight, 0)
        nn.init.constant_(self.conv4_defe_offset.weight, 0)
        nn.init.constant_(self.conv5_defe_offset.weight, 0)
        nn.init.constant_(self.conv6_defe_offset.weight, 0)
        nn.init.constant_(self.conv7_defe_offset.weight, 0)
        nn.init.constant_(self.conv8_defe_offset.weight, 0)
        nn.init.constant_(self.conv9_defe_offset.weight, 0)

        weight_init.c2_msra_fill(self.conv1_deform)
        weight_init.c2_msra_fill(self.conv2_deform)
        nn.init.constant_(self.conv1_offset.weight, 0)
        nn.init.constant_(self.conv1_offset.bias, 0)
        nn.init.constant_(self.conv2_offset.weight, 0)
        nn.init.constant_(self.conv2_offset.bias, 0)
        nn.init.constant_(self.conv3_offset.weight, 0)
        nn.init.constant_(self.conv3_offset.bias, 0)
        nn.init.constant_(self.conv4_offset.weight, 0)
        nn.init.constant_(self.conv4_offset.bias, 0)



        weight_init.c2_msra_fill(self.conv1_o_real)
        weight_init.c2_msra_fill(self.conv2_o_real)
        weight_init.c2_msra_fill(self.conv3_o_real)
        weight_init.c2_msra_fill(self.conv4_o_real)

        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)
        weight_init.c2_msra_fill(self.conv3)
        weight_init.c2_msra_fill(self.conv4)
        weight_init.c2_msra_fill(self.conv5)
        weight_init.c2_msra_fill(self.conv6)
        weight_init.c2_msra_fill(self.conv7)
        weight_init.c2_msra_fill(self.conv8)
        weight_init.c2_msra_fill(self.conv9)
        weight_init.c2_msra_fill(self.conv10)
        weight_init.c2_msra_fill(self.conv11)
        weight_init.c2_msra_fill(self.conv12)
        weight_init.c2_msra_fill(self.conv13)
        weight_init.c2_msra_fill(self.conv14)


        self.offset = None

    def forward(self, x):
        check = 4
        rate = 0.1
        # if check == 0:
        #     grid_size = (int(x.shape[2] * 3), int(x.shape[2] * 3))
        #     if self.offset is None:
        #         self.offset = (torch.zeros(x.size(0), x.size(1) * 2, *grid_size) + torch.randn(1, x.size(1) * 2, *grid_size)).cuda(x.device)
        #     offset = self.offset[:x.size(0), :, :, :]
        #     x = self.defem_layer1_n(x, offset, grid_size)
        #
        #     x = self.conv1(x)
        #     grid_size = (int(x.shape[2]), int(x.shape[2]))
        #     offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
        #     x = self.defem_layer1_n(x, offset, grid_size)
        #     x = F.relu_(x)
        #     x = self.conv2(x)
        #     grid_size = (int(x.shape[2]), int(x.shape[2]))
        #     offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
        #     x = self.defem_layer1_n(x, offset, grid_size)
        #     x = F.relu_(x)
        #     x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        #
        #     x = self.conv4(x)
        #     grid_size = (int(x.shape[2]), int(x.shape[2]))
        #     offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
        #     x = self.defem_layer1_n(x, offset, grid_size)
        #     x = F.relu_(x)
        #     x = self.conv6(x)
        #     grid_size = (int(x.shape[2]), int(x.shape[2]))
        #     offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
        #     x = self.defem_layer1_n(x, offset, grid_size)
        #     x = F.relu_(x)
        #     x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        #
        #     x = self.conv7(x)
        #     grid_size = (int(x.shape[2]), int(x.shape[2]))
        #     offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
        #     x = self.defem_layer1_n(x, offset, grid_size)
        #     x = F.relu_(x)
        #     x = self.conv8(x)
        #     grid_size = (int(x.shape[2]), int(x.shape[2]))
        #     offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
        #     x = self.defem_layer1_n(x, offset, grid_size)
        #     x = F.relu_(x)
        #     x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        #
        #     x = self.conv9(x)
        #     grid_size = (int(x.shape[2]), int(x.shape[2]))
        #     offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
        #     x = self.defem_layer1_n(x, offset, grid_size)
        #     x = F.relu_(x)
        #     x = self.conv10(x)
        #     grid_size = (int(x.shape[2]), int(x.shape[2]))
        #     offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
        #     x = self.defem_layer1_n(x, offset, grid_size)
        #     x = F.relu_(x)
        #     x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        # elif check == 1:
        #     x = self.conv1(x)
        #     x = F.relu_(x)
        #     x = self.conv2(x)
        #     x = F.relu_(x)
        #     x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        #
        #     x = self.conv4(x)
        #     x = F.relu_(x)
        #     x = self.conv6(x)
        #     x = F.relu_(x)
        #     x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        #
        #     grid_size = (int(x.shape[2]), int(x.shape[2]))
        #     offset = self.conv1_offset(x)
        #     x = self.defem_layer1_n(x, offset, grid_size)
        #     x = self.conv7(x)
        #     x = F.relu_(x)
        #     x = self.conv8(x)
        #     x = F.relu_(x)
        #     x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        #
        #     grid_size = (int(x.shape[2]), int(x.shape[2]))
        #     offset = self.conv2_offset(x)
        #     x = self.defem_layer1_n(x, offset, grid_size)
        #     x = self.conv9(x)
        #     x = F.relu_(x)
        #     x = self.conv10(x)
        #     x = F.relu_(x)
        #     x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        if self.model_type == 'oid':
            # if rate != 0:
            #     grid_size = (int(x.shape[2]), int(x.shape[2]))
            #     offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
            #     x = self.defem_layer1_n(x, offset, grid_size)
            # out = []
            # for idx in range(4):
            #     x_tmp = torch.cat(
            #         [x[:, :, (0 + idx) % 4, :, :], x[:, :, (1 + idx) % 4, :, :], x[:, :, (2 + idx) % 4, :, :], x[:, :, (3 + idx) % 4, :, :]]
            #         , dim=1)
            #     x_tmp = self.conv1_oid(x_tmp, idx * 2)
            #     out.append(F.max_pool2d(x_tmp, kernel_size=3, stride=2, padding=1))
            # x = torch.stack(out, dim=2)
            rate = 0.1
            if rate != 0:
                grid_size = (int(x.shape[2]), int(x.shape[3]))
                offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
                x = self.defem_layer1_n(x, offset, grid_size)
            x = self.conv0_oid(x)
            x = self.conv1_1x1(x)
            if rate != 0:
                grid_size = (int(x.shape[2]), int(x.shape[3]))
                offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
                x = self.defem_layer1_n(x, offset, grid_size)
            x = self.conv1_oid(x)
            x = self.conv2_1x1(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

            if rate != 0:
                grid_size = (int(x.shape[2]), int(x.shape[3]))
                offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
                x = self.defem_layer1_n(x, offset, grid_size)
            x = self.conv2_oid(x)
            x = self.conv3_1x1(x)
            if rate != 0:
                grid_size = (int(x.shape[2]), int(x.shape[3]))
                offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
                x = self.defem_layer1_n(x, offset, grid_size)
            x = self.conv3_oid(x)
            x = self.conv4_1x1(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

            if rate != 0:
                grid_size = (int(x.shape[2]), int(x.shape[3]))
                offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
                x = self.defem_layer1_n(x, offset, grid_size)
            x = self.conv4_oid(x)
            x = self.conv5_1x1(x)
            if rate != 0:
                grid_size = (int(x.shape[2]), int(x.shape[3]))
                offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
                x = self.defem_layer1_n(x, offset, grid_size)
            x = self.conv5_oid(x)
            x = self.conv6_1x1(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

            if rate != 0:
                grid_size = (int(x.shape[2]), int(x.shape[3]))
                offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
                x = self.defem_layer1_n(x, offset, grid_size)
            x = self.conv6_oid(x)
            x = self.conv7_1x1(x)
            if rate != 0:
                grid_size = (int(x.shape[2]), int(x.shape[3]))
                offset = torch.cat([torch.randn_like(x), torch.randn_like(x)], dim=1) * rate
                x = self.defem_layer1_n(x, offset, grid_size)
            x = self.conv7_oid(x)
            x = self.conv8_1x1(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)


        elif self.model_type == 'normal':
            # grid_size = (int(x.shape[2] * 3), int(x.shape[2] * 3))
            # if self.offset is None:
            #     self.offset = (torch.zeros(x.size(0), x.size(1) * 2, *grid_size) + torch.randn(1, x.size(1) * 2, *grid_size)).cuda(x.device)
            # offset = self.offset[:x.size(0), :, :, :]
            # x = self.defem_layer1_n(x, offset, grid_size)
            x = self.conv1(x)
            x = F.relu_(x)
            x = self.conv2(x)
            x = F.relu_(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

            x = self.conv4(x)
            x = F.relu_(x)
            x = self.conv6(x)
            x = F.relu_(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

            x = self.conv7(x)
            x = F.relu_(x)
            x = self.conv8(x)
            x = F.relu_(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

            x = self.conv9(x)
            x = F.relu_(x)
            x = self.conv10(x)
            x = F.relu_(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        elif self.model_type == 'ge':

        elif self.model_type == 'oin':

        elif self.model_type == 'oig_vanilla':

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.linear(x)

        return x


def train(train_loader, model, criterion, optimizer, epoch, print_freq, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    writer = ProgressSummary(
        writer=writer,
        num_batches=len(val_loader),
        meters={'Loss':losses, 'Accuracy':top1},
        prefix='Train_')
    max_iter = len(train_loader)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda()
        target = target.cuda()
        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)
            writer.summary(max_iter * epoch + i)


def validate(val_loader, model, criterion, epoch, print_freq, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    writer = ProgressSummary(
        writer=writer,
        num_batches=len(val_loader),
        meters={'Loss':losses, 'Accuracy':top1},
        prefix='Val_')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        writer.summary(epoch)
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class ProgressSummary(object):
    def __init__(self, writer, num_batches, meters, prefix=""):
        self.writer = writer
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def summary(self, batch):
        for k, v in self.meters.items():
            self.writer.add_scalar(self.prefix + k, v.avg, batch)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
import time
import os
if __name__ == '__main__':
    batch_size = 256
    lr = 0.001
    epochs = 100
    print_freq = 100

    model_name = MODEL_NAME

    model = Model(1, 10)
    model = model.cuda()#torch.nn.DataParallel(model.cuda())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    cifar_data_path = '/ws/data/open_datasets/classification/cifar10'
    input_size = 32
    train_dataset = datasets.CIFAR10(cifar_data_path, train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.Grayscale(3),
                                       transforms.RandomRotation(TRAIN_ROTATION),
                                       transforms.Grayscale(1),
                                       # transforms.RandomHorizontalFlip(),
                                       # transforms.RandomVerticalFlip(),
                                       transforms.Resize((input_size, input_size)),
                                       transforms.ToTensor(),
                                   ]))
    val_dataset = datasets.CIFAR10(cifar_data_path, train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.Grayscale(3),
                                     transforms.RandomRotation(TEST_ROTATION),
                                     transforms.Grayscale(1),
                                     # transforms.RandomHorizontalFlip(),
                                     transforms.Resize((input_size, input_size)),
                                     transforms.ToTensor(),
                                 ]))

    # mnist_data_path = '/ws/data/open_datasets/classification/mnist'
    # input_size = 32
    # train_dataset = datasets.MNIST(mnist_data_path, train=True, download=True,
    #                                transform=transforms.Compose([
    #                                    transforms.Grayscale(3),
    #                                    # transforms.RandomRotation(30),
    #                                    transforms.Grayscale(1),
    #                                    # transforms.RandomHorizontalFlip(),
    #                                    # transforms.RandomVerticalFlip(),
    #                                    transforms.Resize((input_size, input_size)),
    #                                    transforms.ToTensor(),
    #                                ]))
    # val_dataset = datasets.MNIST(mnist_data_path, train=False, download=True,
    #                              transform=transforms.Compose([
    #                                  transforms.Grayscale(3),
    #                                  # transforms.RandomRotation(90),
    #                                  transforms.Grayscale(1),
    #                                  # transforms.RandomHorizontalFlip(),
    #                                  transforms.Resize((input_size, input_size)),
    #                                  transforms.ToTensor(),
    #                              ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=1)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=1)

    log_path = os.path.join('/ws/data/cpark/logs/classification', model_name)
    writer = SummaryWriter(log_dir=log_path)

    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, print_freq=print_freq, writer=writer)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, print_freq=print_freq, writer=writer)

