import torch.nn.functional as F
import torch.nn as nn
import torch
from tool.yolo import YoloLayer, get_region_boxes


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference=False):
        assert (x.data.dim() == 4)
        # _, _, tH, tW = target_size

        if inference:

            # B = x.data.size(0)
            # C = x.data.size(1)
            # H = x.data.size(2)
            # W = x.data.size(3)

            return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1). \
                expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3),
                       target_size[3] // x.size(3)). \
                contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
        else:
            return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("Convolution block seems to go wrong")
            exit(1)

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv(ch, ch, 1, 1, "mish"))
            resblock_one.append(Conv(ch, ch, 3, 1, "mish"))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample0(nn.Module):
    def __init__(self):
        super(DownSample0, self).__init__()
        self.conv1 = Conv(32, 64, 3, 2, "mish")
        self.conv2 = Conv(64, 64, 1, 1, "mish")

        self.conv4 = Conv(64, 64, 1, 1, "mish")  # 這邊從1延伸出來
        self.conv5 = Conv(64, 32, 1, 1, "mish")
        self.conv6 = Conv(32, 64, 3, 1, "mish")  # 這邊有shortcut從4連過來

        self.conv8 = Conv(64, 64, 1, 1, "mish")

        self.conv10 = Conv(128, 64, 1, 1, "mish")  # 這邊的input是conv2+conv8 所以有128

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        x3 = self.conv4(x1)
        x4 = self.conv5(x3)
        x5 = x3 + self.conv6(x4)
        x6 = self.conv8(x5)

        x6 = torch.cat([x6, x2], dim=1)
        x7 = self.conv10(x6)
        return x7


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, res_blocks):
        super(DownSample, self).__init__()
        self.conv1 = Conv(in_channels, in_channels * 2, 3, 2, "mish")
        self.conv2 = Conv(in_channels * 2, in_channels, 1, 1, "mish")

        self.conv4 = Conv(in_channels * 2, in_channels, 1, 1, "mish")  # 這邊從1延伸出來
        self.resblock = ResBlock(ch=in_channels, nblocks=res_blocks)
        self.conv11 = Conv(in_channels, in_channels, 1, 1, "mish")

        self.conv13 = Conv(in_channels * 2, out_channels, 1, 1, "mish")  # 這邊的input是conv2+conv11 所以有128

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv4(x1)

        r = self.resblock(x3)
        x4 = self.conv11(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv13(x4)
        return x5


class Neck(nn.Module):
    def __init__(self, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv(1024, 512, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        # A maximum pool is applied to a sliding kernel of size say, 1×1, 5×5, 9×9, 13×13.
        # The spatial dimension is preserved. The features maps from different kernel sizes are then
        # concatenated together as output.
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        up = self.upsample1(x7, downsample4.size(), self.inference)
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size(), self.inference)
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class Yolov4Head(nn.Module):
    def __init__(self, output_ch, n_classes, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv(256, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo1 = YoloLayer(
            anchor_mask=[0, 1, 2], num_classes=n_classes,
            anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
            num_anchors=9, stride=8)

        # R -4
        self.conv3 = Conv(128, 256, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = Conv(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv(512, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo2 = YoloLayer(
            anchor_mask=[3, 4, 5], num_classes=n_classes,
            anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
            num_anchors=9, stride=16)

        # R -4
        self.conv11 = Conv(256, 512, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = Conv(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv(1024, 512, 1, 1, 'leaky')
        self.conv17 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo3 = YoloLayer(
            anchor_mask=[6, 7, 8], num_classes=n_classes,
            anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
            num_anchors=9, stride=32)

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)

        if self.inference:
            y1 = self.yolo1(x2)
            y2 = self.yolo2(x10)
            y3 = self.yolo3(x18)

            return get_region_boxes([y1, y2, y3])

        else:
            return [x2, x10, x18]


class Yolo(nn.Module):
    def __init__(self, n_classes=80, inference=False):
        super().__init__()

        output_ch = (4 + 1 + n_classes) * 3

        # backbone
        self.conv0 = Conv(3, 32, 3, 1, "mish")
        self.down1 = DownSample0()
        self.down2 = DownSample(64, 128, 2)
        self.down3 = DownSample(128, 256, 8)
        self.down4 = DownSample(256, 512, 8)
        self.down5 = DownSample(512, 1024, 4)

        # neck
        self.neek = Neck(inference)

        # head
        self.head = Yolov4Head(output_ch, n_classes, inference)

    def forward(self, i):
        i = self.conv0(i)
        d1 = self.down1(i)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neek(d5, d4, d3)

        output = self.head(x20, x13, x6)
        return output
