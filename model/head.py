from model.utils import *
from model.yololayer import YoloLayer


class Head(nn.Module):
    def __init__(self, output_ch, n_classes):
        super().__init__()
        self.conv1 = Conv(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv(256, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo1 = YoloLayer(
            num_classes=n_classes,
            anchors=[[12, 16], [19, 36], [40, 28]],
            stride=8, scale_x_y=1.2, ignore_thresh=0.7)

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
            num_classes=n_classes,
            anchors=[[36, 75], [76, 55], [72, 146]],
            stride=16, scale_x_y=1.1, ignore_thresh=0.7)

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
            num_classes=n_classes,
            anchors=[[142, 110], [192, 243], [459, 401]],
            stride=32, scale_x_y=1.05, ignore_thresh=0.7)

    def forward(self, input1, input2, input3, target):
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

        y1, loss1 = self.yolo1(x2, target)
        y2, loss2 = self.yolo2(x10, target)
        y3, loss3 = self.yolo3(x18, target)

        return torch.cat([y1, y2, y3], 1), loss1 + loss2 + loss3
