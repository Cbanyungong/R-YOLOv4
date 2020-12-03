import torch.nn as nn
from model.backbone import Backbone
from model.neck import Neck
from model.head import Head


class Yolo(nn.Module):
    def __init__(self, n_classes=80):
        super().__init__()
        output_ch = (4 + 1 + n_classes) * 3
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head(output_ch, n_classes)

    def forward(self, i, target=None):
        if target is None:
            inference = True
        else:
            inference = False

        d3, d4, d5 = self.backbone(i)
        x20, x13, x6 = self.neck(d5, d4, d3, inference)
        output, loss = self.head(x20, x13, x6, target)
        return output, loss
