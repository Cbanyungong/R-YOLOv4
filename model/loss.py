import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    # Reference: https://amaarora.github.io/2020/06/29/FocalLoss.html
    def __init__(self, FloatTensor, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = FloatTensor([0, alpha])
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


def bbox_xywh_ciou(pred_boxes, target_boxes):
    # Reference: https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/86a370aa2cadea6ba7e5dffb2efc4bacc4c863ea/
    #            utils/box/box_utils.py#L47
    """
    :param pred_boxes: [num_of_objects, 4], boxes predicted by yolo and have been scaled
    :param target_boxes: [num_of_objects, 4], ground truth boxes and have been scaled
    :return: ciou loss
    """
    assert pred_boxes.size() == target_boxes.size()

    # xywh -> xyxy
    pred_boxes = torch.cat(
        [pred_boxes[..., :2] - pred_boxes[..., 2:] / 2,
         pred_boxes[..., :2] + pred_boxes[..., 2:] / 2], dim=-1)
    target_boxes = torch.cat(
        [target_boxes[..., :2] - target_boxes[..., 2:] / 2,
         target_boxes[..., :2] + target_boxes[..., 2:] / 2], dim=-1)

    w1 = pred_boxes[:, 2] - pred_boxes[:, 0]
    h1 = pred_boxes[:, 3] - pred_boxes[:, 1]
    w2 = target_boxes[:, 2] - target_boxes[:, 0]
    h2 = target_boxes[:, 3] - target_boxes[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (pred_boxes[:, 2] + pred_boxes[:, 0]) / 2
    center_y1 = (pred_boxes[:, 3] + pred_boxes[:, 1]) / 2
    center_x2 = (target_boxes[:, 2] + target_boxes[:, 0]) / 2
    center_y2 = (target_boxes[:, 3] + target_boxes[:, 1]) / 2

    inter_max_xy = torch.min(pred_boxes[:, 2:], target_boxes[:, 2:])
    inter_min_xy = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
    out_max_xy = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
    out_min_xy = torch.min(pred_boxes[:, :2], target_boxes[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    u = inter_diag / outer_diag
    iou = inter_area / union
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)

    # alpha is a constant, it don't have gradient
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v)

    ciou_loss = iou - (u + alpha * v)
    ciou_loss = torch.clamp(ciou_loss, min=-1.0, max=1.0)
    return iou, ciou_loss