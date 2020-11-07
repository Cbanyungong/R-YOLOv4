import numpy as np
import torch.nn as nn
import torch


def get_region_boxes(boxes_and_confs):
    boxes_list = []
    confs_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)

    return [boxes, confs]


def iou(box1, box2):
    t1_x1, t1_y1, t1_x2, t1_y2 = box1
    t2_x1, t2_y1, t2_x2, t2_y2 = box2

    # 不知道為什麼有些值真的會小於0
    if t1_x1 < 0 or t1_x2 < 0 or t1_y1 < 0 or t1_y2 < 0 or t2_x1 < 0 or t2_x2 < 0 or t2_y1 < 0 or t2_y2 < 0:
        return 1

    interX = min(t1_x2, t2_x2) - max(t1_x1, t2_x1)
    interY = min(t1_y2, t2_y2) - max(t1_y1, t2_y1)

    if interX <= 0 or interY <= 0:
        return 0.0

    inter_area = interX * interY
    union_area = (t1_x2 - t1_x1) * (t1_y2 - t1_y1) + (t2_x2 - t2_x1) * (t2_y2 - t2_y1) - inter_area

    return inter_area / union_area


def nms(boxes, confs, nms_thresh=0.5):
    dict_of_box = []
    order = np.argsort(confs)
    boxes = boxes[order]
    for i in range(len(order)):
        dict_of_box.append(tuple([order[i], boxes[i]]))

    # nms algorithm
    keep = []
    while len(dict_of_box) > 0:
        box = dict_of_box.pop()

        # box looks like-> [(0, array([0.18615767, 0.22766608, 0.748049  , 0.7332627 ], dtype=float32)),
        #                   (1, array([0.19176558, 0.23242362, 0.73947287, 0.7266734 ], dtype=float32)), ...
        keep.append(box[0])

        box_that_can_be_kept = []
        for b in dict_of_box:
            if iou(box[1], b[1]) < nms_thresh:
                box_that_can_be_kept.append(b)

        dict_of_box = box_that_can_be_kept

    return np.array(keep)


def yolo_forward(output, num_classes, anchors, num_anchors, scale_x_y):
    """
    :param output: output from yolo layer
    :param conf_thresh: threshold for non-maximum-suppress
    :param num_classes: number of classes
    :param anchors: anchors corresponding to mask
    :param num_anchors: number of classed will be detected in each cell
    :param scale_x_y: still don't know
    :return: boxes coordinate and confidence(box_confidence * box_class_probs)
    """
    # Tensors for cuda support
    FloatTensor = torch.cuda.FloatTensor if output.is_cuda else torch.FloatTensor

    # output.shape-> [num_samples, num_anchors * (num_classes + 5), grid_size, grid_size]
    num_samples = output.size(0)
    grid_size = output.size(2)

    # prediction.shape-> torch.Size([1, num_anchors, grid_size, grid_size, num_classes + 5])
    prediction = (
        output.view(num_samples, num_anchors, num_classes + 5, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
    )

    bx = torch.sigmoid(prediction[..., 0:1]) * scale_x_y - 0.5 * (scale_x_y - 1)
    by = torch.sigmoid(prediction[..., 1:2]) * scale_x_y - 0.5 * (scale_x_y - 1)
    bw = torch.exp(prediction[..., 2:3])
    bh = torch.exp(prediction[..., 3:4])
    det_confs = torch.sigmoid(prediction[..., 4:5])
    cls_confs = torch.sigmoid(prediction[..., 5:])

    # grid.shape-> [1, 1, 52, 52, 1]
    # 預測出來的(x, y)是相對於每個cell左上角的點，因此這邊需要由左上角往右下角配合grid_size加上對應的offset，畫出的圖才會在正確的位置上
    grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size, 1]).type(FloatTensor)
    grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size, 1]).type(
        torch.FloatTensor)

    # anchor.shape-> [1, 3, 1, 1, 1]
    # 這邊我認為是要從anchors的大小來還原實際上的寬和高
    anchor_w = FloatTensor([anchors[i * 2] for i in range(num_anchors)])
    anchor_h = FloatTensor([anchors[i * 2 + 1] for i in range(num_anchors)])
    anchor_w = anchor_w.view([1, num_anchors, 1, 1, 1])
    anchor_h = anchor_h.view([1, num_anchors, 1, 1, 1])

    # add offset and then normalize to 0~1
    bx = (bx.data + grid_x) / grid_size
    by = (by.data + grid_y) / grid_size
    bw = (bw * anchor_w) / grid_size
    bh = (bh * anchor_h) / grid_size

    bx1 = (bx - bw * 0.5)
    by1 = (by - bh * 0.5)
    bx2 = (bx + bw * 0.5)
    by2 = (by + bh * 0.5)

    # boxes: [batch, num_anchors, grid_size, grid_size, 4] -> [batch, num_anchors * grid_size * grid_size, 1, 4]
    boxes = torch.cat((bx1, by1, bx2, by2), dim=4).view(num_samples, num_anchors * grid_size * grid_size, 1, 4)

    # confs: [batch, num_anchors * grid_size * grid_size, num_classes]
    confs = (cls_confs * det_confs).view(num_samples, num_anchors * grid_size * grid_size, num_classes)

    return boxes, confs


class YoloLayer(nn.Module):
    def __init__(self, anchor_mask, num_classes, anchors, num_anchors, stride):
        super(YoloLayer, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.stride = stride
        self.scale_x_y = 1
        self.thresh = 0.6

    def forward(self, output):
        anchor_step = len(self.anchors) // self.num_anchors

        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * anchor_step: (m + 1) * anchor_step]

        masked_anchors = [anchor / self.stride for anchor in masked_anchors]
        boxes, confs = yolo_forward(output, self.num_classes, masked_anchors, len(self.anchor_mask),
                                    scale_x_y=self.scale_x_y)
        return boxes, confs
