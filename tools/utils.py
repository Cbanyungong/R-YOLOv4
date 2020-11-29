import math
import torch
import numpy as np


def bbox_xywh_ciou(pred_boxes, target_boxes):
    assert pred_boxes.size() == target_boxes.size(), "Unmatch size of pred_boxes and target_boxes"
    device = pred_boxes.device

    # get coordinates
    pred_boxes = xywh2xyxy(pred_boxes)
    target_boxes = xywh2xyxy(target_boxes)
    b1_x1, b1_x2, b1_y1, b1_y2 = pred_boxes[..., 0], pred_boxes[..., 1], pred_boxes[..., 2], pred_boxes[..., 3]
    b2_x1, b2_x2, b2_y1, b2_y2 = target_boxes[..., 0], target_boxes[..., 1], target_boxes[..., 2], target_boxes[..., 3]
    x1, x2 = torch.max(b1_x1, b2_x1), torch.min(b1_x2, b2_x2)
    y1, y2 = torch.max(b1_y1, b2_y1), torch.min(b1_y2, b2_y2)

    # cal ious
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    union_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area
    ious = inter_area / (union_area + 1e-16)

    # cal center distance
    center_dis = torch.pow(pred_boxes[..., 0] - target_boxes[..., 0], 2) + \
                 torch.pow(pred_boxes[..., 1] - target_boxes[..., 1], 2)

    # cal outer boxes
    min_x, max_x = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
    min_y, max_y = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
    outer_diagonal_line = torch.pow(max_x - min_x, 2) + torch.pow(max_y - min_y, 2)

    # cal penalty term
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(pred_boxes[..., 2] / (target_boxes[..., 2] + 1e-16)) -
        torch.atan(pred_boxes[..., 3] / (target_boxes[..., 3] + 1e-16)), 2)
    alpha = v / (1 - ious + v)

    # cal ciou
    cious = 1 - ious + center_dis / outer_diagonal_line + alpha * v

    return torch.tensor(ious, device=device, dtype=torch.float), cious


def anchor_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def xywh2xyxy(x):
    y = torch.empty(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


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
    union_area = (t1_x2 - t1_x1) * (t1_y2 - t1_y1) + (t2_x2 - t2_x1) * (t2_y2 - t2_y1) - inter_area + 1e-16

    return inter_area / union_area


def nms(boxes, confs, nms_thresh=0.5):
    confs = np.squeeze(confs)
    order = np.argsort(confs)

    # nms algorithm
    keep = []
    while len(order) > 0:
        first_box_index, order = order[-1], order[:-1]
        box = boxes[first_box_index]

        # box looks like-> [(0, array([0.18615767, 0.22766608, 0.748049  , 0.7332627 ], dtype=float32)),
        #                   (1, array([0.19176558, 0.23242362, 0.73947287, 0.7266734 ], dtype=float32)), ...
        keep.append(first_box_index)
        box_order_that_can_be_kept = []
        for b in order:
            if iou(box, boxes[b]) < nms_thresh:
                box_order_that_can_be_kept.append(b)

        order = box_order_that_can_be_kept

    return np.array(keep)


def get_color(c, x, max_val):
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    ratio = float(x) / max_val * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio = ratio - i
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
    return int(r * 255)


def plot_boxes(img_path, boxes, class_names, color=None):
    import cv2
    img = np.array(cv2.imread(img_path))

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)

        cls_id = np.squeeze(box[5])
        classes = len(class_names)
        offset = cls_id * 123457 % classes
        red = get_color(2, offset, classes)
        green = get_color(1, offset, classes)
        blue = get_color(0, offset, classes)
        if color is None:
            rgb = (red, green, blue)
        img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)

    output_path = "outputs/" + img_path.split('/')[-1]
    cv2.imwrite(output_path, img)


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names
