import numpy as np
import torch
from tools.utils import skewiou


def post_process(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Args:
        prediction: size-> [batch, ((grid x grid) + (grid x grid) + (grid x grid)) x num_anchors, 8]
                    ex: [1, ((52 x 52) + (26 x 26) + (13 x 13)) x 18, 8] in my case
                    last dimension-> [x, y, w, h, a, conf, num_classes]
    Returns:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    output = [None for _ in range(len(prediction))]
    for batch, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 5] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 5] * image_pred[:, 6:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 6:].max(1, keepdim=True)  # class_preds-> index of classes
        detections = torch.cat((image_pred[:, :6], class_confs.float(), class_preds.float()), 1)

        # non-maximum suppression
        keep_boxes = []
        labels = detections[:, -1].unique()
        for label in labels:
            detect = detections[detections[:, -1] == label]
            while len(detect):
                large_overlap = skewiou(detect[0, :5], detect[:, :5]) > nms_thres
                # Indices of boxes with lower confidence scores, large IOUs and matching labels
                weights = detect[large_overlap, 5:6]
                # Merge overlapping bboxes by order of confidence
                detect[0, :4] = (weights * detect[large_overlap, :4]).sum(0) / weights.sum()
                keep_boxes += [detect[0].detach()]
                detect = detect[~large_overlap]
            if keep_boxes:
                output[batch] = torch.stack(keep_boxes)

    return output


def diou(box1, box2):
    """
    calculates iou for two boxes
    """
    # iou
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
    iou = inter_area / union_area

    # diou
    center_x1, center_y1 = (t1_x1 + t1_x2) / 2, (t1_y1 + t1_y2) / 2
    center_x2, center_y2 = (t2_x1 + t2_x2) / 2, (t2_y1 + t2_y2) / 2
    outerX_square = pow(min(t1_x1, t2_x1) - max(t1_x2, t2_x2), 2)
    outerY_square = pow(min(t1_y1, t2_y1) - max(t1_y2, t2_y2), 2)
    c_square = outerX_square + outerY_square
    center_distance = pow(center_x1 - center_x2, 2) + pow(center_y1 - center_y2, 2)
    diou = center_distance / c_square

    return iou - diou
