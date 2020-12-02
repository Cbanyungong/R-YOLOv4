import numpy as np
import torch


def xywh2xyxy(x):
    """
    convert center coordinates, width and height to top left points and down right points
    """
    y = torch.empty(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


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


def nms(boxes, confs, nms_thresh=0.5):
    """
    :param boxes: boxes that overlap and have the same label should be eliminated
    :param confs: predicted confidences of those boxes
    :param nms_thresh: nms threshold
    :return:
    """
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
            if diou(box, boxes[b]) < nms_thresh:
                box_order_that_can_be_kept.append(b)

        order = box_order_that_can_be_kept

    return np.array(keep)


def process(prediction, conf_thresh, nms_thresh):
    # 416 / 8, 16, 32 -> 52, 26, 13
    # box_array.shape-> [batch,  ((52 x 52) + (26 x 26) + (13 x 13)) x 3, 4]
    # confs.shape-> [batch,  ((52 x 52) + (26 x 26) + (13 x 13)) x 3, num_classes]
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    box_array = prediction[..., :4].detach().cpu().numpy()
    confs = prediction[..., 4:5].detach() * prediction[..., 5:].detach()
    batch_size = len(prediction)

    _, test_id = torch.max(prediction[..., 4:5].detach(), dim=1, keepdim=True)

    # [batch, num, num_classes] --> [batch, num]
    # torch.max returns a namedtuple (values, indices) where values is the maximum value of each row of the
    # input tensor in the given dimension dim. And indices is the index location of each maximum value found (argmax).
    max_conf, max_id = torch.max(confs, dim=2, keepdim=True)
    max_conf, max_id = max_conf.cpu().numpy(), max_id.cpu().numpy()

    bboxes_batch = []
    for i in range(batch_size):
        # Thresholding by Object Confidence
        # squeeze的原因是要讓shape-> (n, 1) 變成 shape-> (n)
        argwhere = np.squeeze(max_conf[i] > conf_thresh)
        box_candidate = box_array[i, argwhere, :]  # 信心值有大於門檻的bounding box
        confs_candidate = max_conf[i, argwhere]  # 這些bounding box的信心值
        classes_label = max_id[i, argwhere]  # 這些bounding box對應到的label
        classes = np.unique(classes_label)

        bboxes = []
        # nms for candidate classes
        for j in classes:
            cls_argwhere = np.squeeze(classes_label == j, axis=1)  # axis=1 lets [[True]] -> [True] instead of -> True
            bbox = box_candidate[cls_argwhere, :]
            confs = confs_candidate[cls_argwhere]
            label = classes_label[cls_argwhere]

            keep = nms(bbox, confs, nms_thresh)

            if keep.size > 0:
                bbox = bbox[keep, :]
                confs = confs[keep]
                label = label[keep]

                for k in range(bbox.shape[0]):
                    bboxes.append([bbox[k, 0], bbox[k, 1], bbox[k, 2], bbox[k, 3], confs[k], label[k]])

        bboxes_batch.append(bboxes)

    return bboxes_batch