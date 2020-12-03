from model.loss import *


def anchor_wh_iou(wh1, wh2):
    """
    :param wh1: width and height of ground truth boxes
    :param wh2: width and height of anchor boxes
    :return: iou
    """
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


class YoloLayer(nn.Module):
    def __init__(self, num_classes, anchors, stride, scale_x_y, ignore_thresh):
        super(YoloLayer, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.stride = stride
        self.scale_x_y = scale_x_y
        self.ignore_thresh = ignore_thresh
        # mask_anchors :
        # [1.5, 2.0, 2.375, 4.5, 5.0, 3.5],  as stride = 8
        # [2.25, 4.6875, 4.75, 3.4375, 4.5, 9.125],  as stride = 16
        # [4.4375, 3.4375, 6.0, 7.59375, 14.34375, 12.53125],  as stride = 32
        self.masked_anchors = [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]

        self.reduction = 'mean'
        self.use_ciou_loss = True

        self.obj_scale = 1
        self.noobj_scale = 100
        self.lambda_coord = 5
        self.lambda_ciou_scale = 3.54
        self.lambda_conf_scale = 64.3
        self.lambda_cls_scale = 37.4

    def build_targets(self, pred_boxes, pred_cls, target):

        ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

        nB, nA, nG, _, nC = pred_cls.size()

        # Output tensors
        obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
        noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
        class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
        iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
        tx = FloatTensor(nB, nA, nG, nG).fill_(0)
        ty = FloatTensor(nB, nA, nG, nG).fill_(0)
        tw = FloatTensor(nB, nA, nG, nG).fill_(0)
        th = FloatTensor(nB, nA, nG, nG).fill_(0)
        tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

        # Convert to position relative to box
        pred_boxes = pred_boxes * nG
        target_boxes = target[:, 2:6] * nG
        gxy = target_boxes[:, :2]
        gwh = target_boxes[:, 2:]
        # Get anchors with best iou
        ious = torch.stack([anchor_wh_iou(anchor, gwh) for anchor in self.masked_anchors])
        best_ious, best_n = ious.max(0)
        # Separate target values
        b, target_labels = target[:, :2].long().t()
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi, gj = gxy.long().t()
        # Set masks
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0
        tconf = obj_mask.float()

        # Set noobj mask to zero where iou exceeds ignore threshold
        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[b[i], anchor_ious > self.ignore_thresh, gj[i], gi[i]] = 0
        # Coordinates
        tx[b, best_n, gj, gi] = gx - gx.floor()
        ty[b, best_n, gj, gi] = gy - gy.floor()
        # Width and height
        tw[b, best_n, gj, gi] = torch.log(gw / self.masked_anchors[best_n][:, 0] + 1e-16)
        th[b, best_n, gj, gi] = torch.log(gh / self.masked_anchors[best_n][:, 1] + 1e-16)
        # One-hot encoding of label
        tcls[b, best_n, gj, gi, target_labels] = 1
        # Compute label correctness and iou at best anchor
        class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()

        iou, ciou_loss = bbox_xywh_ciou(pred_boxes[b, best_n, gj, gi], target_boxes)

        iou_scores[b, best_n, gj, gi] = iou
        ciou_loss = (1.0 - ciou_loss)

        if self.reduction == 'mean':
            ciou_loss = ciou_loss.mean()
        else:
            ciou_loss = ciou_loss.sum()

        obj_mask = obj_mask.type(torch.bool)
        noobj_mask = noobj_mask.type(torch.bool)

        return iou_scores, ciou_loss, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

    def forward(self, output, target=None):
        # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
        # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        # strides = [8, 16, 32]
        # anchor_step = len(anchors) // num_anchors

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if output.is_cuda else torch.FloatTensor

        # output.shape-> [batch_size, num_anchors * (num_classes + 5), grid_size, grid_size]
        batch_size, grid_size = output.size(0), output.size(2)

        # prediction.shape-> torch.Size([1, num_anchors, grid_size, grid_size, num_classes + 5])
        prediction = (
            output.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                  .permute(0, 1, 3, 4, 2).contiguous()
        )

        pred_x = torch.sigmoid(prediction[..., 0]) * self.scale_x_y - (self.scale_x_y - 1) / 2
        pred_y = torch.sigmoid(prediction[..., 1]) * self.scale_x_y - (self.scale_x_y - 1) / 2
        pred_w = prediction[..., 2]
        pred_h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # grid.shape-> [1, 1, 52, 52, 1]
        # 預測出來的(pred_x, pred_y)是相對於每個cell左上角的點，因此這邊需要由左上角往右下角配合grid_size加上對應的offset，畫出的圖才會在正確的位置上
        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type(FloatTensor)
        grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(FloatTensor)

        # anchor.shape-> [1, 3, 1, 1, 1]
        # 這邊我認為是要從anchors的大小來還原實際上的寬和高
        self.masked_anchors = FloatTensor(self.masked_anchors)
        anchor_w = self.masked_anchors[:, 0].view([1, self.num_anchors, 1, 1])
        anchor_h = self.masked_anchors[:, 1].view([1, self.num_anchors, 1, 1])

        # add offset and then normalize to 0~1
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = (pred_x + grid_x) / grid_size
        pred_boxes[..., 1] = (pred_y + grid_y) / grid_size
        pred_boxes[..., 2] = (torch.exp(pred_w) * anchor_w) / grid_size
        pred_boxes[..., 3] = (torch.exp(pred_h) * anchor_h) / grid_size

        output = torch.cat(
            (
                pred_boxes.view(batch_size, -1, 4),
                pred_conf.view(batch_size, -1, 1),
                pred_cls.view(batch_size, -1, self.num_classes),
            ),
            -1,
        )

        if target is None:
            return output, 0
        else:
            iou_scores, ciou_loss, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = self.build_targets(
                pred_boxes=pred_boxes, pred_cls=pred_cls, target=target
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = F.mse_loss(pred_x[obj_mask], tx[obj_mask], reduction=self.reduction)
            loss_y = F.mse_loss(pred_y[obj_mask], ty[obj_mask], reduction=self.reduction)
            loss_w = F.mse_loss(pred_w[obj_mask], tw[obj_mask], reduction=self.reduction)
            loss_h = F.mse_loss(pred_h[obj_mask], th[obj_mask], reduction=self.reduction)

            #loss_conf_obj = F.binary_cross_entropy(pred_conf[obj_mask], tconf[obj_mask], reduction=self.reduction)
            #loss_conf_noobj = F.binary_cross_entropy(pred_conf[noobj_mask], tconf[noobj_mask], reduction=self.reduction)
            FOCAL = FocalLoss(FloatTensor)
            loss_conf = (
                    FOCAL(pred_conf[obj_mask], tconf[obj_mask])
                    + FOCAL(pred_conf[noobj_mask], tconf[noobj_mask])
            )
            loss_cls = F.binary_cross_entropy(pred_cls[obj_mask], tcls[obj_mask], reduction=self.reduction)

            if self.use_ciou_loss:
                #loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
                total_loss = (
                    self.lambda_ciou_scale * ciou_loss +
                    loss_conf +
                    self.lambda_cls_scale * loss_cls
                )
                print("{}, {}, {}".format(self.lambda_ciou_scale * ciou_loss,
                                          loss_conf,
                                          self.lambda_cls_scale * loss_cls))
            else:
                loss_box = self.lambda_coord * (loss_x + loss_y + loss_w + loss_h)
                #loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
                total_loss = loss_box + loss_conf + loss_cls
                print("{}, {}, {}".format(loss_box, loss_conf, loss_cls))

            return output, total_loss
