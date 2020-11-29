import torch
import time
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tools.utils import load_class_names, plot_boxes
from tools.utils import nms, xywh2xyxy
from tools.load import ImageFolder
from model.model import Yolo


def post_processing(prediction, conf_thresh, nms_thresh):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/img", help="path to dataset")
    parser.add_argument("--weights_path", type=str, default="weights/myyolo2.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.4, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    class_names = load_class_names(args.class_path)
    pretrained_dict = torch.load(args.weights_path, map_location=torch.device('cpu'))
    model = Yolo(n_classes=80)
    model = model.to(device)
    model.load_state_dict(pretrained_dict)

    model.eval()

    dataset = ImageFolder(args.image_folder, img_size=args.img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    boxes = []
    imgs = []

    start = time.time()
    for img_path, img in dataloader:
        img = torch.autograd.Variable(img.type(FloatTensor))

        with torch.no_grad():
            temp = time.time()
            output, _ = model(img)  # batch=1 -> [1, n, n], batch=3 -> [3, n, n]
            temp1 = time.time()
            box = post_processing(output, args.conf_thres, args.nms_thres)
            temp2 = time.time()
            boxes.extend(box)
            print('-----------------------------------')
            num = 0
            for b in box:
                num += len(b)
            print("{}-> {} objects found".format(img_path, num))
            print("Inference time : ", round(temp1 - temp, 5))
            print("Post-processing time : ", round(temp2 - temp1, 5))
            print('-----------------------------------')

        imgs.extend(img_path)

    for i, (img_path, box) in enumerate(zip(imgs, boxes)):
        plot_boxes(img_path, box, class_names)

    end = time.time()

    print('-----------------------------------')
    print("Total detecting time : ", round(end - start, 5))
    print('-----------------------------------')
