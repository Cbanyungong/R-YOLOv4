import torch
import time
import numpy as np
from tool.utils import load_class_names, plot_boxes_cv2
from tool.yolo import nms
from model import Yolo


def post_processing(conf_thresh, nms_thresh, output):
    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # 416 / 8, 16, 32 -> 52, 26, 13
    # [batch,  ((52 x 52) + (26 x 26) + 13 x 13)) x 3, 1, 4]
    box_array = output[0]

    # [batch,  ((52 x 52) + (26 x 26) + 13 x 13)) x 3, num_classes]
    confs = output[1]

    t1 = time.time()

    if type(box_array).__name__ != 'ndarray':  # torch.Tensor
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    # [batch,  ((52 x 52) + (26 x 26) + 13 x 13)) x 3, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        # 1.Thresholding by Object Confidence
        # First, we filter boxes based on their objectness score.
        # Generally, boxes having scores below a threshold are ignored.
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]
        l_max_id_list = list(set(l_max_id))

        print(l_max_id)
        print(l_max_id_list)

        bboxes = []
        # nms for candidate classes
        for j in l_max_id_list:

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms(ll_box_array, ll_max_conf, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append(
                        [ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3],
                         ll_max_conf[k], ll_max_id[k]])

        bboxes_batch.append(bboxes)

    t3 = time.time()

    print('-----------------------------------')
    print('       max and argmax : %f' % (t2 - t1))
    print('                  nms : %f' % (t3 - t2))
    print('Post processing total : %f' % (t3 - t1))
    print('-----------------------------------')

    print(bboxes_batch)
    return bboxes_batch


def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)

    t1 = time.time()

    output = model(img)

    t2 = time.time()

    print('-----------------------------------')
    print('           Preprocess : %f' % (t1 - t0))
    print('      Model Inference : %f' % (t2 - t1))
    print('-----------------------------------')

    return post_processing(conf_thresh, nms_thresh, output)


if __name__ == "__main__":
    import sys
    import cv2

    namesfile = None
    if len(sys.argv) == 6:
        n_classes = int(sys.argv[1])
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        height = int(sys.argv[4])
        width = int(sys.argv[5])
    elif len(sys.argv) == 7:
        n_classes = int(sys.argv[1])
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        height = sys.argv[4]
        width = int(sys.argv[5])
        namesfile = int(sys.argv[6])
    else:
        print('Usage: ')
        print('  python models.py num_classes weightfile imgfile namefile')

    model = Yolo(yolov4conv137weight=None, n_classes=n_classes, inference=True)

    pretrained_dict = torch.load(weightfile)
    model.load_state_dict(pretrained_dict)

    """
    use_cuda = True
    if use_cuda:
        model.cuda()
    """

    img = cv2.imread(imgfile)
    if img is None:
        print("Can't load image")
        exit(1)

    # Inference input size is 416*416 does not mean training size is the same
    # Training size could be 608*608 or even other sizes
    # Optional inference sizes:
    #   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
    #   Width in {320, 416, 512, 608, ... 320 + 96 * m}
    sized = cv2.resize(img, (width, height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    boxes = do_detect(model, sized, 0.4, 0.5, use_cuda=False)

    if namesfile == None:
        if n_classes == 20:
            namesfile = 'data/voc.names'
        elif n_classes == 80:
            namesfile = 'data/coco.names'
        else:
            print("please give namefile")

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes[0], 'predictions.jpg', class_names)
