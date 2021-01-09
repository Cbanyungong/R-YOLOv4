import math
import numpy as np


def R(theta):
    """
    Args:
        theta: must be radian
    Returns: rotation matrix
    """
    r = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return r


def T(x, y):
    """
    Args:
        x, y: values to translate
    Returns: translation matrix
    """
    t = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])
    return t


def rotate(center_x, center_y, a, p):
    P = np.dot(T(center_x, center_y), np.dot(R(a), np.dot(T(-center_x, -center_y), p)))
    return P[:2]


def xywha2xyxyxyxy(p):
    """
    Args:
        p: 1-d tensor which contains (x, y, w, h, a)
    Returns: bbox coordinates (x1, y1, x2, y2, x3, y3, x4, y4) which is transferred from (x, y, w, h, a)
    """
    x, y, w, h, a = p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4]

    x1, y1, x2, y2 = x + w / 2, y - h / 2, x + w / 2, y + h / 2
    x3, y3, x4, y4 = x - w / 2, y + h / 2, x - w / 2, y - h / 2

    P1 = np.array((x1, y1, 1)).reshape(3, -1)
    P2 = np.array((x2, y2, 1)).reshape(3, -1)
    P3 = np.array((x3, y3, 1)).reshape(3, -1)
    P4 = np.array((x4, y4, 1)).reshape(3, -1)
    P = np.stack((P1, P2, P3, P4)).squeeze(2).T
    P = rotate(x, y, a, P)
    X1, X2, X3, X4 = P[0]
    Y1, Y2, Y3, Y4 = P[1]

    return X1, Y1, X2, Y2, X3, Y3, X4, Y4


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, :4] = xywh2xyxy(boxes[:, :4])
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = ((x1 - pad_x // 2) / unpad_w) * orig_w
    y1 = ((y1 - pad_y // 2) / unpad_h) * orig_h
    x2 = ((x2 - pad_x // 2) / unpad_w) * orig_w
    y2 = ((y2 - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 0] = (x1 + x2) / 2
    boxes[:, 1] = (y1 + y2) / 2
    boxes[:, 2] = (x2 - x1)
    boxes[:, 3] = (y2 - y1)
    return boxes


def get_color(c, x, max_val):
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    ratio = float(x) / max_val * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio = ratio - i
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
    return int(r * 255)


def plot_boxes(img_path, boxes, class_names, img_size, color=None):
    import cv2 as cv
    img = np.array(cv.imread(img_path))

    boxes = rescale_boxes(boxes, img_size, img.shape[:2])
    boxes = np.array(boxes)

    for i in range(len(boxes)):
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        theta = box[4]

        X1, Y1, X2, Y2, X3, Y3, X4, Y4 = xywha2xyxyxyxy(np.array([x, y, w, h, theta]))
        X1, Y1, X2, Y2, X3, Y3, X4, Y4 = int(X1), int(Y1), int(X2), int(Y2), int(X3), int(Y3), int(X4), int(Y4)

        bbox = np.int0([(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)])
        cv.drawContours(img, [bbox], 0, (0, 255, 0), 2)

        #textimg = np.zeros(img.shape, dtype=img.dtype)
        #cv.putText(img, str(round(box[4], 2)), (X1, Y1), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 1)
        #M = cv.getRotationMatrix2D((X1, Y1), int(theta * 180 / np.pi), 1)
        #textimg = cv.warpAffine(textimg, M, (img.shape[1], img.shape[0]), flags=cv.INTER_LINEAR)
        #img = cv.add(img, textimg)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)

        cls_id = np.squeeze(int(box[7]))
        classes = len(class_names)
        offset = cls_id * 123457 % classes
        red = get_color(2, offset, classes)
        green = get_color(1, offset, classes)
        blue = get_color(0, offset, classes)
        if color is None:
            rgb = (red, green, blue)

        #rect = cv.minAreaRect(box[:5])
        #box = cv.cv.BoxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
        #box = np.int0(box)

        img = cv.putText(img, class_names[cls_id] + ":" + str(round(box[5] * box[6], 2)),
                         (X1, Y1), cv.FONT_HERSHEY_SIMPLEX, 0.6, rgb, 1)

    output_path = "outputs/" + img_path.split('/')[-1]
    cv.imwrite(output_path, img)


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names
