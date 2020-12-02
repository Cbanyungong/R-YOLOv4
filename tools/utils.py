import math
import numpy as np


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
