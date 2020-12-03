import torch
from model.model import Yolo
from collections import OrderedDict


model = Yolo(n_classes=80)

pretrained_dict = torch.load("weights/yolov4.pth")
generator_state_dict = model.state_dict()

new_state_dict = OrderedDict()
for i, (k, v) in enumerate(pretrained_dict.items()):
    new_state_dict[i] = v

final_state_dict = OrderedDict()
for i, (k, v) in enumerate(generator_state_dict.items()):
    final_state_dict[k] = new_state_dict[i]

model.load_state_dict(final_state_dict)
torch.save(model.state_dict(), "weights/myyolo2.pth")
