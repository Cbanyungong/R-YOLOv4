import torch
import time
import math
import random
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from tools.utils import load_class_names
from tools.load import ListDataset
from model.model import Yolo


def create_lr_scheduler(optimizer, epochs):
    """Create learning rate scheduler for training process"""
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)

    return lr_scheduler


def weights_init_normal(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def init():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", type=str, default="data/train", help="path to dataset")
    parser.add_argument("--test_folder", type=str, default="data/test", help="path to dataset")
    parser.add_argument("--weights_path", type=str, default="weights/myyolo2.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    args = parser.parse_args()

    init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_names = load_class_names(args.class_path)
    pretrained_dict = torch.load(args.weights_path)
    model = Yolo(n_classes=1)
    model = model.to(device)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    # 第552項開始為yololayer，訓練時不需要用到
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    # pretrained_dict = {k: v for i, (k, v) in enumerate(pretrained_dict.items()) if i < 552}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    # model.apply(weights_init_normal)  # 權重初始化？
    model.load_state_dict(model_dict)

    train_dataset = ListDataset(args.train_folder)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   pin_memory=True, collate_fn=train_dataset.collate_fn)
    num_iters_per_epoch = len(train_dataloader)

    # optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = create_lr_scheduler(optimizer, args.epochs)

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        for batch, (_, imgs, targets) in enumerate(train_dataloader):
            global_step = num_iters_per_epoch * (epoch - 1) + batch + 1
            imgs = Variable(imgs.to(device), requires_grad=True)
            targets = Variable(targets.to(device), requires_grad=False)

            outputs, loss = model(imgs, targets)

            loss.backward()

            if global_step % 5 == 0:
                print("update")
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, args.epochs, batch, len(train_dataloader))
            print(loss)
            print(log_str)

    torch.save(model.state_dict(), "weights/yolov4_train.pth")
