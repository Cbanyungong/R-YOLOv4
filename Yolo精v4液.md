# Yolo精v4液

Weights我放在這：
[Google Drive](https://drive.google.com/drive/u/1/folders/1KgKY3j66GQnpGVUAbdw7-tNgHxCPDT_F)

我程式碼參考這些：
[yolov3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
[yolov4](https://github.com/Tianxiaomo/pytorch-YOLOv4/tree/4ccef0ec8fe984e059378813e33b3740929e0c19)

那個yolov4其實我覺得寫得還滿完整，可是真的是看懂要花很多時間。

這個示意圖超讚 [from](https://becominghuman.ai/explaining-yolov4-a-one-stage-detector-cdac0826cbd7)
![](https://i.imgur.com/joIhWJ1.jpg)

以下這些網站是關於Yolo概念我覺得還不錯：
[懶丁](https://linnil1.medium.com/yolov4如何變強-相關細節介紹-2cb3c4404849)
[寶妹](https://medium.com/@chingi071/yolo演進-3-yolov4詳細介紹-5ab2490754ef)
[阿盤](https://jonathan-hui.medium.com/yolov4-c9901eaa8e61)
[肚毛](https://becominghuman.ai/explaining-yolov4-a-one-stage-detector-cdac0826cbd7)

### 首先
Github Respository裡面有一個cfg資料夾，裡面的cfg就是作者寫的yolov4架構，關於要怎麼頗析這些文字請參考：
[傑西1](https://medium.com/@chih.sheng.huang821/深度學習-物件偵測yolov1-yolov2和yolov3-cfg-檔解讀-75793cd61a01)
[傑西2](https://medium.com/@chih.sheng.huang821/深度學習-物件偵測yolov1-yolov2和yolov3-cfg-檔解讀-二-f5c2347bea68)

所以model.py裡面，就是照著這個結構用Pytorch做出來的模型而已。
真的是很長，不過裡面的寫法就跟上方的圖一模一樣。
```python
def forward(self, i):
    # Backbone
    i = self.conv0(i)
    d1 = self.down1(i)
    d2 = self.down2(d1)
    d3 = self.down3(d2)
    d4 = self.down4(d3)
    d5 = self.down5(d4)

    x20, x13, x6 = self.neek(d5, d4, d3)

    output = self.head(x20, x13, x6)
    return output
```

### 執行程式碼
如果檔案clone下來原封不動地執行的話，輸入這行指令就可以了
```
python detect.py 80 weights/myyolo.pth data/cars.jpg 608 608
```
其中(608 608)是input圖片的大小，常用的也有(416, 416)，預測會較不準確但是速度會較快

### Detect
這邊我只有研究Yolo output什麼，還有要怎麼處理他的output然後來畫出bounding box而已。

我覺得先看完這兩篇大概就能懂我接下來想表達的是啥。
[懶包1](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)
[懶包2](https://medium.com/analytics-vidhya/yolo-v3-theory-explained-33100f6d193)

```python
# boxes: [batch, num_anchors, grid_size, grid_size, 4] -> [batch, num_anchors * grid_size * grid_size, 1, 4]
boxes = torch.cat((bx1, by1, bx2, by2), dim=4).view(num_samples, num_anchors * grid_size * grid_size, 1, 4)

# confs: [batch, num_anchors * grid_size * grid_size, num_classes]
confs = (cls_confs * det_confs).view(num_samples, num_anchors * grid_size * grid_size, num_classes)
```
這個是Yolov4的output

這邊可以看到最後輸出的boxes的維度就是-> [batch, num_anchors * grid_size * grid_size, 1, 4]
batch是用在同時detect很多圖片的時候用的，我還沒有寫到一次偵測多張圖片所以目前batch都是1。num_anchors是每一個圖片的cell可以偵測到的object數量，yolov4設定是3，然後grid_size就會根據你的輸入的大小來決定，最後的4就是圖片的左上方座標和右下方座標。

再來就要進行nms演算法來篩選這些預測出來的box，下面這篇我覺得講得很好。
[NMS](https://medium.com/@chih.sheng.huang821/機器-深度學習-物件偵測-non-maximum-suppression-nms-aa70c45adffa)

In case 你忘記IOU

![](https://i.imgur.com/XR4JYjH.png)

```python
# [batch, num, num_classes] --> [batch, num]
max_conf = np.max(confs, axis=2)
max_id = np.argmax(confs, axis=2)

for i in range(box_array.shape[0]):
    # 1.Thresholding by Object Confidence
    # First, we filter boxes based on their objectness score.
    # Generally, boxes having scores below a threshold are ignored.
    argwhere = max_conf[i] > conf_thresh
    l_box_array = box_array[i, argwhere, :]
    l_max_conf = max_conf[i, argwhere]
    l_max_id = max_id[i, argwhere]
    l_max_id_list = list(set(l_max_id))
```

這邊一開始在做的事情就是，假設你輸入的input size是416的話，我們就會得到這麼多bounding box-> ((52 x 52) + (26 x 26) + 13 x 13)) x 3。而這些bounding box都有他們對應的信心值（信心值=這個cell有物體的機率 * 每個class的機率）。這個意思就是說，假如這個cell預測出這裡有物體存在的機率有0.9，然後所有class的機率裡面，狗的機率最高有0.8，那麼這個bounding box 的 confidence就是0.72。
因此，再用nms之前，我們可以先把信心值過低的cell先刪掉，就可以省下滿多計算。

```python
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
```

而從這些結果中我們就可以找出這些bounding box預測出的種類，然後從這些種類中的bounding box裡面每一種分開下去找，因為既然有預測出來那就至少要有一個那個種類的bounding box


```python
def nms(boxes, confs, nms_thresh=0.5):
    dict_of_box = []
    order = np.argsort(confs)
    boxes = boxes[order]
    for i in range(len(order)):
        dict_of_box.append(tuple([order[i], boxes[i]]))

    # nms algorithm
    keep = []
    while len(dict_of_box) > 0:
        box = dict_of_box.pop()

        # box looks like-> [(0, array([0.18615767, 0.22766608, 0.748049  , 0.7332627 ], dtype=float32)),
        #                   (1, array([0.19176558, 0.23242362, 0.73947287, 0.7266734 ], dtype=float32)), ...
        keep.append(box[0])

        box_that_can_be_kept = []
        for b in dict_of_box:
            if iou(box[1], b[1]) < nms_thresh:
                box_that_can_be_kept.append(b)

        dict_of_box = box_that_can_be_kept

    return np.array(keep)
```

這個我寫出的NMS其實就跟這個Psuedocode一模一樣

![](https://i.imgur.com/KZ9uZFG.png)


好了，最後找出合格的bounding box後，根據這些點把框框畫出來就行了。
最後就剩下要怎麼train了，這個影片有講到怎麼寫Loss但是他是用Yolo1來講的[this](https://www.youtube.com/watch?v=n9_XyCGr-MI)
有不懂都可以馬上問我。

