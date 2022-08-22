import os, cv2, torch, torchvision
from pytorchyolo import detect, models
import torchvision.models as tvmodels
import torchvision.transforms as T
import torch.nn as nn

avgpool = nn.AdaptiveAvgPool2d((1, 1))

# Load the YOLO model
model = models.load_model(
#   "config\\yolov3.cfg", 
#   "weights\\yolov3.weights")
  "config\\yolov3-tiny.cfg", 
  "weights\\yolov3-tiny.weights")

# VGGNet = tvmodels.vgg11(pretrained=True)
VGGNet = tvmodels.vgg16(pretrained=True)
VGGNet.eval().cuda()

trf = T.Compose([
    # T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test1658726098997\\cam1"
# imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test1658725823165\\cam1"
img_files = os.listdir(imgdir)
init = False
i = 0
init_feat = None
while not init and i < len(img_files):
    init_img = cv2.imread(os.path.join(imgdir, img_files[i]))   # h, w, c
    init_img = cv2.cvtColor(init_img, cv2.COLOR_RGB2BGR)
    boxes = detect.detect_image(model, init_img)
    x1, y1, x2, y2, conf, cls = boxes[0]    # max conf while min cls (person)
    i += 1
    if cls == 0:
        # print(x2 - x1, y2 - y1)
        # tmp_img = torch.tensor(init_img[int(y1): int(y2), int(x1): int(x2), :], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        tmp_img = (init_img[int(y1): int(y2), int(x1): int(x2), :])
        tmp_img.swapaxes(0, 2)
        tmp_img.swapaxes(1, 2)
        tmp_img = trf(tmp_img).unsqueeze(0).cuda()
        # print(tmp_img)
        # print(tmp_img.shape)    # torch.Size([1, 3, 270, 112])
        init_feat = avgpool(VGGNet(tmp_img)).squeeze(-1).squeeze(-1)
        # print(init_feat.shape)  # torch.Size([1, 512])
        # quit()
        init = True

assert init

def EuclideanDistances(a,b):
    sq_a = a**2
    # sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sum_sq_a = torch.sum(sq_a,dim=1)
    sq_b = b**2
    # sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    sum_sq_b = torch.sum(sq_b,dim=1)
    bt = b.t()
    return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))
 
 
# a = torch.rand(1,512).cuda()
# print(EuclideanDistances(a, init_feat))
# print(EuclideanDistances(init_feat, a))

while True:
    p_imgs = []
    sc_img = cv2.imread(os.path.join(imgdir, img_files[i]))
    sc_img = cv2.cvtColor(sc_img, cv2.COLOR_RGB2BGR)
    boxes = detect.detect_image(model, sc_img)
    new_boxes = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        if cls == 0:
            # p_img = torch.tensor(sc_img[int(y1): int(y2), int(x1): int(x2), :], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            p_img = (sc_img[int(y1): int(y2), int(x1): int(x2), :])
            p_img.swapaxes(0, 2)
            p_img.swapaxes(1, 2)
            p_img = trf(p_img).unsqueeze(0).cuda()
            p_imgs.append(p_img)
            new_boxes.append(box)
    boxes = new_boxes
    # i += 1
    if len(p_imgs) == 0:
        tracking_box = [0, 0, 0, 0, 0, 0]
    else: 
        tracking_box = boxes[0]
    if len(p_imgs) > 1:
        # p_imgs = torch.cat(p_imgs)  # shape(h, w) need to be aligned
        min_dis = float('inf')
        md_idx = 0
        print('CMP')
        for idx, p_img in enumerate(p_imgs):
            p_feat = avgpool(VGGNet(p_img)).squeeze(-1).squeeze(-1)
            dis = EuclideanDistances(init_feat, p_feat)
            print(dis)
            if dis < min_dis:
                min_dis = dis
                md_idx = idx
                print(idx)
        tracking_box = boxes[idx]
    x1, y1, x2, y2, conf, cls = tracking_box

    sc_img = cv2.cvtColor(sc_img, cv2.COLOR_RGB2BGR)
    cv2.rectangle(sc_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
    cv2.imshow('img', sc_img)
    key = cv2.waitKey()
    if key == ord('q'):
        break
    elif key == ord('w') and i < len(img_files) - 1:
        i += 1
    elif key == ord('s') and i > 0:
        i -= 1
    elif key == ord('f'):
        print(i)

    




# for box in boxes:
#     x1, y1, x2, y2, conf, cls = box
#     if cls == 0 and conf > 0.7:
#         cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
# cv2.imshow('img', img)
# cv2.waitKey()


# print(boxes)
# Output will be a numpy array in the following format:
# [[x1, y1, x2, y2, confidence, class]]