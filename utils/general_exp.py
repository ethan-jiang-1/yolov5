import numpy as np
import cv2
import os
import torch
import torchvision
import torch.nn as nn
from utils.general import xyxy2xywh, xywh2xyxy, scale_coords


ENABLE_CLASSIFER = False
ENABLE_DUMP_CROP_IMGS = False
ENABLE_DUMP_CROP_LOG = False
s_classifiy_cnt = -1


def enable_classifier(enable_disable):
    global ENABLE_CLASSIFER
    ENABLE_CLASSIFER = enable_disable
    print("ENABLE_CLASSIFER", ENABLE_CLASSIFER)
    return ENABLE_CLASSIFER

def enable_dump_corp_imgs(enable_disable):
    global ENABLE_DUMP_CROP_IMGS
    ENABLE_DUMP_CROP_IMGS = enable_disable
    print("ENABLE_DUMP_CROP_IMGS", ENABLE_DUMP_CROP_IMGS)
    return ENABLE_DUMP_CROP_IMGS

def has_classifier_enabled():
    return ENABLE_CLASSIFER


def _load_classifier(name='resnet50', n=2):
    if not ENABLE_CLASSIFER:
        return None

    # Loads a pretrained model reshaped to n-class output
    model = torchvision.models.__dict__[name](pretrained=True)

    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def load_classifier_exp(device):
    if not ENABLE_CLASSIFER:
        return None

    try:
        modelc = _load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
        return modelc
    except Exception as ex:
        print("Exception occured", ex)
        return None


def apply_classifier_exp(x, model, img, im0):
    global s_classifiy_cnt
    s_classifiy_cnt += 1

    # Apply a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR

                if ENABLE_DUMP_CROP_IMGS:
                    os.makedirs("run_crops", exist_ok=True)
                    cv2.imwrite('run_crops/cls_crop_{:03}_{:02}_{:02}.jpg'.format(s_classifiy_cnt, i, j), cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            if model is not None:
                pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
                x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def annotator_box_label_exp(annotator, xyxy, label, color=None):
    dx = abs(xyxy[0] - xyxy[2])
    dy = abs(xyxy[1] - xyxy[3])
    if dx < 48 or dy < 48:
        label = None

    if label is not None:
        if label.find("BK0") != -1:
            label = None
        elif label.find("hand") != -1:
            label = None
        elif label.find("HBU") != -1:
            label = label.replace("HBU", "")

    if label is not None:
        annotator.box_label(xyxy, label, color)
    else:
        print("ignored by annotator_box_label_exp", label, xyxy)
