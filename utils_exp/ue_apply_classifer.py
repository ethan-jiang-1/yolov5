import numpy as np
import cv2
import os
import torch
import torchvision
import torch.nn as nn
from utils.general import xyxy2xywh, xywh2xyxy, scale_coords
from utils_exp.ue_control import set_control_flag, get_control_flag


s_classifiy_cnt = -1


def enable_classifier(enable_disable):
    return set_control_flag("FLAG_ENABLE_CLASSIFER", enable_disable)

def enable_dump_corp_imgs(enable_disable):
    return set_control_flag("FLAG_ENABLE_DUMP_CROP_IMGS", enable_disable)

def has_classifier_enabled():
    return get_control_flag("FLAG_ENABLE_CLASSIFER")

def has_real_classifier_hooked():
    return get_control_flag("FLAG_HOOK_REAL_2ND_CLASSIFIER")


def _load_classifier(name='resnet50', n=2):
    if not has_classifier_enabled():
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
    if not has_classifier_enabled():
        return None

    if not has_real_classifier_hooked():
        print("WARNING: 2nd stage classifier not hooked !")
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
            # ethan modify: this add padding
            if get_control_flag("FLAG_ADD_PADDING_TO_2ND_CLASSIFIER"):
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

                if get_control_flag("FLAG_ENABLE_DUMP_CROP_IMGS"):
                    name = "runs/detect_crops/cls_crop_{:03}_{:02}_{:02}".format(s_classifiy_cnt, i, j)
                    os.makedirs("runs/detect_crops/", exist_ok=True)
                    cv2.imwrite(name + ".jpg", cutout)
                    with open(name + ".txt", "wt+") as f:
                        f.write(str(a))

                # Ethan, the following is kind of transform before feed to 2nd-stage classifier
                # should be replaced by timm's transfrom if we use timm
                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            if model is not None:
                pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
                x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x
