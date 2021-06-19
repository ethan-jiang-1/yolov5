from models.experimental import attempt_load
from utils.torch_utils import select_device, time_synchronized
from utils.general import non_max_suppression, xyxy2xywh
from utils.datasets import LoadImages
from models.yolo import Model
import torch


class InspectModel(object):
    @classmethod
    def norm_img(cls, img):
        device = select_device("cpu")
        img_norm = torch.from_numpy(img).to(device)
        img_norm = img_norm.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        if img_norm.ndimension() == 3:
            img_norm = img_norm.unsqueeze(0)
        return img_norm

    @classmethod
    def inspect_and_eval_model(cls, model, dataset):
        for path, img, im0s, vid_cap in dataset:
            print(path, "padded img.shape:", img.shape, "org_img.shape:", im0s.shape, vid_cap)

            img_norm = cls.norm_img(img)

            t1 = time_synchronized()

            print("*pred_raw input shape", img_norm.shape)
            pred_raw = model(img_norm, augment=False)[0]
            print("*pred_raw output (len/shape):\t", "{}/{}".format(len(pred_raw), [pred_raw[i].shape for i in range(len(pred_raw))]))

            # Apply NMS
            # opt_conf_thres = 0.25
            # opt_iou_thres = 0.45
            # opt_classes = None
            # opt_agnostic_nms = False
            # pred = non_max_suppression(pred_raw, opt_conf_thres, opt_iou_thres, classes=opt_classes, agnostic=opt_agnostic_nms)
            pred = non_max_suppression(pred_raw)
            print("*pred -post NMS (len /shape):\t", "{}/{}".format(len(pred), [pred[i].shape for i in range(len(pred))]))

            t2 = time_synchronized()
            print("time gap for full pred/NMS", t2 - t1, "\n")

            for _, det in enumerate(pred):
                cls.output_detection(det, img, im0s, model)

    @classmethod
    def output_detection(cls, det, img, im0s, model):
        if len(det) == 0:
            return

        im0 = im0s.copy()
        s = "{}x{}:  ".format(img.shape[1], img.shape[2])  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        # Rescale boxes from img_size to im0 size
        #det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        print(s)

        # Write results
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) 
            print(line)

    @classmethod
    def create_dataset(cls, model):
        source = "data/images/"
        imgsz = 640
        stride = int(model.stride.max()) 
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        return dataset


def inspect_empty_model():
    model = Model(cfg="models/yolov5s.yaml")
    dataset = InspectModel.create_dataset(model)
    #print(model)
    InspectModel.inspect_and_eval_model(model, dataset)


def inspect_trained_model():
    #weights = ['yolov5s.pt']  # if not local pt found
    weights = ['saved_models/yolov5s.pt']
    device = select_device("cpu")
    model = attempt_load(weights, map_location=device) 

    dataset = InspectModel.create_dataset(model)
    #print(model)
    InspectModel.inspect_and_eval_model(model, dataset)


if __name__ == "__main__":
    #inspect_empty_model()
    inspect_trained_model()
