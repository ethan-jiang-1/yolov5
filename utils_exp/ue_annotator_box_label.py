import os
import sys
import cv2
import shutil

def _add_sys_path(new_sys_path):
    if new_sys_path not in sys.path:
        if os.path.isdir(new_sys_path):
            print("add sys.path", new_sys_path)
            sys.path.append(new_sys_path)

def extend_sys_paths():
    this_folder = os.path.dirname(__file__)
    parent_folder = os.path.dirname(this_folder)
    parent_folder = os.path.dirname(this_folder)

    new_sys_path = parent_folder + "/yolov5"
    _add_sys_path(new_sys_path)

    new_sys_path = parent_folder 
    _add_sys_path(new_sys_path)


extend_sys_paths()

from utils_exp.ue_control import set_control_flag, get_control_flag
from utils_exp.ue_detection_tracker import DetectionTracker
from utils.plots import colors
from utils.general import colorstr
from utils_exp.ue_detection_tracker import dump_tracking_info


def has_object_tracking():
    return get_control_flag("FLAG_OBJECT_TRACKING")

def enable_object_tracking(enable_disable):
    return set_control_flag("FLAG_OBJECT_TRACKING", enable_disable)

def need_dump_tracking_imgs():
    return get_control_flag("FLAG_ENABLE_DUMP_TRACK_IMGS")

def enable_dump_track_imgs(enable_disable):
    return set_control_flag("FLAG_ENABLE_DUMP_TRACK_IMGS", enable_disable)

def _get_obj_type(label):
    if label is not None:
        if label.find("HBU02-BK") != -1:
            return "OtBack"
        elif label.find("hand") != -1:
            return "OtHand"
        return "OtUnlabel"
    return "OtLabeld"
            

def _get_numpy_xyxy(xyxy):
    np_xyxy = xyxy
    if hasattr(np_xyxy, "detach"):
        np_xyxy = np_xyxy.detach().cpu().numpy()   
    return np_xyxy

def _get_irregular_params():
    conf_thres = 0.8

    lc_min_hw = 48
    lc_min_area = 36 * 36
    lc_max_hw_ratio = 2

    hc_min_hw = 36
    hc_min_area = 24 * 24
    hc_max_hw_ratio = 2

    return conf_thres, lc_min_hw, lc_min_area, lc_max_hw_ratio, hc_min_hw, hc_min_area, hc_max_hw_ratio

def _is_object_irregular(annotator, np_xyxy, ot_type, label, conf):
    dx = abs(np_xyxy[0] - np_xyxy[2])
    dy = abs(np_xyxy[1] - np_xyxy[3])

    conf_thres, lc_min_hw, lc_min_area, lc_max_hw_ratio, hc_min_hw, hc_min_area, hc_max_hw_ratio = _get_irregular_params()
    if conf < conf_thres:
        if (dx < lc_min_hw) or (dy < lc_min_hw):
            if ot_type == "OtBack":
                return "BorderBk({}/{}".format(dx, dy) 
            return "BorderNm({}/{})".format(dx, dy) 

        if (dx * dy < lc_min_area):
            if ot_type == "OtBack":
                return "AreaBk({}/{}/{}".format(dx, dy, dx * dy) 
            return "AreaNm({}/{}/{})".format(dx, dy, dx * dy)        

        hw_ratio = max(dx/dy, dy/dx) 
        if (hw_ratio > lc_max_hw_ratio):
            if ot_type == "OtBack":
                return "HwRatioBk({})".format(hw_ratio)
            return "HwRatioNm({})".format(hw_ratio)
    else:
        if (dx < hc_min_hw) or (dy < hc_min_hw):        
            if ot_type == "OtBack":
                return "BorderHcBk({}/{}".format(dx, dy) 
            return "BorderHcNm({}/{})".format(dx, dy) 

        if (dx * dy < hc_min_area):
            if ot_type == "OtBack":
                return "AreaHcBk({}/{}/{}".format(dx, dy, dx * dy) 
            return "AreaHcNm({}/{}/{})".format(dx, dy, dx * dy)        

        hw_ratio = max(dx/dy, dy/dx) 
        if (hw_ratio > hc_max_hw_ratio):
            if ot_type == "OtBack":
                return "HwRatioHcBk({})".format(hw_ratio)
            return "HwRatioHcNm({})".format(hw_ratio)

    return None

def _log_save_dir(save_dir, msg):
    if save_dir is None:
        return

    txt_path = save_dir / "anno_tracking.txt"
    with open(txt_path, 'a') as f:
        f.write(msg + '\n')    

def annotator_box_label_exp(annotator, xyxy, label, color=None, save_dir=None, conf=0):
    np_xyxy = _get_numpy_xyxy(xyxy)
    ot_type = _get_obj_type(label)

    reason = _is_object_irregular(annotator, np_xyxy, ot_type, label, conf) 
    if reason is not None:
        if not get_control_flag("FLAG_LABEL_IRREGULAR"): 
             _log_save_dir(save_dir, "filter out irregular object {} {} reason: {}".format(label, np_xyxy, reason))
             return
        else:
            if label is not None:
                label = "* " + label
            _log_save_dir(save_dir, "keep irregular object {} {} reason: {}".format(label, np_xyxy, reason))

    reason = None
    if label is not None:
        if ot_type == "OtBack" and not get_control_flag("FLAG_LABEL_BK"):
            label = None
            reason = "FLAG_LABEL_BK:{}".format(get_control_flag("FLAG_LABEL_BK"))
        elif ot_type == "OtHand" and not get_control_flag("FLAG_LABEL_HAND"):
            label = None
            reason = "FLAG_LABEL_HAND:{}".format(get_control_flag("FLAG_LABEL_HAND"))
        elif label.find("HBU") != -1 and get_control_flag("FLAG_SHORTEN_LABEL"):
            label = label.replace("HBU", "")

    if label is not None:
        annotator.box_label(xyxy, label, color)
    else:
        _log_save_dir(save_dir, "filter out normal object {} {} reason: {}".format(label, np_xyxy, reason))


def _fun_draw_tracked_fun(objectID, track_found, centroid, klx, avg_xywh, img=None, extra=None, save_dir=None):
    frame = img
    if frame is None:
        return 

    if track_found is None:
        return 

    annotator = extra
    if annotator is None:
        return 
    
    color_box = colors(int(klx), True)
    # color_txt = (255, 255, 255)
    label = ""
    conf = float(track_found.conf.item())
    if s_names:
        label = "{} {:02}".format(s_names[klx], int(100*conf + 0.5))

    x, y, w, h = avg_xywh
    box = (int(x - 0.5 * w), int(y - 0.5 * h), int(x + 0.5 * w), int(y + 0.5 * h))
    # annotator.box_label(box, label=label, color=color_box, txt_color=color_txt)
    annotator_box_label_exp(annotator, box, label, color=color_box, save_dir=save_dir, conf=conf)


s_dtt = None
s_names = None
def track_detections_exp(annotator, detections, names, save_dir=None):
    global s_dtt
    global s_names
    im = annotator.im
    if s_dtt is None:
        if im is not None:
            ch, cw = im.shape[0], im.shape[1]
            s_dtt = DetectionTracker(ch, cw, names)
            s_names = names
            print(colorstr("blue", "\ntrack_detection_exp kicked...\n"))
            dump_tracking_info()

    if s_dtt is None:
        return
    
    s_dtt.draw_traced_detections(detections, fun_draw_tracked_fun=_fun_draw_tracked_fun, img=im, extra=annotator, save_dir=save_dir)
    

s_track_count = 0
def _track_annotated_img_exp(save_path, im0):
    global s_track_count
    if not has_object_tracking():
        return False
    
    if not need_dump_tracking_imgs():
        return False

    dir_name = os.path.dirname(save_path)
    if s_track_count == 0:
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name, exist_ok=True)

    cv2.imwrite(save_path, im0)
    s_track_count += 1
    return True


if __name__ == "__main__":
    print("hi")
