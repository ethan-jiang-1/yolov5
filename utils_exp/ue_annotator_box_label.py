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


def has_object_tracking():
    return get_control_flag("FLAG_OBJECT_TRACKING")

def enable_object_trackig(enable_disable):
    return set_control_flag("FLAG_OBJECT_TRACKING", enable_disable)

def need_dump_tracking_imgs():
    return get_control_flag("FLAG_ENABLE_DUMP_TRACK_IMGS")

def enable_dump_track_imgs(enable_disable):
    return set_control_flag("FLAG_ENABLE_DUMP_TRACK_IMGS", enable_disable)

def annotator_box_label_exp(annotator, xyxy, label, color=None):
    dx = abs(xyxy[0] - xyxy[2])
    dy = abs(xyxy[1] - xyxy[3])

    if (dx < 48 or dy < 48) and not get_control_flag("FLAG_LABEL_SMALL"):
        label = None

    if label is not None:
        if label.find("BK") != -1 and not get_control_flag("FLAG_LABEL_BK"):
            label = None
        elif label.find("hand") != -1 and not get_control_flag("FLAG_LABEL_HAND"):
            label = None
        elif label.find("HBU") != -1 and get_control_flag("FLAG_SHORTEN_LABEL"):
            label = label.replace("HBU", "")

    annotator.box_label(xyxy, label, color)

def track_box_label_exp(annotator, xyxy, label, color, conf, cls, i, txt_path_stem):
    annotator_box_label_exp(annotator, xyxy, label, color=color)
    jpg_path = txt_path_stem.replace("/labels/", "/images/") + ".jpg"
    _track_annotated_img_exp(jpg_path, annotator.im)
    

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
