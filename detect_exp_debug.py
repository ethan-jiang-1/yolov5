import sys
import os
import shutil

def _add_sys_path(new_sys_path):
    if new_sys_path not in sys.path:
        if os.path.isdir(new_sys_path):
            print("add sys.path", new_sys_path)
            sys.path.append(new_sys_path)

def extend_sys_paths():
    this_folder = os.path.dirname(__file__)
    parent_folder = os.path.dirname(this_folder)

    new_sys_path = parent_folder + "/yolov5"
    _add_sys_path(new_sys_path)

    new_sys_path = parent_folder 
    _add_sys_path(new_sys_path)


extend_sys_paths()

from detect_exp import parse_opt, main, ROOT 
from utils_exp.ue_apply_classifer import enable_classifier, enable_dump_corp_imgs
from utils_exp.ue_annotator_box_label import enable_object_trackig  # , enable_dump_track_imgs

OD_2ND_CLASSIFIER = False
OD_MODEL_TYPE = "m"
OD_TRACKING_SAMPLE = True

OD_SOURCE_TYPE = "mp4"  # "webcam", "image", "mp4", "mp4jpg"


def _get_label_params():
    cmd = ""
    if OD_TRACKING_SAMPLE:
        cmd += "--hide-conf "
        cmd += "--hide-labels "
        cmd += "--line-thickness 1 "
    else:
        cmd += "--line-thickness 2 "
    return cmd

def _get_weight_pt():
    if OD_MODEL_TYPE == "s":
        #cmd = "--weights weights/yolov5s/run_sac60_r2_e650_model-best.pt "
        cmd = "--weights weights/yolov5s/run_sac60_r2_e830_model-best.pt "
    elif OD_MODEL_TYPE == "m":
        #cmd = "--weights weights/yolov5m/run_mac60_r1_e360_model-last.pt "
        cmd = "--weights weights/yolov5m/run_mac60_r2_e460_model-last.pt " 
    elif OD_MODEL_TYPE == "l":
        cmd = "--weights weights/yolov5l/run_lac60_r1_e310_model-best.pt "
    else:
        raise ValueError("not support")
    return cmd

def _get_source():
    if OD_SOURCE_TYPE == "webcam":
        source = "0"  # webcam usb
        #source = "1"  # webcam screen
    elif OD_SOURCE_TYPE == "mp4":
        #source = "../ds_yolov5_exam/exam_tracking/video_0/track_0.mp4"
        source = "../ds_yolov5_exam/exam_tracking/video_a/track_a.mp4"
    elif OD_SOURCE_TYPE == "mp4jpg":
        source = "../dx_mp4_jpg"
    elif OD_SOURCE_TYPE == "image":
        source = "/ds_yolov5_exam/exam_tracking/tracking_0"
    else:
        raise ValueError("not-support")

    print()
    print("OD_SOURCE_TYPE:", OD_SOURCE_TYPE, "OD_MODEL_TYPE", OD_MODEL_TYPE, "source", source)
    print()
    cmd = "--source {}".format(source)
    return cmd + " "

def _makeup_argv():
    cmd = "detect.py "
   
    cmd += "--imgsz 640 "
    cmd += "--save-txt "

    cmd += "--conf-thres 0.40 "

    cmd += "--project runs/detect "
    cmd += "--name debug_{}_{} ".format(OD_SOURCE_TYPE, OD_MODEL_TYPE)

    cmd += _get_label_params()
    cmd += _get_weight_pt()
    cmd += _get_source()

    cmd = cmd.strip()
    sys.argv = cmd.split(" ")
    print(sys.argv)

def _cleanup_output():
    dir_runs_deubg = "runs/detect/debug"
    if os.path.isdir(dir_runs_deubg):
        shutil.rmtree(dir_runs_deubg)
    dir_runs_crop = "runs/detect_crops"
    if os.path.isdir(dir_runs_crop):
        shutil.rmtree(dir_runs_crop)        

def _prepare_env():
    print()
    print("ROOT", ROOT)
    dir_yolov5 = os.path.dirname(__file__)
    os.chdir(dir_yolov5)

    _cleanup_output()

    if OD_2ND_CLASSIFIER:
        enable_classifier(True)
        enable_dump_corp_imgs(True)

    if OD_SOURCE_TYPE in ["mp4", "mp4jpg"]:
        enable_object_trackig(True)
        #enable_dump_track_imgs(True)

def do_detect_exp():

    _prepare_env()
    _makeup_argv()

    opt = parse_opt()
    main(opt)


if __name__ == "__main__":
    do_detect_exp()

