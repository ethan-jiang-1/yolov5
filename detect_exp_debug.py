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
from utils_exp.ue_annotator_box_label import enable_object_trackig, enable_dump_track_imgs

OD_SOURCE_TYPE = "mp4"  # "webcam"   # "webcam", "image", "mp4"
OD_2ND_CLASSIFIER = False


def _get_weight_pt():
    cmd = "--weights weights/yolov5s/run_sac60_r2_e650_model-best.pt "
    #cmd = "--weights weights/yolov5m/run_mac60_r1_e360_model-last.pt "
    #cmd = "--weights weights/yolov5l/run_lac60_r1_e310_model-best.pt "
    return cmd

def _get_source():
    if OD_SOURCE_TYPE == "webcam":
        cmd = "--source 0"  # webcam usb
        #cmd = "--source 1"  # webcam screen
        #cmd = "--source 2"  # webcam screen
    elif OD_SOURCE_TYPE == "mp4":
        cmd = "--source ../ds_yolov5_exam/exam_tracking/video_0/track.mp4"
    else:
        #cmd = "--source ../ds_yolov5_exam/exam_sac15/images_0"
        #cmd = "--source ../ds_yolov5_exam/exam_sac15/images_b0/84696d2c-000b-4c2d-b88d-be30a2f5ecc3.jpeg"
        #cmd = "--source ../ds_yolov5_exam/exam_sac60/images_r0/sac60_test_4024.jpg"
        cmd = "--source ../ds_yolov5_exam/exam_tracking/tracking_0"

    return cmd + " "

def _makeup_argv():
    cmd = "detect.py "
   
    cmd += "--imgsz 640 "
    #cmd += "--line-thickness 1 "
    cmd += "--line-thickness 2 "
    cmd += "--save-txt "

    cmd += "--conf-thres 0.40 "

    cmd += "--project runs/detect "
    cmd += "--name debug "

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

    if OD_SOURCE_TYPE == "mp4":
        enable_object_trackig(True)
        #enable_dump_track_imgs(True)

def do_detect_exp():

    _prepare_env()
    _makeup_argv()

    opt = parse_opt()
    main(opt)


if __name__ == "__main__":
    do_detect_exp()

