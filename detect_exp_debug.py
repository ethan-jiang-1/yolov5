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
from utils_exp.ue_annotator_box_label import enable_object_trackig

USING_WEB_CAM = False
USING_2ND_CLASSIFIER = False
USING_OBJ_TRACKING = True

def _get_weight_pt():
    cmd = "--weights weights/yolov5s/run_sac60_r2_e650_model-best.pt "
    #cmd = "--weights weights/yolov5m/run_mac60_r1_e360_model-last.pt "
    return cmd

def _get_source():
    if not USING_WEB_CAM:
        #cmd = "--source ../ds_yolov5_exam/exam_sac15/images_0"
        #cmd = "--source ../ds_yolov5_exam/exam_sac15/images_b0/84696d2c-000b-4c2d-b88d-be30a2f5ecc3.jpeg"
        #cmd = "--source ../ds_yolov5_exam/exam_sac60/images_r0/sac60_test_4024.jpg"
        cmd = "--source ../ds_yolov5_exam/exam_tracking/tracking_0"
    else:
        cmd = "--source 0"  # webcam usb
        #cmd = "--source 1"  # webcam screen
        #cmd = "--source 2"  # webcam screen
    return cmd + " "

def _makeup_argv():
    cmd = "detect.py "
   
    cmd += "--imgsz 640 "
    #cmd += "--line-thickness 1 "
    cmd += "--line-thickness 3 "

    cmd += "--save-txt "
    cmd += "--name debug "
    cmd += "--conf-thres 0.60 "

    cmd += "--project runs/detect "
    
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

    if USING_2ND_CLASSIFIER:
        enable_classifier(True)
        enable_dump_corp_imgs(True)

    if USING_OBJ_TRACKING:
        enable_object_trackig(True)

def do_detect_exp():

    _prepare_env()

    _makeup_argv()

    opt = parse_opt()
    main(opt)


if __name__ == "__main__":
    do_detect_exp()

