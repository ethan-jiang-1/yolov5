import sys
import os
import shutil
from detect_exp import parse_opt, main, ROOT 
from utils_exp.ue_apply_classifer import enable_classifier, enable_dump_corp_imgs

def _makeup_argv():
    cmd = "detect.py "
   
    cmd += "--weights weights/yolov5s/run_sac60_r2_e650_model-best.pt "
    #cmd += "--weights weights/yolov5m/run_mac60_r1_e360_model-last.pt "
   
    cmd += "--imgsz 640 "
    cmd += "--line-thickness 1 "
    cmd += "--save-txt "
    cmd += "--name debug "

    cmd += "--project runs/detect "
    
    #cmd += "--source ../ds_yolov5_exam/exam_sac15/images_0"
    #cmd += "--source ../ds_yolov5_exam/exam_sac15/images_b0/84696d2c-000b-4c2d-b88d-be30a2f5ecc3.jpeg"

    cmd += "--source ../ds_yolov5_exam/exam_sac60/images_r0/sac60_test_4024.jpg"

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

    enable_classifier(True)
    enable_dump_corp_imgs(True)

def do_detect_exp():

    _prepare_env()

    _makeup_argv()

    opt = parse_opt()
    main(opt)


if __name__ == "__main__":
    do_detect_exp()

