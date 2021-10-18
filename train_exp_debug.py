import sys
import os
# import shutil


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

CONTENT_DATA_ALL_YAML = """
train: ../ds_merge_all/train/images
val: ../ds_merge_all/val/images
test: ../ds_merge_all/test/images

nc: 60
names: ['HBU02-A34', 'HBU02-A37', 'HBU02-A42', 'HBU02-A27', 'HBU02-A28', 'HBU02-A46', 'HBU02-A25', 'HBU02-A84', 'HBU02-A29', 'HBU02-A31', 'HBU02-A33', 'HBU02-A53', 'HBU02-A26S', 'HBU02-A26', 'HBU02-A06', 'HBU02-A20', 'HBU02-A17', 'HBU02-A10', 'HBU02-A32S', 'HBU02-A32', 'HBU03-A84', 'HBU02-A22', 'HBU02-BK0', 'HBU02-BK1', 'HBU02-A30', 'HBU02-A24', 'HBU02-A23', 'HBU02-A47', 'HBU02-A48', 'HBU02-A23S', 'HBU02-A25S', 'HBU02-A19', 'HBUJC0-03', 'HBU02-A04', 'HBU02-A07', 'HBU02-A29S', 'HBU02-A33S', 'HBU02-A58', 'HBU-LR16', 'HBUJC0-02', 'HBU02-A16', 'HBUJC0-01', 'HBU02-A13', 'HBU02-A31S', 'HBU02-A59', 'HBU02-A64', 'HBU02-A66', 'HBU02-A69', 'HBU02-A71', 'hand', 'HBU02-A22S', 'HBU02-A28S', 'HBU02-A27S', 'HBU02-A24S', 'HBU02-A30S', 'HBU02-A21', 'HBU02-A03S', 'HBUJCA-01', 'HBUJCB-01', 'HBUJCC-01']
"""


# from utils.loggers import LOGGERS
from train_exp import parse_opt, main, ROOT 


def _prepare_env():
    print()
    print("ROOT", ROOT)
    dir_yolov5 = os.path.dirname(__file__)
    os.chdir(dir_yolov5)

    # global LOGGERS
    # LOGGERS = ('csv')

def _update_data_all_yaml():
    data_all_yaml_path = "../data_all.yaml"
    with open(data_all_yaml_path, "wt+") as f:
        f.writelines(CONTENT_DATA_ALL_YAML)
    return data_all_yaml_path

def _make_cached_type():
    os.environ["MIXED_COMPRESS_PARAMS"] = ".jpg:90"
    #os.environ["MIXED_COMPRESS_PARAMS"] = ".png:3"
    return "mixed" # "ram"

def _makeup_argv():
    epochs = 10
    img_size = 640
    cache_type = _make_cached_type()

    batch_size = 24

    data_yaml = _update_data_all_yaml()
    hyp_yaml = "data_exp/hyps/hyp.finetune.yaml"
    cfg_yaml = "models/yolov5s.yaml"

    resume_weight_pt = "weights/yolov5s/run_sac60_r2_e830_model-best.pt"
    pm_save_peroid = 10
    pm_stop_patience = 10

    train_cmd = "train_exp.py "
    train_cmd += "--img {} ".format(img_size) 
    train_cmd += "--batch {} ".format(batch_size) 
    train_cmd += "--epochs {} ".format(epochs)
    train_cmd += "--data {} ".format(data_yaml)
    train_cmd += "--cfg {} ".format(cfg_yaml)
    train_cmd += "--project runs/train "
    train_cmd += "--name yolov5s_debug "
    train_cmd += "--cache {} ".format(cache_type)
    train_cmd += "--hyp {} ".format(hyp_yaml)
    train_cmd += "--exist-ok "
 
    if resume_weight_pt is not None:
        train_cmd += "--weights {} ".format(resume_weight_pt)
    if pm_save_peroid >= 10:
        train_cmd += "--save_period {} ".format(pm_save_peroid)
    if pm_stop_patience >= 20:
        train_cmd += "--patience {} ".format(pm_stop_patience)

    cmd = train_cmd.strip()
    sys.argv = cmd.split(" ")
    print(sys.argv)

def do_train_exp():

    _prepare_env()
    _makeup_argv()

    opt = parse_opt()
    main(opt)



if __name__ == "__main__":
    do_train_exp()
