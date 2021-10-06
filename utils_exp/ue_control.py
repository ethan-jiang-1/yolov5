s_control_names = {
    "FLAG_ENABLE_CLASSIFER":  False,
    "FLAG_ENABLE_DUMP_CROP_IMGS": False,

    "FLAG_ENABLE_DUMP_CROP_LOG": False,
    "FLAG_HOOK_REAL_2ND_CLASSIFIER": False,
    "FLAG_ADD_PADDING_TO_2ND_CLASSIFIER": False,

    "FLAG_SHORTEN_LABEL": True,
    "FLAG_LABEL_BK": True,
    "FLAG_LABEL_HAND": True,
    "FLAG_LABEL_SMALL": True}


def set_control_flag(control_name, enable_disable):
    global s_control_names
    if control_name in s_control_names:
        print("Change {} from {} to {}".format(control_name, s_control_names[control_name], enable_disable))
        s_control_names[control_name] = enable_disable
        return s_control_names[control_name]
    return None

def get_control_flag(control_name):
    if control_name in s_control_names:
        return s_control_names[control_name]
    return None
