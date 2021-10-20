s_control_names = {
    "FLAG_ENABLE_CLASSIFER":  False,
    "FLAG_ENABLE_DUMP_CROP_IMGS": False,

    "FLAG_ENABLE_DUMP_CROP_LOG": False,
    "FLAG_HOOK_REAL_2ND_CLASSIFIER": False,
    "FLAG_ADD_PADDING_TO_2ND_CLASSIFIER": False,

    "FLAG_SHORTEN_LABEL": False,
    "FLAG_LABEL_BK": False,
    "FLAG_LABEL_HAND": False,
    "FLAG_LABEL_IRREGULAR": False,
    
    "FLAG_OBJECT_TRACKING": False,
    "FLAG_ENABLE_DUMP_TRACK_IMGS": False}

s_control_params = {
    "PARAM_BL_IRR_HC": None,
    "PARAM_BL_IRR_LC": None
}

def dump_control_flags():
    print("\nControl Flags")
    for key, val in s_control_names.items():
        print(key, "\t", val)
    print("\nControl Params")
    for key, val in s_control_params.items():
        print(key, "\t", val)
    print()

def set_control_flag(control_name, enable_disable):
    global s_control_names
    if control_name in s_control_names:
        print("Change {} from {} to {}".format(control_name, s_control_names[control_name], enable_disable))
        s_control_names[control_name] = enable_disable
        return s_control_names[control_name]
    raise ValueError("uknown Flag: {}".format(control_name))

def get_control_flag(control_name):
    if control_name in s_control_names:
        return s_control_names[control_name]
    raise ValueError("uknown Flag: {}".format(control_name))

def set_control_param(control_name, param):
    global s_control_params
    if control_name in s_control_params:
        print("Change {} from {} to {}".format(control_name, s_control_params[control_name], param))
        s_control_params[control_name] = param
        return s_control_params[control_name]
    raise ValueError("uknown Param: {}".format(control_name))

def get_control_param(control_name):
    if control_name in s_control_params:
        return s_control_params[control_name]
    raise ValueError("uknown Param: {}".format(control_name))
