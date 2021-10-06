from utils_exp.ue_control import get_control_flag


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

    if label is not None:
        annotator.box_label(xyxy, label, color)
    else:
        print("ignored by annotator_box_label_exp", label, xyxy)
