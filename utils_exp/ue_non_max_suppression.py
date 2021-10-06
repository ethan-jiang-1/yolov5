from utils.general import non_max_suppression

def non_max_suppression_exp(prediction, 
                            conf_thres=0.25, 
                            iou_thres=0.45, 
                            classes=None, 
                            agnostic=False, 
                            multi_label=False,
                            labels=(), 
                            max_det=300):
                
    return non_max_suppression(prediction, 
                               conf_thres=conf_thres, 
                               iou_thres=iou_thres, 
                               classes=classes, 
                               agnostic=agnostic, 
                               multi_label=multi_label, 
                               labels=labels,
                               max_det=max_det)
