import os
import sys


root_exp = os.path.dirname(__file__)
root_yolov5 = os.path.dirname(root_exp)
root_app = os.path.dirname(root_yolov5)

if root_yolov5 not in sys.path:
    sys.path.append(root_yolov5)

from utils.datasets import dataset_stats
from utils.datasets import create_dataloader

def test_datase_stats():
    path = root_app + "/ds_merge_all/data_all.yaml"
    stats = dataset_stats(path)
    return stats

def test_create_dataloader():
    train_path = root_app + "/ds_merge_all/test"
    imgsz = 640
    batch_size = 16
    stride = 32
    train_loader, dataset = create_dataloader(train_path, 
                                              imgsz,
                                              batch_size,
                                              stride)
    print(train_loader)
    print(dataset)
    return train_loader, dataset



if __name__ == "__main__":
    print("hi")
    #test_datase_stats()
    test_create_dataloader()

