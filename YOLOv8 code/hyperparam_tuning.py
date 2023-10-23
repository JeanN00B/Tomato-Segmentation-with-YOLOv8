from ultralytics import YOLO
from ray import tune
import wandb
import ray
from ray.tune import CLIReporter
from ray.tune.search.bayesopt import BayesOptSeach
from ray.tune.schedulers import ASHAScheduler
from IPython import display
display.clear_output()
import ultralytics
ultralytics.checks()
import cv2
import numpy as np


hyperparam_space = {
    "epochs": tune.choice([80]),
    "batch": tune.uniform(8, 20),
    "lr0": tune.uniform(1e-3, 1e-1),
    "lrf": tune.uniform(0.1, 1.0),
    "momentum": tune.uniform(0.6, 0.98),
    "weight_decay": tune.uniform(0.0, 0.001),
    "warmup_epochs": tune.uniform(0.0, 5.0),
    "warmup_momentum": tune.uniform(0.0, 0.95),
    "box": tune.uniform(0.02, 0.2),
    "cls": tune.uniform(0.2, 4.0),
    "hsv_h": tune.uniform(0.0, 0.1),
    "hsv_s": tune.uniform(0.0, 0.1),
    "hsv_v": tune.uniform(0.0, 0.1),
    "degrees": tune.uniform(0.0, 45.0),
    "translate": tune.uniform(0.0, 0.9),
    "scale": tune.uniform(0.0, 0.9),
    "shear": tune.uniform(0.0, 10.0),
    "perspective": tune.uniform(0.0, 0.001),
    "flipud": tune.uniform(0.0, 1.0),
    "fliplr": tune.uniform(0.0, 1.0),
    "mosaic": tune.uniform(0.0, 1.0),
    "mixup": tune.uniform(0.0, 1.0),
    "copy_paste": tune.uniform(0.0, 1.0)
}


model = YOLO("/usr/src/ultralytics/models/100epochs-2k-img-segmentation.pt ")

wandb.init(
    project = "yolov8-tomato-segm-l_model",
    config = hyperparam_space,
    name = "yolov8-tuning-segm"
)

results = model.tune(
    data = "/usr/src/ultralytics/tomato-dataset/tomato-dataset-1.3kimg/data.yaml",
    space = hyperparam_space,
    train_args={
        "workers": 4,
        "epochs" : 80,
        "patience":0,
        "batch": 16,
        "imgsz": 640
    })

print("==+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=++=+=+=+")
print("==+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=++=+=+=+")
print("==+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=++=+=+=+")
print(results.get_best_result(metric = 'metrics/mAP50(M)', mode = 'max'))
print(results.get_best_result(metric = 'metrics/precision(M)', mode = 'max'))
print(results.get_best_result(metric = 'metrics/recall(M)', mode = 'max'))
print("==+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=++=+=+=+")
print("==+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=++=+=+=+")
print("==+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=++=+=+=+")

wandb.stop()
