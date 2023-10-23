from ultralytics import YOLO
import torch
from ultralytics.yolo.v8.segment.predict import SegmentationPredictor
import cv2

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("using GPU")
else:
    device = torch.device("cpu")
    print("using CPU")

model = YOLO("/home/jean/MEGA/YACHAY/10mo SEMESTRE/TESIS/Database & Code/500epochs_tomato_tuned_model.pt")
results = model.predict(source='0', show=True, conf=0.6, stream=True)
print(results)
