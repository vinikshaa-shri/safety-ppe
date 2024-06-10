import ultralytics
from ultralytics import YOLO
infer = YOLO("/home/kritilabs/Desktop/safetyppe/runs/detect/train10/weights/best.pt")
infer.predict("/home/kritilabs/Desktop/safetyppe/PPE_detection-1/valid/images/00060_jpg.rf.5b2b21098580d3b68867bd43d582bde2.jpg", save=True)
print("Precision:", results['precision'])
print("Recall:", results['recall'])
print("mAP@0.5:", results['map50'])   # mAP at IoU=0.5
print("mAP@0.5:0.95:", results['map']) # mAP averaged over IoU thresholds from 0.5 to 0.95
print("F1 Score:", results['f1'])