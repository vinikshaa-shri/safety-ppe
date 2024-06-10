from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from YAML

# Train the model
results = model.train(data="/home/kritilabs/Desktop/safetyppe/PPE_detection-1/data.yaml", epochs=200, imgsz=640)
