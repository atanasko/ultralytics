from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n-cls.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="wod.yaml", epochs=100, imgsz=640)
