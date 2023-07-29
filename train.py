from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n-cls.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="wod.yaml", epochs=100, imgsz=640, batch=32, cache='ram', augment=False)  # device=[0, 1]
# results = model.train(data="wod.yaml", epochs=100, imgsz=640, cache='disk', augment=False)
