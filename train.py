from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8n-obb.yaml')

# Train the model using the WOD dataset for 30 epochs
model.train(data='wod.yaml', epochs=30, imgsz=640, batch=32, device=[0, 1])

# Evaluate the model's performance on the validation set
results = model.val()
