from ultralytics import YOLO

# Create a new YOLO model from scratch
# model = YOLO('yolov8.yaml')
model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
# model.train(data='coco128.yaml', epochs=10, imgsz=640, batch=32, device=[0, 1])
model.train(data='wod.yaml', epochs=30, imgsz=640, batch=32)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')
