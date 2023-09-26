from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('/data/DEVELOPMENT/AUTONOMOUS/project/ultralytics/runs/detect/train/weights/best.pt')
# model = YOLO('/data/DEVELOPMENT/AUTONOMOUS/project/ultralytics/runs/detect/train/weights/last.pt')

# Run inference on 'bus.jpg'
results = model('/data/DEVELOPMENT/AUTONOMOUS/project/ultralytics/img.png')  # results list

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image