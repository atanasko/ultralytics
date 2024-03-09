from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('runs/obb/train4/weights/best.pt')

results = model('/pc_obb_dataset/images/testing/10980133015080705026_780_000_800_000_0.png')  # results list

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image