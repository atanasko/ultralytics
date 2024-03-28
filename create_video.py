import numpy
from PIL import Image
from ultralytics import YOLO
import cv2

# Load a trained YOLOv8 model
model = YOLO('/data/MS/NMU/code/ultralytics/runs/obb/train4/weights/best.pt')

video_filename = 'results/result.mp4'
codec = cv2.VideoWriter_fourcc(*'mp4v')
vid_writer = cv2.VideoWriter(video_filename, codec, 30, (640, 640))

# Run inference
for i in range(198):
    results = model(
        '/data/DEVELOPMENT/DATA/pc_obb_dataset/images/testing/2601205676330128831_4880_000_4900_000_' + str(i) + '.png',
        show_labels=False)  # results list

    # Show the results
    for r in results:
        im_array = r.plot(labels=False)  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        vid_writer.write(numpy.array(im))
        # im.show()  # show image
vid_writer.release()
