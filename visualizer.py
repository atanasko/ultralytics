import io
import numpy as np
from PIL import Image
import cv2


def vis_detection_img(image, od_results):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Create a copy of the original image
    image_with_box = opencv_image.copy()

    for result in od_results:
        boxes = result.boxes
        for box in boxes:
            bbox = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            if int(box.cls) == 0:
                x1, y1, x2, y2 = bbox
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.rectangle(image_with_box, pt1, pt2, (0, 255, 0), thickness=2)

    return image_with_box


def vis_detection_img_1(image, od_results):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Create a copy of the original image
    image_with_box = opencv_image.copy()

    for result in od_results:
        obbs = result.obb
        for obb in obbs:
            bbox = obb.xywhr.data.cpu().numpy()[0]
            cls = obb.cls.data.cpu().numpy()[0]

            rect = ((bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox[4])
            box = cv2.boxPoints(rect).astype(np.int0)
            cv2.drawContours(image_with_box, [box], 0, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Rotated Bounding Box', image_with_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return image_with_box
