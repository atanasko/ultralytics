# Advanced-Machine-Learning

This project use [Waymo Open Dataset](https://waymo.com/open/) (WOD) to train and test YOLOv8 object detection in BEV images
using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
In order to setup the project, install python 3.10. Download the project
from [GitHub repository](https://github.com/atanasko/Ambient-Intelligence).
Install required packages

```
pip install -r requirements.txt
```

In order to grab data from WOD, install [Google cloud CLI](https://cloud.google.com/storage/docs/gsutil_install#deb)
like:

```
# sudo apt-get update

# sudo apt-get install apt-transport-https ca-certificates gnupg curl sudo

# curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

# echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# sudo apt-get update && sudo apt-get install google-cloud-cli
```

```
# gcloud auth login
```

Download WOD training, testing and validation dataset from Waymo Open
Dataset [download](https://waymo.com/open/download/) into ex.

```
/data/wod
```

directory.

In order to train the YOLOv8 to detect objects in BEV images created from Lidar PC, data from WOD need to be converted into YOLOv8 OBB format.
The YOLO OBB format designates bounding boxes by their four corner points with coordinates normalized between 0 and 1. It follows this format:

```
class_index, x1, y1, x2, y2, x3, y3, x4, y4
```
Conversion is performed by using ```wod_converter.py``` script like:

```
# python wod_converter.py -w <wod_dataset_dir> -o <yolov8_dataset_dir>
```

After conversion is finished, OBB model can be trained using the ```train.py``` script.

```
# python train.py
```

Trained model can be used to detect objects in BEV images created from Lidar Point Cloud like ```detect.py``` script:

```
# python detect.py
```

In the following video results from object detection in BEV images from Lidar Point Cloud are displayed

![Watch the video](results/result.gif) 