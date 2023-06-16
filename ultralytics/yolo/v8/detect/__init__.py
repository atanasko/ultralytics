# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor, predict
from .train import DetectionTrainer, WodDetectionTrainer, train
from .val import DetectionValidator, val

__all__ = 'DetectionPredictor', 'predict', 'DetectionTrainer', 'WodDetectionTrainer', 'train', 'DetectionValidator', 'val'
