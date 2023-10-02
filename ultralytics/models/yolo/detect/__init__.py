# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor
from .train import DetectionTrainer, WodDetectionTrainer
from .val import DetectionValidator, WodDetectionValidator

__all__ = 'DetectionPredictor', 'DetectionTrainer', 'DetectionValidator', 'WodDetectionTrainer', 'WodDetectionValidator'
