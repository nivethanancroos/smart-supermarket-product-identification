"""Shared paths and constants for the product identification pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "archive(2)" / "images"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH = MODEL_DIR / "best_model.pth"
CLASSES_PATH = MODEL_DIR / "classes.json"

# YOLO detector (alternative to the classical detection.py + classification.py
# two-stage pipeline): trained on a synthetic labeled basket dataset.
YOLO_DATASET_DIR = PROJECT_ROOT / "yolo_dataset"
YOLO_MODEL_PATH = MODEL_DIR / "yolo_best.pt"

# D2S (Densely Segmented Supermarket Dataset): real multi-item shelf photos
# with COCO-format bbox annotations over 60 branded product categories.
# d2s_to_yolo.py converts these into a YOLO-format dataset (images
# symlinked, not copied, to keep disk usage low) for yolo_train.py.
D2S_RAW_DIR = PROJECT_ROOT / "dataset"
D2S_YOLO_DATASET_DIR = PROJECT_ROOT / "d2s_yolo"
D2S_YOLO_MODEL_PATH = MODEL_DIR / "d2s_yolo_best.pt"

IMG_SIZE = 224

# The classifier only knows the 25 trained categories, so anything else
# (e.g. a product not in the dataset) still gets forced into one of them
# by softmax. Below this confidence, we report "UNKNOWN" instead of a
# possibly-wrong label.
UNKNOWN_CONFIDENCE_THRESHOLD = 0.45
UNKNOWN_LABEL = "UNKNOWN"

# Background color used for synthetic basket canvases (light gray "table").
# Kept distinct from typical product colors so contour-based detection can
# separate objects from the background reliably.
BASKET_BG_COLOR = (235, 235, 230)  # BGR

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
