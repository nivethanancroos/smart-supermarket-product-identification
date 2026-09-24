"""Wraps a trained YOLO model for end-to-end product detection.

Unlike the classical pipeline (detection.py finds regions, then
classification.py labels each crop), YOLO performs detection and
classification in a single learned model, trained on the synthetic
labeled basket dataset (see yolo_dataset.py / yolo_train.py).
"""

import numpy as np

from src import config


class YoloProductDetector:
    def __init__(self, weights_path=config.YOLO_MODEL_PATH, conf_threshold: float = 0.35):
        from ultralytics import YOLO  # deferred: heavy import, only needed here

        self.model = YOLO(str(weights_path))
        self.conf_threshold = conf_threshold

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        """Returns a list of {"bbox": [x, y, w, h], "label": str, "confidence": float}."""
        results = self.model.predict(
            image_bgr, conf=self.conf_threshold, agnostic_nms=True, verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            detections.append({
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "label": self.model.names[cls_id],
                "confidence": confidence,
            })

        detections.sort(key=lambda d: (d["bbox"][1] // 100, d["bbox"][0]))
        return detections
