"""Shared rendering for detection boxes/labels, used by both the CLI
pipeline (main.py) and the web API (api/app.py).

Box thickness and text size scale with the source image's resolution —
a fixed thickness/font size looks fine on a small demo image but becomes
illegibly thin on D2S's full-resolution (~1920x1440) photos.
"""

import cv2
import numpy as np

from src import config


def format_label(label: str) -> str:
    """D2S class names are snake_case product codes (e.g.
    'gepa_bio_und_fair_kraeuterteemischung') — turn them into something a
    person can actually read at a glance."""
    if label == config.UNKNOWN_LABEL:
        return label
    return label.replace("_", " ").title()


def draw_annotations(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    annotated = image.copy()
    scale = max(annotated.shape[:2]) / 1000
    box_thickness = max(2, round(2 * scale))
    font_scale = max(0.6, 0.7 * scale)
    text_thickness = max(1, round(2 * scale))

    for det in detections:
        x, y, w, h = det["bbox"]
        label = format_label(det["label"])
        text = f"{label} {det['confidence']:.0%}"
        color = (0, 140, 255) if det["label"] == config.UNKNOWN_LABEL else (0, 200, 0)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, box_thickness)

        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        # Prefer a label above the box; drop it below if there's no room
        # (e.g. the box touches the top edge of the image).
        label_bottom = y - 8 if y - 8 - text_h - baseline > 0 else y + h + text_h + baseline + 8
        cv2.rectangle(
            annotated,
            (x, label_bottom - text_h - baseline - 6),
            (x + text_w + 10, label_bottom + baseline),
            color,
            -1,
        )
        cv2.putText(
            annotated, text, (x + 5, label_bottom - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA,
        )

    return annotated
