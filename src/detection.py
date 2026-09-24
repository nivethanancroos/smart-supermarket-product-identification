"""Module 2: Object detection and segmentation.

Isolates individual products from a basket image using classical image
processing (background-color distance / Otsu thresholding + morphology +
contour detection) rather than a learned detector — this keeps the whole
pipeline explainable and matches the "digital image processing" scope of
the assignment.

Only handles clearly separated products (per the assignment's own note:
"Single object or clearly separated products preferred"). Products packed
edge-to-edge with no visible background gap — e.g. a dense supermarket
shelf photo — will come back as one region, since there's no gap for
classical thresholding to split on; reliably splitting touching objects
needs a trained detector (YOLO, etc.) with bounding-box annotations,
which is out of scope here.
"""

import cv2
import numpy as np


def _background_mask(image: np.ndarray, bg_color: tuple[int, int, int] | None) -> np.ndarray:
    if bg_color is not None:
        diff = np.linalg.norm(image.astype(np.int32) - np.array(bg_color), axis=2)
        mask = (diff > 25).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_products(
    image: np.ndarray,
    bg_color: tuple[int, int, int] | None = None,
    min_area_ratio: float = 0.002,
    pad: int = 6,
) -> list[dict]:
    """Detect candidate product regions in `image`.

    Returns a list of {"bbox": [x, y, w, h], "crop": np.ndarray} sorted
    left-to-right, top-to-bottom.
    """
    height, width = image.shape[:2]
    mask = _background_mask(image, bg_color)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * width * height

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)

        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, width), min(y + h + pad, height)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        detections.append({"bbox": [x0, y0, x1 - x0, y1 - y0], "crop": crop})

    detections.sort(key=lambda d: (d["bbox"][1] // 100, d["bbox"][0]))
    return detections


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Preview detection boxes on an image.")
    parser.add_argument("image")
    parser.add_argument("--out", default="outputs/detection_preview.png")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    dets = detect_products(img)
    preview = img.copy()
    for det in dets:
        x, y, w, h = det["bbox"]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

   