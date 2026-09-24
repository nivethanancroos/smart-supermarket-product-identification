"""FastAPI backend exposing the product detection pipeline over HTTP.

Run with:
    uvicorn api.app:app --reload --port 8000

Then open http://localhost:8000 in a browser.
"""

import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from src import classification, config, detection, preprocessing
from src.visualize import draw_annotations
from src.yolo_detector import YoloProductDetector

app = FastAPI(title="Smart Supermarket Product Identification API")

_classifier: classification.ProductClassifier | None = None
_yolo: YoloProductDetector | None = None


def get_classifier() -> classification.ProductClassifier:
    global _classifier
    if _classifier is None:
        if not config.MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Classical model not trained yet. Run `python -m src.train` first.",
            )
        _classifier = classification.ProductClassifier()
    return _classifier


def get_yolo() -> YoloProductDetector:
    global _yolo
    if _yolo is None:
        if not config.D2S_YOLO_MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="D2S YOLO model not trained yet. Run `python -m src.d2s_to_yolo` then "
                       "`python -m src.yolo_train --data d2s_yolo/dataset.yaml --name d2s "
                       "--out models/d2s_yolo_best.pt` first.",
            )
        _yolo = YoloProductDetector(weights_path=config.D2S_YOLO_MODEL_PATH)
    return _yolo


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), detector: str = Form("classical")):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    if detector not in ("classical", "yolo"):
        raise HTTPException(status_code=400, detail="detector must be 'classical' or 'yolo'")

    raw = await file.read()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    if detector == "yolo":
        yolo = get_yolo()
        detections = [
            {"bbox": d["bbox"], "label": d["label"], "confidence": round(d["confidence"], 4)}
            for d in yolo.detect(image)
        ]
    else:
        classifier = get_classifier()

        preprocessed = preprocessing.preprocess_for_detection(image)
        raw_detections = detection.detect_products(preprocessed)

        # A single tightly-framed product photo may not have enough background
        # contrast to segment into contours — fall back to treating the whole
        # image as one product so the demo still returns a result.
        if not raw_detections:
            height, width = image.shape[:2]
            raw_detections = [{"bbox": [0, 0, width, height], "crop": image}]

        detections = []
        for det in raw_detections:
            crop = preprocessing.preprocess_crop_for_classifier(det["crop"])
            label, confidence = classifier.predict(crop)
            detections.append({"bbox": det["bbox"], "label": label, "confidence": round(confidence, 4)})

    annotated = draw_annotations(image, detections)

    ok, buffer = cv2.imencode(".png", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image")
    annotated_b64 = "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")

    counts: dict[str, int] = {}
    for det in detections:
        counts[det["label"]] = counts.get(det["label"], 0) + 1
    total = len(detections)
    percentages = {label: round(count / total * 100, 2) for label, count in counts.items()} if total else {}

    return {
        "detector": detector,
        "total_products": total,
        "detections": detections,
        "category_counts": counts,
        "category_percentages": percentages,
        "annotated_image": annotated_b64,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "classical_model_available": config.MODEL_PATH.exists(),
        "yolo_model_available": config.D2S_YOLO_MODEL_PATH.exists(),
    }


frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
