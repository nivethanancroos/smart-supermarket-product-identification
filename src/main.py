"""End-to-end pipeline: preprocessing -> detection -> classification -> report.

Usage:
    python -m src.main --generate-basket --num-items 6
    python -m src.main --image path/to/basket.jpg
"""

import argparse
import json
from pathlib import Path

import cv2

from src import basket_generator, classification, config, detection, preprocessing, report
from src.visualize import draw_annotations



def score_against_ground_truth(detections: list[dict], ground_truth: list[dict]) -> float:
    """Rough accuracy check for the demo: fraction of ground-truth items
    whose predicted label matches, matched by nearest box center."""
    if not ground_truth:
        return 0.0

    def center(bbox):
        x, y, w, h = bbox
        return x + w / 2, y + h / 2

    correct = 0
    used = set()
    for gt in ground_truth:
        gt_cx, gt_cy = center(gt["bbox"])
        best_idx, best_dist = None, float("inf")
        for i, det in enumerate(detections):
            if i in used:
                continue
            dcx, dcy = center(det["bbox"])
            dist = (dcx - gt_cx) ** 2 + (dcy - gt_cy) ** 2
            if dist < best_dist:
                best_dist, best_idx = dist, i
        if best_idx is not None:
            used.add(best_idx)
            if detections[best_idx]["label"] == gt["label"]:
                correct += 1

    return correct / len(ground_truth)


def main():
    parser = argparse.ArgumentParser(description="Run the smart product identification pipeline.")
    parser.add_argument("--image", help="Path to a basket image to analyze")
    parser.add_argument("--generate-basket", action="store_true",
                         help="Generate a synthetic basket image from the dataset instead")
    parser.add_argument("--num-items", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", default=str(config.MODEL_PATH))
    parser.add_argument("--classes", default=str(config.CLASSES_PATH))
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    parser.add_argument("--detector", choices=["classical", "yolo"], default="yolo",
                         help="classical = OpenCV detection + CNN classification (two stages, 25 classes); "
                              "yolo = single trained YOLO model on D2S (detection + classification together, 60 classes)")
    parser.add_argument("--yolo-weights", default=str(config.D2S_YOLO_MODEL_PATH))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = None
    bg_color = None

    if args.generate_basket:
        image, ground_truth = basket_generator.generate_basket_image(
            num_items=args.num_items, seed=args.seed,
        )
        basket_generator.save_basket(image, ground_truth, output_dir)
        bg_color = config.BASKET_BG_COLOR
    elif args.image:
        image = preprocessing.load_image(args.image)
    else:
        parser.error("Provide --image <path> or --generate-basket")

    if args.detector == "yolo":
        yolo = YoloProductDetector(weights_path=args.yolo_weights)
        detections = yolo.detect(image)
        print(f"Detected {len(detections)} products (YOLO)")
    else:
        preprocessed = preprocessing.preprocess_for_detection(image)
        raw_detections = detection.detect_products(preprocessed, bg_color=bg_color)
        print(f"Detected {len(raw_detections)} candidate product regions")

        classifier = classification.ProductClassifier(model_path=args.model, classes_path=args.classes)

        detections = []
        for det in raw_detections:
            crop = preprocessing.preprocess_crop_for_classifier(det["crop"])
            label, confidence = classifier.predict(crop)
            detections.append({"bbox": det["bbox"], "label": label, "confidence": confidence})

    annotated = draw_annotations(image, detections)
    annotated_path = output_dir / "annotated_result.png"
    cv2.imwrite(str(annotated_path), annotated)
    print(f"Saved annotated result to {annotated_path}")

    report.generate_report(detections, output_dir)

    if ground_truth is not None:
        accuracy = score_against_ground_truth(detections, ground_truth)
        print(f"\nGround-truth match rate (demo sanity check): {accuracy:.2%}")
        (output_dir / "detections.json").write_text(json.dumps(detections, indent=2))


if __name__ == "__main__":
    main()
