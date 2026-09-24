"""Fine-tunes a pretrained YOLOv8 detector on the synthetic labeled basket
dataset (see yolo_dataset.py). This is a one-stage alternative to the
classical detection.py + classification.py pipeline: one model learns to
find AND classify products directly from bounding-box supervision.

Usage:
    python -m src.yolo_dataset --n-train 800 --n-val 150   # once
    python -m src.yolo_train --epochs 40
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from src import config


def main():
    parser = argparse.ArgumentParser(description="Train the YOLO product detector.")
    parser.add_argument("--data", default=str(config.YOLO_DATASET_DIR / "dataset.yaml"))
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained YOLO checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2,
                         help="Dataloader worker processes. Keep low on machines with "
                              "limited RAM (each worker holds decoded/augmented images in "
                              "memory) to avoid swap thrashing / system freezes.")
    parser.add_argument("--device", default=None,
                         help="e.g. 0 for first GPU, or 'cpu'. Leave unset to auto-detect.")
    parser.add_argument("--name", default="train",
                         help="Run folder name under outputs/yolo_runs/ — use a distinct name "
                              "per dataset so resume-from-checkpoint doesn't pick up an "
                              "unrelated run (e.g. a different class count).")
    parser.add_argument("--out", default=str(config.YOLO_MODEL_PATH),
                         help="Where to copy the best checkpoint when training finishes.")
    parser.add_argument("--patience", type=int, default=15,
                         help="Stop early if val mAP hasn't improved for this many epochs.")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"Dataset not found at {args.data}. Run `python -m src.yolo_dataset` first.")

    # These are passed on every training call, resumed or not. Ultralytics'
    # resume path normally restores settings from the checkpoint's own
    # args.yaml — but if the checkpoint turns out to be a *finished* run
    # (optimizer state already stripped), it silently falls back to a
    # fresh model.train() call using whatever kwargs were given. Passing
    # nothing there previously meant that fallback ran on pure Ultralytics
    # defaults (the bundled coco8 toy dataset, 80 classes, epochs=100)
    # instead of this project's dataset — always supply the real args so
    # that fallback is correct too.
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        cache=False,
        patience=args.patience,
        project=str(config.OUTPUT_DIR / "yolo_runs"),
        name=args.name,
        exist_ok=True,
        # Augmentation: on top of Ultralytics' defaults (mosaic, hsv jitter,
        # horizontal flip), add mild rotation/perspective/mixup so the model
        # sees more pose/lighting variation than the axis-aligned synthetic
        # basket layout alone provides.
        degrees=10.0,
        perspective=0.0005,
        mixup=0.1,
    )

    # Ultralytics writes weights/last.pt after every epoch. If a previous
    # run under this same --name got interrupted, resume from that
    # checkpoint instead of starting over from epoch 0.
    last_checkpoint = config.OUTPUT_DIR / "yolo_runs" / args.name / "weights" / "last.pt"
    if last_checkpoint.exists():
        print(f"Found existing checkpoint at {last_checkpoint} — attempting to resume.")
        model = YOLO(str(last_checkpoint))
        results = model.train(resume=True, **train_kwargs)
    else:
        model = YOLO(args.model)
        results = model.train(**train_kwargs)

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights, args.out)
    print(f"\nSaved best YOLO weights to {args.out}")


if __name__ == "__main__":
    main()
