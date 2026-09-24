"""Converts D2S's COCO-format annotations into a YOLO-format dataset.

D2S (dataset/d2s_annotations_v1.1 + dataset/d2s_images_v1) ships real
multi-item supermarket shelf photos with COCO-style bbox annotations over
60 branded product categories — unlike yolo_dataset.py's synthetic
composited baskets, these are real photos with natural placement,
occlusion and clutter.

Only the (non-augmented) training and validation splits are converted;
D2S's test split ships with no ground-truth annotations (held out for the
original challenge), and the "augmented" split is a large pre-rendered set
of rotated/relit duplicates that Ultralytics' own training-time
augmentation (mosaic, hsv jitter, flip, degrees, perspective, mixup —
see yolo_train.py) already covers.

Images are symlinked into the output directory rather than copied, since
they're shared with dataset/d2s_images_v1 and copying ~8k images would
needlessly duplicate several GB on a near-full disk.

Usage:
    python -m src.d2s_to_yolo
"""

import json
from pathlib import Path

from src import config

ANNOTATIONS_DIR = config.D2S_RAW_DIR / "d2s_annotations_v1.1" / "annotations"
IMAGES_DIR = config.D2S_RAW_DIR / "d2s_images_v1" / "images"

SPLITS = {"train": "D2S_training.json", "val": "D2S_validation.json"}


def _convert_split(json_name: str, out_dir: Path, split: str, class_id_by_cat: dict[int, int]) -> tuple[int, int]:
    with open(ANNOTATIONS_DIR / json_name) as f:
        data = json.load(f)

    anns_by_image: dict[int, list] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    written, missing = 0, 0
    for im in data["images"]:
        src = IMAGES_DIR / im["file_name"]
        if not src.exists():
            missing += 1
            continue

        link = img_out / im["file_name"]
        if not link.exists():
            link.symlink_to(src.resolve())

        w, h = im["width"], im["height"]
        lines = []
        for ann in anns_by_image.get(im["id"], []):
            x, y, bw, bh = ann["bbox"]
            x, y = max(0.0, min(x, w)), max(0.0, min(y, h))
            bw, bh = max(0.0, min(bw, w - x)), max(0.0, min(bh, h - y))
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            nw, nh = bw / w, bh / h
            lines.append(f"{class_id_by_cat[ann['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_text = "\n".join(lines) + ("\n" if lines else "")
        (lbl_out / f"{Path(im['file_name']).stem}.txt").write_text(label_text)
        written += 1

    return written, missing


def main():
    with open(ANNOTATIONS_DIR / "D2S_training.json") as f:
        categories = sorted(json.load(f)["categories"], key=lambda c: c["id"])
    class_id_by_cat = {c["id"]: i for i, c in enumerate(categories)}
    class_names = [c["name"] for c in categories]

    out_dir = config.D2S_YOLO_DATASET_DIR
    for split, json_name in SPLITS.items():
        written, missing = _convert_split(json_name, out_dir, split, class_id_by_cat)
        print(f"{split}: {written} images linked, {missing} referenced images missing on disk")

    yaml_lines = [f"path: {out_dir.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml_lines += [f"  {i}: {name}" for i, name in enumerate(class_names)]
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n")
    print(f"\nWrote {yaml_path} with {len(class_names)} classes")


if __name__ == "__main__":
    main()
