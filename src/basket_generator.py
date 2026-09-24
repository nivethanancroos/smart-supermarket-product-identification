"""Builds synthetic multi-product "basket" images for the demo.

The dataset only contains one product per image, but the assignment asks
the system to detect and count several products in one shot (like a
checkout basket). This module randomly samples a handful of product
images, places them on a plain canvas in a jittered grid, and returns the
composite image along with ground-truth boxes/labels so detection and
classification accuracy can be sanity-checked end to end.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np

from src import config


def _list_dataset_images(data_dir: Path) -> list[tuple[Path, str]]:
    items = []
    for category_dir in sorted(data_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for img_path in category_dir.iterdir():
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                items.append((img_path, category_dir.name))
    return items


def generate_basket_image(
    data_dir: Path = config.DATA_DIR,
    num_items: int = 6,
    canvas_size: tuple[int, int] = (1000, 800),
    cell_size_range: tuple[int, int] = (160, 230),
    seed: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Compose a synthetic basket image.

    Returns (canvas_bgr, ground_truth) where ground_truth is a list of
    {"label": str, "bbox": [x, y, w, h]} dicts.
    """
    rng = random.Random(seed)
    pool = _list_dataset_images(data_dir)
    if len(pool) < num_items:
        raise ValueError(f"Requested {num_items} items but only {len(pool)} images exist")
    chosen = rng.sample(pool, num_items)

    width, height = canvas_size
    canvas = np.full((height, width, 3), config.BASKET_BG_COLOR, dtype=np.uint8)

    # Lay items out on a grid with enough cells for num_items, then jitter
    # each item's position/size within its cell so boxes stay separated.
    cols = int(np.ceil(np.sqrt(num_items * width / height)))
    rows = int(np.ceil(num_items / cols))
    cell_w, cell_h = width // cols, height // rows

    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)

    ground_truth = []
    for (row, col), (img_path, label) in zip(cells, chosen):
        product = cv2.imread(str(img_path))
        if product is None:
            continue

        size = rng.randint(*cell_size_range)
        size = min(size, cell_w - 20, cell_h - 20)
        product = cv2.resize(product, (size, size), interpolation=cv2.INTER_AREA)

        max_jitter_x = max(cell_w - size - 10, 0)
        max_jitter_y = max(cell_h - size - 10, 0)
        x = col * cell_w + 5 + rng.randint(0, max_jitter_x)
        y = row * cell_h + 5 + rng.randint(0, max_jitter_y)

        canvas[y:y + size, x:x + size] = product
        ground_truth.append({"label": label, "bbox": [int(x), int(y), int(size), int(size)]})

    return canvas, ground_truth


def save_basket(canvas: np.ndarray, ground_truth: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "synthetic_basket.png"
    gt_path = output_dir / "synthetic_basket_ground_truth.json"
    cv2.imwrite(str(image_path), canvas)
    gt_path.write_text(json.dumps(ground_truth, indent=2))
    return image_path, gt_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a synthetic basket image for testing.")
    parser.add_argument("--num-items", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    args = parser.parse_args()

    canvas, gt = generate_basket_image(num_items=args.num_items, seed=args.seed)
    image_path, gt_path = save_basket(canvas, gt, Path(args.output_dir))
    print(f"Saved basket image to {image_path}")
    print(f"Saved ground truth to {gt_path}")
