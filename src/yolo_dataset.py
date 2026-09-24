"""Generates a synthetic multi-product dataset with YOLO-format labels.

Uses the same "paste product photos onto a canvas" idea as
basket_generator.py, but produces many images with a variable number of
items and randomized backgrounds so a YOLO detector can learn to find and
classify products directly — replacing the two-stage classical-detection
(detection.py) + CNN-classification (classification.py) pipeline with one
learned model trained on real bounding-box supervision.

The source photos (archive(2)/images/) are real close-up shelf photos, not
clean product cutouts — each one already carries its own background,
lighting and clutter. To keep composites from looking like flat "stickers"
(hard rectangle edges, foreign background patches, a synthetic-clean
canvas), this module:

  - trims each source photo's outer background margin (soft, feathered
    edge, keeping the object-filled center untouched) instead of pasting
    the full rectangle,
  - composites onto a procedurally textured "tabletop" background instead
    of a flat color, with a soft drop shadow under each item,
  - applies per-item rotation/scale/brightness jitter and occasional
    overlap, since real placement is never a clean jittered grid,
  - finishes with camera-style post-processing (blur, sensor noise, JPEG
    recompression) so low-level image statistics resemble a real photo
    instead of a crisp synthetic render.

Usage:
    python -m src.yolo_dataset --n-train 800 --n-val 150
"""

import argparse
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


def _split_pool(pool: list[tuple[Path, str]], val_ratio: float, seed: int):
    """Split per class so no source product photo appears in both splits."""
    by_class: dict[str, list[Path]] = {}
    for path, label in pool:
        by_class.setdefault(label, []).append(path)

    rng = random.Random(seed)
    train_pool, val_pool = [], []
    for label, paths in by_class.items():
        paths = list(paths)
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * val_ratio))
        val_pool += [(p, label) for p in paths[:n_val]]
        train_pool += [(p, label) for p in paths[n_val:]]
    return train_pool, val_pool


def _foreground_alpha(img: np.ndarray) -> np.ndarray:
    """Soft alpha mask that trims background-colored margins.

    These source photos are real shelf close-ups: the product usually
    fills the center of the frame and any background (shelf, other items)
    shows up at the periphery. We estimate the background color from the
    image border, then only allow pixels in the outer ring to become
    transparent if they match it — the center always stays fully opaque,
    so we never eat into the product itself, even on frames with no
    separable background at all (bag-of-chips-fills-everything case).
    """
    h, w = img.shape[:2]
    border = max(2, min(h, w) // 20)
    border_px = np.concatenate([
        img[:border, :].reshape(-1, 3),
        img[-border:, :].reshape(-1, 3),
        img[:, :border].reshape(-1, 3),
        img[:, -border:].reshape(-1, 3),
    ]).astype(np.float32)
    bg_color = np.median(border_px, axis=0)

    dist = np.linalg.norm(img.astype(np.float32) - bg_color, axis=2)
    bg_likeness = 1.0 - np.clip(dist / 40.0, 0, 1)  # 1 = looks like background

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    nx, ny = (xx - cx) / cx, (yy - cy) / cy
    radial = np.clip(np.maximum(np.abs(nx), np.abs(ny)), 0, 1)  # 0 center -> 1 corner
    edge_weight = np.clip((radial - 0.55) / 0.45, 0, 1) ** 1.5  # only trim outer ring

    alpha = 1.0 - bg_likeness * edge_weight
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), sigmaX=max(1.0, border / 3))
    return np.clip(alpha, 0, 1)


def _rotate_rgba(rgba: np.ndarray, angle: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    diag = int(np.ceil((h ** 2 + w ** 2) ** 0.5))
    canvas = np.zeros((diag, diag, 4), dtype=np.float32)
    y0, x0 = (diag - h) // 2, (diag - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = rgba
    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle, 1.0)
    rotated = cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    return rotated


def _procedural_background(size: tuple[int, int], rng: random.Random) -> np.ndarray:
    """A rough tabletop/countertop-style texture: base tone + layered
    blurred noise for grain + a soft directional lighting gradient."""
    width, height = size
    palette = [
        (200, 220, 235),  # light wood (BGR-ish warm)
        (150, 180, 210),  # tan wood
        (215, 215, 215),  # light gray counter
        (235, 238, 240),  # off-white table
        (120, 140, 150),  # dark counter
    ]
    base = np.array(rng.choice(palette), dtype=np.float32)
    base += np.array([rng.uniform(-10, 10) for _ in range(3)], dtype=np.float32)

    canvas = np.tile(base, (height, width, 1)).astype(np.float32)

    for scale, strength in ((6, 18), (25, 10), (80, 6)):
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(
            0, strength, size=(max(1, height // scale), max(1, width // scale), 1)
        ).astype(np.float32)
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        canvas += noise[..., None]

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    lx, ly = rng.uniform(0, width), rng.uniform(0, height)
    dist = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2)
    light = 1.0 - 0.25 * (dist / dist.max())
    canvas *= light[..., None]

    return np.clip(canvas, 0, 255).astype(np.uint8)


def _paste_rgba(canvas: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    h, w = rgba.shape[:2]
    ch, cw = canvas.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, cw), min(y + h, ch)
    if x1 <= x0 or y1 <= y0:
        return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = src[..., 3:4] / 255.0
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (region * (1 - alpha) + src[..., :3].astype(np.float32) * alpha).astype(np.uint8)


def _drop_shadow(canvas: np.ndarray, alpha: np.ndarray, x: int, y: int, offset: int = 6) -> None:
    ch, cw = canvas.shape[:2]
    h, w = alpha.shape[:2]
    sx, sy = x + offset, y + offset
    x0, y0 = max(sx, 0), max(sy, 0)
    x1, y1 = min(sx + w, cw), min(sy + h, ch)
    if x1 <= x0 or y1 <= y0:
        return
    shadow_alpha = alpha[y0 - sy:y1 - sy, x0 - sx:x1 - sx] * 0.35
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (region * (1 - shadow_alpha[..., None])).astype(np.uint8)


def _camera_effects(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Push a crisp synthetic render towards phone-camera-photo statistics."""
    out = image.astype(np.float32)

    gain = rng.uniform(0.85, 1.15)
    bias = rng.uniform(-15, 15)
    out = out * gain + bias

    if rng.random() < 0.6:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    noise_sigma = rng.uniform(2, 8)
    out += np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, noise_sigma, out.shape)

    out = np.clip(out, 0, 255).astype(np.uint8)

    quality = rng.randint(55, 90)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def generate_image(
    pool: list[tuple[Path, str]],
    classes: list[str],
    canvas_size: tuple[int, int],
    num_items: int,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    width, height = canvas_size
    canvas = _procedural_background(canvas_size, rng)

    num_items = min(num_items, len(pool))
    chosen = rng.sample(pool, num_items)

    cols = int(np.ceil(np.sqrt(num_items * width / height)))
    rows = int(np.ceil(num_items / cols))
    cell_w, cell_h = width // cols, height // rows

    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)

    labels = []
    for (row, col), (img_path, label) in zip(cells, chosen):
        product = cv2.imread(str(img_path))
        if product is None:
            continue

        alpha = _foreground_alpha(product)
        if rng.random() < 0.5:
            product = cv2.flip(product, 1)
            alpha = cv2.flip(alpha, 1)
        rgba = np.dstack([product, (alpha * 255).astype(np.uint8)])

        min_cell = min(cell_w, cell_h)
        size = max(rng.randint(int(min_cell * 0.6), int(min_cell * 0.95)), 40)
        rgba = cv2.resize(rgba, (size, size), interpolation=cv2.INTER_AREA)

        angle = rng.uniform(-12, 12)
        rgba = _rotate_rgba(rgba, angle)
        rgba = np.clip(rgba, 0, 255).astype(np.uint8)

        # Real placement overlaps a little rather than sitting in isolated
        # cells — allow items to spill slightly outside their grid cell.
        overlap_x = int(cell_w * 0.12)
        overlap_y = int(cell_h * 0.12)
        max_x = col * cell_w + cell_w - rgba.shape[1] + overlap_x
        min_x = col * cell_w - overlap_x
        max_y = row * cell_h + cell_h - rgba.shape[0] + overlap_y
        min_y = row * cell_h - overlap_y
        x = rng.randint(min(min_x, max_x), max(min_x, max_x))
        y = rng.randint(min(min_y, max_y), max(min_y, max_y))

        _drop_shadow(canvas, rgba[..., 3].astype(np.float32) / 255.0, x, y)
        _paste_rgba(canvas, rgba, x, y)

        # Ground-truth box: the rotated alpha's actual occupied extent,
        # not the full padded rotation canvas, clipped to image bounds.
        occ = np.argwhere(rgba[..., 3] > 20)
        if occ.size == 0:
            continue
        by0, bx0 = occ.min(axis=0)
        by1, bx1 = occ.max(axis=0)
        abs_x0, abs_y0 = max(x + bx0, 0), max(y + by0, 0)
        abs_x1, abs_y1 = min(x + bx1, width), min(y + by1, height)
        if abs_x1 <= abs_x0 or abs_y1 <= abs_y0:
            continue

        xc = (abs_x0 + abs_x1) / 2 / width
        yc = (abs_y0 + abs_y1) / 2 / height
        nw = (abs_x1 - abs_x0) / width
        nh = (abs_y1 - abs_y0) / height
        labels.append((classes.index(label), xc, yc, nw, nh))

    canvas = _camera_effects(canvas, rng)
    return canvas, labels


def generate_dataset(
    data_dir: Path = config.DATA_DIR,
    out_dir: Path = config.YOLO_DATASET_DIR,
    n_train: int = 800,
    n_val: int = 150,
    min_items: int = 3,
    max_items: int = 9,
    canvas_size: tuple[int, int] = (720, 720),
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    pool = _list_dataset_images(data_dir)
    classes = sorted({label for _, label in pool})
    train_pool, val_pool = _split_pool(pool, val_ratio, seed)
    rng = random.Random(seed)

    for split, n, split_pool in (("train", n_train, train_pool), ("val", n_val, val_pool)):
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            num_items = rng.randint(min_items, max_items)
            # Casual phone photos aren't always a perfect square crop.
            aspect = rng.choice([(1.0, 1.0), (4, 3), (3, 4)])
            w = canvas_size[0]
            h = int(w * aspect[1] / aspect[0])
            canvas, labels = generate_image(split_pool, classes, (w, h), num_items, rng)
            name = f"{split}_{i:05d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), canvas)
            with open(lbl_dir / f"{name}.txt", "w") as f:
                for cls_id, xc, yc, bw, bh in labels:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    yaml_lines = [f"path: {out_dir.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml_lines += [f"  {i}: {name}" for i, name in enumerate(classes)]
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n")

    print(f"Generated {n_train} train / {n_val} val images to {out_dir}")
    print(f"Classes ({len(classes)}): {classes}")
    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic YOLO-format basket dataset.")
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-val", type=int, default=150)
    parser.add_argument("--min-items", type=int, default=3)
    parser.add_argument("--max-items", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        n_train=args.n_train, n_val=args.n_val,
        min_items=args.min_items, max_items=args.max_items, seed=args.seed,
    )
