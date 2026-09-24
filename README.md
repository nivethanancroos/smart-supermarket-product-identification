# Smart Supermarket Product Identification System

EC9570 (Digital Image Processing) group assignment — detects products in a
photo, classifies each one, counts them per category, and generates a
statistical summary report (console table, CSV, charts) plus a drag-and-drop
web demo.

Two detectors exist side by side:

- **YOLO on D2S (primary)** — a single trained YOLOv8 model that finds and
  classifies products in one pass, fine-tuned on real multi-item supermarket
  photos (60 branded product categories). This is the one to use.
- **Classical (legacy)** — the original two-stage assignment pipeline
  (OpenCV contour detection + a ResNet18 classifier on 25 broad categories,
  trained on single-product photos with a synthetic basket compositor to
  fake multi-item scenes). Kept for reference/comparison; see
  [Legacy classical pipeline](#legacy-classical-pipeline) at the bottom.

## Quick start (most people want this)

You were handed this repo with the trained YOLO weights already in
`models/d2s_yolo_best.pt` — you do **not** need the dataset to run or test
anything below.

```bash
pip install -r requirements.txt
```

**Web console** (drag & drop an image, see it side by side with detected
boxes + a per-category count sidebar):

```bash
uvicorn api.app:app --reload --port 8000
```

Open http://localhost:8000, drop in a photo. Select the "YOLO — D2S
supermarket detector" option (default).

**Command line**, on any photo:

```bash
python -m src.main --image path/to/photo.jpg
```

Outputs land in `outputs/`: `annotated_result.png` (boxes + labels),
`report.csv`, `category_bar_chart.png`, `category_pie_chart.png`, plus a
console-printed total + per-category table.

Note: the D2S model was trained on its own 60 branded (mostly German/EU)
grocery products — see the class list in `d2s_yolo/dataset.yaml` if you
regenerate it, or in `models/d2s_yolo_best.pt`'s embedded names. A photo of
unrelated products will still produce boxes, but the labels will be whatever
D2S class looks closest, not necessarily correct.

## What's in this handoff vs. what isn't

Only code + the trained model were shared — the datasets and generated
training artifacts were deliberately left out because they're large,
reproducible, and not needed to run or test the pipeline:

| Excluded | Size | Why |
|---|---|---|
| `dataset/` (raw D2S download) | 6.8 GB | Only needed to retrain from scratch — see below |
| `d2s_yolo/` (converted YOLO-format dataset) | ~64 MB | Regenerated from `dataset/` by `src/d2s_to_yolo.py` |
| `yolo_dataset/`, `archive(2)/` | ~3.8 GB | Belong to the legacy classical/synthetic pipeline |
| `outputs/`, `runs/` | ~230 MB | Training logs/plots/checkpoints from past runs — regenerated fresh each time you train or run the pipeline |
| `yolov8n.pt`, `yolo26n.pt` | ~12 MB | Pretrained base checkpoints — `ultralytics` auto-downloads these on first use if missing |
| `models/yolo_best.pt` | 6 MB | Superseded — the old 10-class synthetic-dataset model, replaced by `d2s_yolo_best.pt` |
| `test_images/`, `Project-1.pdf` | ~2.6 MB | Old demo images / assignment brief, not needed to run anything |

**Included:** `src/`, `api/`, `frontend/`, `requirements.txt`, this
`README.md`, and `models/d2s_yolo_best.pt` (the trained D2S detector).

If you also want the legacy classical pipeline to work without retraining
it yourself, ask for `models/best_model.pth` + `models/classes.json` too —
those weren't included by default since the D2S detector is the one
actually in use.

## Retraining from scratch (optional)

Only needed if you want to reproduce or improve on the D2S model itself —
skip this if you're just testing/demoing with the provided weights.

1. Get the D2S dataset — it's the MVTec "Densely Segmented Supermarket
   (D2S) Dataset", published at **https://www.mvtec.com/research** (search
   "D2S dataset" there for the download; ~6.8 GB, both the annotations and
   images archives). Licensed CC BY-NC-SA 4.0 (non-commercial, share-alike
   — fine for this assignment, keep that in mind if you reuse it elsewhere).
   Place it at the project root as:
   ```
   dataset/d2s_annotations_v1.1/annotations/*.json
   dataset/d2s_images_v1/images/*.jpg
   ```
2. Convert COCO-format annotations to YOLO format (images are symlinked,
   not copied, so this is fast and uses almost no extra disk):
   ```bash
   python -m src.d2s_to_yolo
   ```
   This writes `d2s_yolo/` (images/labels for train+val, `dataset.yaml`).
3. Train:
   ```bash
   python -m src.yolo_train --data d2s_yolo/dataset.yaml --model yolov8n.pt \
     --epochs 40 --imgsz 512 --batch 8 --workers 1 --device 0 \
     --name d2s --patience 15 --out models/d2s_yolo_best.pt
   ```
   Notes if you're on a laptop with limited RAM/VRAM (this was tuned on a
   4GB-VRAM / 7.5GB-RAM machine):
   - `--device 0` uses the GPU if you have one (`cpu` otherwise) — training
     on CPU on a RAM-constrained machine risks swapping/freezing.
   - `--workers 1` keeps only one background process decoding images at a
     time; raise it if you have RAM/cores to spare, for faster training.
   - `--patience 15` stops early if validation mAP plateaus, instead of
     always running the full epoch count.
   - Always give a fresh `--name` per experiment — the script auto-resumes
     from `outputs/yolo_runs/<name>/weights/last.pt` if one already exists
     under that name, which you don't want when starting a genuinely new
     run with different settings.
   - It auto-resumes if interrupted (Ctrl+C is safe) — rerun the same
     command and it'll continue from the last checkpoint under that name.

   A 40-epoch run took about an hour on an RTX 3050 laptop GPU and reached
   mAP50 ≈ 0.69 / mAP50-95 ≈ 0.54 on D2S's validation split. Some visually
   near-identical classes (e.g. different apple varieties, different tea
   flavors in the same box) are inherently harder and may need more epochs
   or targeted data to improve.

## Project structure

```
src/
  config.py          shared paths/constants
  d2s_to_yolo.py      COCO -> YOLO label conversion for D2S
  yolo_train.py       trains the YOLO detector (see command above)
  yolo_detector.py     YoloProductDetector — loads weights, runs inference
  visualize.py         draws detection boxes/labels on an image
  main.py              CLI entry point: detect -> annotate -> report
  report.py            counts detections per category, writes CSV/charts
  preprocessing.py, detection.py, basket_generator.py,
  classification.py, train.py, yolo_dataset.py   legacy classical pipeline
api/
  app.py               FastAPI backend (/api/analyze, /api/health)
frontend/
  index.html           drag-and-drop web console
models/
  d2s_yolo_best.pt      trained D2S detector (included)
```

## Legacy classical pipeline

The original assignment pipeline is still present and selectable (via
`--detector classical` on the CLI, or the "Classical" option in the web
console), but needs `models/best_model.pth` + `models/classes.json` to run
— see "What's in this handoff" above. To retrain it from scratch you'd also
need `archive(2)/images/` (25 grocery categories, single product per photo,
not included). See the module breakdown below if you're picking up that
half of the assignment:

1. **`src/preprocessing.py`** — image loading, denoising, resizing.
2. **`src/basket_generator.py`** — composes a synthetic multi-product
   "basket" image from the single-product dataset (with ground-truth boxes)
   since that dataset has no natural multi-item photos to detect on.
3. **`src/detection.py`** — OpenCV contour-based segmentation: separates
   products from the background via color-distance thresholding (when the
   background color is known, e.g. the synthetic basket) or Otsu
   thresholding (unknown background, e.g. a real uploaded photo) +
   morphology, then returns bounding boxes and crops via contour detection.

   **Scope limitation:** this only separates *clearly separated* products.
   Products packed edge-to-edge with no visible background gap — e.g. a
   dense supermarket shelf photo — come back as one region, since there's
   no gap for classical thresholding to split on. This is exactly the gap
   the YOLO/D2S detector above was built to close.
4. **`src/train.py`** — fine-tunes a pretrained ResNet18 on the 25
   categories, saves `models/best_model.pth` + `models/classes.json`, plus
   training curves / confusion matrix / a per-class report under `outputs/`.
   Run with `python -m src.train --epochs 12`.
5. **`src/classification.py`** — loads the trained model and predicts a
   label + confidence for a single product crop.

Suggested module split for two team members (matches the assignment's
"each member owns one or more core modules" requirement):

- **Member A:** `preprocessing.py`, `detection.py`, `basket_generator.py`
- **Member B:** `train.py`, `classification.py`, `report.py`
- `main.py` is the integration point.

## GitHub workflow

- Each member works on a feature branch for their module(s) and opens a
  pull request into `main` — contribution history is reviewed at the demo.
- No formal report is required; be ready to explain the module you built
  and why (e.g. why YOLO over contour detection for dense scenes, why
  transfer learning, how the report numbers are computed) during the
  10-minute demo/viva.
