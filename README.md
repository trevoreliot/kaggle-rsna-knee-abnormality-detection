# RSNA Knee Abnormality Detection

Kaggle: [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).
Predict 12 study-level findings (ACL, MCL, menisci, OA, effusion, …) from knee MRI;
the metric is macro ROC-AUC.

The current focus is **Grad-CAM weak supervision**: explain a strong public model,
turn its heatmaps into pseudo-boxes, and later train a detector on those boxes.
Progress and open items live in [`TODO.org`](TODO.org); agent and hardware notes are in
[`GEMINI.md`](GEMINI.md).

## Setup

```sh
uv sync
```

Data (git-ignored) is expected under `assets/big-data/`:

```
assets/big-data/rsna-knee-abnormality-detection/   # competition files + train_series/ DICOMs
    custom_labels/slh/kneelabels-stevenleehans/    # community LLM soft labels
assets/big-data/public_weights/raptor-knee-maxspan/
```

Public model weights (CoaTNet, CC0, 0.928 public LB):

```sh
uvx kaggle datasets download -d dreaddevelopment/raptor-knee-maxspan \
    -p assets/big-data/public_weights/raptor-knee-maxspan --unzip
```

## What works

| Module | Purpose |
|---|---|
| `src/explain/raptor.py` | Loads the public "raptor" CoaTNet and reproduces its preprocessing (64 slices / 5 plane slots, 140 mm crop, 62 three-slice windows). Matches the checkpoint's gold AUC (0.920 vs 0.921). |
| `src/explain/raptor_gradcam.py` | Grad-CAM / LayerCAM per label and window at CoaTNet stages 1–3, map ensembles, deletion faithfulness and blob-compactness metrics, HTML gallery. |
| `src/explain/pseudo_boxes.py` | Two-level pseudo-boxes: region box from 12×12 Grad-CAM, peak point and tight box from the LayerCAM ensemble, per-label choice (`BOX_LEVEL`), model-based box scoring. |
| `src/explain/yolo_dataset.py` | Exports pseudo-boxes as an Ultralytics dataset (2.5D slices, split by study, gold studies in val). |
| `src/explain/yolo_eval.py` | Tests whether the pseudo-box YOLO helps classification (as a score and as a cropper for raptor). It does not. |
| `src/data/laterality.py` | Left/right knee per study (vote of DICOM tag, series description, report text, geometry; 99.8% resolved) and the medial image direction per series. |
| `src/data/study_cache.py` | 32 GB uint8 memmap of every study's 64-slice volume (raptor preprocessing), for training. |
| `src/data/labels.py` | Soft labels from the public LLM report-label tables; `consensus4` = mean of four sources. |
| `src/models/mil25d.py` | Phase 5: our own 2.5D attention-MIL classifier (timm backbone on 3-slice windows), 5-fold CV by report group, 58 gold studies held out. |
| `src/data/extract_2d_slices.py` | Extracts every train series to per-series-normalised PNGs. |

The older `src/models/` prototypes (YOLO cropper, CNN classifier, 2.5D transformer) and
`src/data/dagster_assets.py` are **prototype stubs** that still run on random data.

## CLI

Grad-CAM on the 58 manually labelled ("gold") studies:

```sh
uv run python -m src.explain.raptor_gradcam --studies gold \
    --maps L3 L1+L2 LC1 LC1+L3 --out assets/processed/gradcam_raptor/gold_layercam
```

- `--maps`: `L<k>` Grad-CAM or `LC<k>` LayerCAM at stage k (1: 48×48, 2: 24×24, 3: 12×12);
  join with `+` for a geometric-mean ensemble.
- `--studies`: `gold`, an integer N (random non-gold studies) or a file of StudyInstanceUIDs.
- `--reuse DIR`: load study volumes saved by an earlier run instead of reading DICOMs.
- `--gallery-only`: rebuild `gallery.html` from an existing output directory.

- `--fast`: bulk mode for pseudo-boxes: maps `L3 LC1+L3` only, no deletion test or gallery,
  resumable (re-run the same command to continue).  ~2.6 s/study.

Outputs: `preds.csv`, `deletion.csv`, `summary.csv`, `studies/*.npz`, `gallery.html`.

Bulk run over the whole train set (~3 h, ~24 GB):

```sh
uv run python -m src.explain.raptor_gradcam --studies all --fast \
    --out assets/processed/gradcam_raptor/train_fast
```

Grad-CAM on our own model (fold 0 never trained on the gold studies; ConvNeXt-T at 256 px
has stage grids 0-3 = 64/32/16/8), then boxes from its 16x16 region and 64x64 point maps:

```sh
uv run python -m src.explain.raptor_gradcam --studies gold --maps L2 L1 LC0 LC1+L2 \
    --model assets/processed/mil_runs/convnext_tiny_cons4_5fold/fold0/best.pt \
    --reuse assets/processed/gradcam_raptor/gold_multi --out assets/processed/gradcam_own/gold_cons4_f0
uv run python -m src.explain.pseudo_boxes --src assets/processed/gradcam_own/gold_cons4_f0 \
    --region-map L2 --point-map LC0 --all-labels --eval
```

Pseudo-boxes from a Grad-CAM output directory that has `L3` and `LC1+L3` maps:

```sh
uv run python -m src.explain.pseudo_boxes \
    --src assets/processed/gradcam_raptor/gold_layercam --eval
```

Writes `<src>/boxes/pseudo_boxes.csv` (final box in `x0..y1`, YOLO-normalised `cx cy w h`,
both on the 336 px / 140 mm crop), `box_scores.csv` with `--eval`, and `boxes.html`.
ACL and Medial Meniscus are excluded for now (`--all-labels` to include them).

YOLO on the pseudo-boxes (no horizontal flips: medial / lateral classes depend on image side):

```sh
uv run python -m src.explain.yolo_dataset --src assets/processed/gradcam_raptor/train_fast
uv run yolo detect train data=assets/processed/yolo_pseudo/data.yaml model=yolo11s.pt imgsz=384 \
    epochs=100 patience=20 batch=64 fliplr=0 hsv_h=0 hsv_s=0 \
    project=$PWD/assets/processed/yolo_runs name=yolo11s_v1
uv run python -m src.explain.yolo_eval --weights assets/processed/yolo_runs/yolo11s_v1/weights/best.pt
```

Laterality (writes `assets/processed/laterality_<split>.csv` and
`series_orientation_<split>.csv`, ~15 s):

```sh
uv run python -m src.data.laterality --split train
```

Own model (Phase 5).  Label tables need the public CSVs under `custom_labels/`
(`stevenleehans`, `pilkwang`, `lixin73`, `laymond`; see `src/data/labels.py`):

```sh
uv run python -m src.data.study_cache --from-npz assets/processed/gradcam_raptor/train_fast
uv run python -m src.data.labels
uv run python -m src.models.mil25d --fold all --name convnext_tiny_cons4_5fold   # ~2.2 h
uv run python -m src.models.mil25d --fold 0 --img 384 --grad-ckpt               # options
```

Folds (`assets/processed/folds.csv`) are rebuilt deterministically when missing.

Registered pipelines:

```sh
uv run python main.py --pipeline raptor_gradcam   # same as the Grad-CAM run with defaults
uv run python main.py --pipeline mil25d           # fold 0 of the own model with defaults
```

## Findings so far

- 12×12 Grad-CAM gives compact, faithful regions; 48×48 maps are faithful but speckled.
- CoaTNet Grad-CAM has a spurious hot cell at grid (0, 0); the box generator suppresses it.
- Pseudo-boxes beat random boxes on the knee for 10 of 12 labels (81–100%). ACL and
  medial meniscus are not reliable yet.
- A YOLO11s trained on 32k pseudo-boxes learns the large findings (Baker's, effusion) but
  does not improve classification: as a score it lowers raptor's AUC when blended, and
  re-centring or zooming raptor's crop on its detections does not help (0.943 -> 0.943 / 0.931).

- Own model (ConvNeXt-T, 256 px, consensus labels): out-of-fold macro AUC 0.899, 5-fold gold
  0.899 (raptor 0.920).  Consensus labels gained more than 384 px input, at a third of the cost.
- Grad-CAM on the own model (`raptor_gradcam --model <best.pt>`): 16x16 maps 100% faithful
  (raptor 12x12: 92%); pseudo-boxes beat random boxes for all 12 labels, and medial meniscus
  boxes land on the medial meniscus (64/64 on the medial half).  ACL boxes follow secondary
  signs (effusion, lateral bone bruise) rather than the ligament.

Details and numbers: [`TODO.org`](TODO.org), Phase 4.
