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
| `src/data/extract_2d_slices.py` | Extracts every train series to per-series-normalised PNGs. |

`src/models/` (YOLO cropper, CNN classifier, 2.5D transformer) and
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

Outputs: `preds.csv`, `deletion.csv`, `summary.csv`, `studies/*.npz`, `gallery.html`.

Pseudo-boxes from a Grad-CAM output directory that has `L3` and `LC1+L3` maps:

```sh
uv run python -m src.explain.pseudo_boxes \
    --src assets/processed/gradcam_raptor/gold_layercam --eval
```

Writes `<src>/boxes/pseudo_boxes.csv` (final box in `x0..y1`, YOLO-normalised `cx cy w h`,
both on the 336 px / 140 mm crop), `box_scores.csv` with `--eval`, and `boxes.html`.
ACL and Medial Meniscus are excluded for now (`--all-labels` to include them).

Registered pipelines:

```sh
uv run python main.py --pipeline raptor_gradcam   # same as the Grad-CAM run with defaults
```

## Findings so far

- 12×12 Grad-CAM gives compact, faithful regions; 48×48 maps are faithful but speckled.
- CoaTNet Grad-CAM has a spurious hot cell at grid (0, 0); the box generator suppresses it.
- Pseudo-boxes beat random boxes on the knee for 10 of 12 labels (81–100%). ACL and
  medial meniscus are not reliable yet.

Details and numbers: [`TODO.org`](TODO.org), Phase 4.
