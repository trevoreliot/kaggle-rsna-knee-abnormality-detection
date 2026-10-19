"""Ultralytics YOLO dataset from two-level pseudo-boxes.

One image per (study, slice) that carries at least one pseudo-box, plus
background slices from studies with no box.  Images are the 336 px / 140 mm
crop the raptor model sees, 2.5D: channels are slices (s-1, s, s+1) of the
same plane slot (clamped at the slot edges), like the model's 3-slice windows.

Split by study: every gold study goes to val (so detection can later be read
against the manual labels), plus a random ``--val-frac`` of the rest.

Training note: do not use horizontal flips.  Medial / lateral classes depend
on image side (coronal / axial), so ``fliplr`` would swap their meaning; HSV
hue / saturation jitter would also mix the slice channels.

Usage:
    uv run python -m src.explain.yolo_dataset --src assets/processed/gradcam_raptor/train_fast
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.explain import raptor
from src.explain.pseudo_boxes import BOX_LEVEL

OUT_DIR = Path("assets/processed/yolo_pseudo")
CLASSES = [l for l in raptor.LABELS if BOX_LEVEL.get(l)]


def _rgb(vol, mask, s):
    """Slices (s-1, s, s+1) within s's plane slot, missing neighbours -> s."""
    slot = raptor.SLOT_OF_SLICE
    out = []
    for t in (s - 1, s, s + 1):
        ok = 0 <= t < len(vol) and slot[t] == slot[s] and mask[t]
        out.append(vol[t if ok else s])
    return np.stack(out, -1)


def _export_study(job):
    """Write one study's images and label files; returns the image count."""
    uid, split, slices, rows, out, npz = job
    z = np.load(npz)
    vol, mask = z["vol"], z["mask"]
    for s in slices:
        stem = f"{uid}_{s:02d}"
        # cv2 writes BGR: reverse so (s-1, s, s+1) reads as RGB in Ultralytics.
        cv2.imwrite(str(out / "images" / split / f"{stem}.png"), _rgb(vol, mask, s)[..., ::-1])
        lines = [f"{CLASSES.index(lab)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                 for sl, lab, cx, cy, w, h in rows if sl == s]
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(slices)


def build(src, out=OUT_DIR, val_frac=0.1, bg_per_study=3, workers=8, seed=0):
    src, out = Path(src), Path(out)
    boxes = pd.read_csv(src / "boxes" / "pseudo_boxes.csv")
    boxes = boxes[boxes.label.isin(CLASSES)]
    preds = pd.read_csv(src / "preds.csv")
    rng = np.random.default_rng(seed)

    studies = sorted(preds.study.unique())
    gold = set(preds.loc[preds.gold.notna(), "study"])
    rest = [u for u in studies if u not in gold]
    val = gold | set(rng.choice(rest, int(len(rest) * val_frac), replace=False))

    for d in ("images", "labels"):
        for sp in ("train", "val"):
            (out / d / sp).mkdir(parents=True, exist_ok=True)

    by_study = {u: g for u, g in boxes.groupby("study")}
    jobs = []
    for u in studies:
        split = "val" if u in val else "train"
        npz = src / "studies" / f"{u}.npz"
        if u in by_study:
            g = by_study[u]
            rows = list(zip(g.slice, g.label, g.cx, g.cy, g.w, g.h))
            jobs.append((u, split, sorted(int(s) for s in g.slice.unique()), rows, out, npz))
        elif bg_per_study:
            mask = np.load(npz)["mask"]
            valid = np.flatnonzero(mask)
            pick = rng.choice(valid, min(bg_per_study, len(valid)), replace=False)
            jobs.append((u, split, sorted(int(s) for s in pick), [], out, npz))

    with ProcessPoolExecutor(workers) as pool:
        n = sum(pool.map(_export_study, jobs, chunksize=16))

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    n_bg = sum(len(j[2]) for j in jobs if not j[3])
    n_val = sum(len(j[2]) for j in jobs if j[1] == "val")
    print(f"{n} images ({n_bg} background) from {len(jobs)} studies; "
          f"val {n_val} images / {sum(1 for j in jobs if j[1] == 'val')} studies "
          f"({len(gold)} gold) -> {out / 'data.yaml'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="raptor_gradcam output dir with boxes/")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--bg-per-study", type=int, default=3,
                    help="background slices per study without boxes")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    build(a.src, a.out, a.val_frac, a.bg_per_study, a.workers)


if __name__ == "__main__":
    main()
