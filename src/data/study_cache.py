"""Fixed-size study cache for training: one memory-mapped uint8 array.

Layout (``assets/processed/study_cache/<split>/``):
  vol.u8      (N, 64, 336, 336) uint8, raw memmap: raptor preprocessing
              (5 plane slots, 140 mm centre crop, 2-98% windowing per series)
  mask.npy    (N, 64) uint8, valid slices
  index.csv   row -> StudyInstanceUID

Built from an existing ``raptor_gradcam`` output (``studies/*.npz`` already hold
exactly this volume), or from DICOM for studies that have no npz (the test set).

Usage:
    uv run python -m src.data.study_cache --from-npz assets/processed/gradcam_raptor/train_fast
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.explain import raptor

CACHE_DIR = Path("assets/processed/study_cache")
SHAPE = (raptor.MAXS, raptor.IMG, raptor.IMG)


class StudyCache:
    """Read side: ``cache[uid] -> (vol, mask)`` without loading the whole array."""

    def __init__(self, split="train", root=CACHE_DIR):
        d = Path(root) / split
        self.uids = pd.read_csv(d / "index.csv").StudyInstanceUID.tolist()
        self.row = {u: i for i, u in enumerate(self.uids)}
        self.vol = np.memmap(d / "vol.u8", np.uint8, "r", shape=(len(self.uids), *SHAPE))
        self.mask = np.load(d / "mask.npy")

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, uid):
        i = self.row[uid]
        return np.asarray(self.vol[i]), self.mask[i]


def _from_npz(path):
    z = np.load(path)
    return z["vol"], z["mask"]


def _from_dicom(args):
    uid, rows, split = args
    vol, mask, _ = raptor.build_study(uid, rows, str(raptor.DATA_DIR / f"{split}_series"))
    return vol, mask


def build(split, npz_dir=None, root=CACHE_DIR, threads=8):
    d = Path(root) / split
    d.mkdir(parents=True, exist_ok=True)
    series = raptor.load_series_table(split)
    uids = sorted(series)
    vol = np.memmap(d / "vol.u8", np.uint8, "w+", shape=(len(uids), *SHAPE))
    mask = np.zeros((len(uids), raptor.MAXS), np.uint8)

    def load(uid):
        p = Path(npz_dir) / "studies" / f"{uid}.npz" if npz_dir else None
        try:
            return _from_npz(p) if p and p.exists() else _from_dicom((uid, series[uid], split))
        except Exception as e:  # corrupt study: keep an empty volume, mask stays 0
            print(f"  {uid}: {type(e).__name__}: {e}")
            return None, None

    with ThreadPoolExecutor(threads) as pool:
        for i, (v, m) in enumerate(pool.map(load, uids)):
            if v is not None:
                vol[i], mask[i] = v, m
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(uids)}")
    vol.flush()
    np.save(d / "mask.npy", mask)
    pd.DataFrame({"StudyInstanceUID": uids}).to_csv(d / "index.csv", index=False)
    print(f"{len(uids)} studies -> {d} ({vol.nbytes / 1e9:.1f} GB); "
          f"empty {(mask.sum(1) == 0).sum()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--from-npz", default=None, help="raptor_gradcam output dir to copy volumes from")
    ap.add_argument("--out", default=str(CACHE_DIR))
    a = ap.parse_args()
    build(a.split, a.from_npz, a.out)


if __name__ == "__main__":
    main()
