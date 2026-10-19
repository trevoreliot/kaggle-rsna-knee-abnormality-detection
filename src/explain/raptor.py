"""Public "raptor" CoaTNet knee model and the preprocessing it expects.

Weights: Kaggle dataset ``dreaddevelopment/raptor-knee-maxspan`` (CC0), file
``raptor_ft_coatnet_v5_full_swa.pt`` -- 0.928 public LB as a single model.

Preprocessing mirrors the "maxspan-v5" arm of ``notebooks/kaggle/baseline.py``:
64 slices per study drawn from five plane/sequence slots, 140 mm centre crop
using DICOM PixelSpacing, 336 px cache, 62 three-slice windows upsampled to 384.
"""

import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydicom.pixels import apply_modality_lut

DATA_DIR = Path("assets/big-data/rsna-knee-abnormality-detection")
WEIGHTS = Path(
    "assets/big-data/public_weights/raptor-knee-maxspan/"
    "raptor_ft_coatnet_v5_full_swa.pt"
)

IMG = 336
RES = 384
CROP_MM = 140.0
SPAN_LO, SPAN_HI = 0.02, 0.98
K_EVAL = 62
# (plane, Fluid_Sensitive preference (-1 = any), number of slices)
SLOTS = [
    ("Sagittal", 1, 18),
    ("Sagittal", 0, 14),
    ("Coronal", 1, 12),
    ("Coronal", 0, 8),
    ("Axial", -1, 12),
]
SLOT_NAMES = ["Sag fluid", "Sag T1/PD", "Cor fluid", "Cor T1/PD", "Axial"]
MAXS = sum(k for _, _, k in SLOTS)
SLOT_OF_SLICE = np.repeat(np.arange(len(SLOTS)), [k for _, _, k in SLOTS])

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class RaptorClassifier(nn.Module):
    """Backbone per 3-slice window, then per-label attention pooling over windows."""

    def __init__(self, backbone, f_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(f_dim)
        self.att = nn.Sequential(
            nn.Linear(f_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n)
        )
        self.clsW = nn.Parameter(torch.zeros(n, f_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        self.n = n

    def attention(self, feats):
        """Per-label softmax weights over windows: (B, K, F) -> (B, K, n)."""
        return torch.softmax(self.att(self.norm(feats)), dim=1)

    def head(self, feats):
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def forward(self, x):
        b, k = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return self.head(f.view(b, k, -1))


def load_model(device, path=WEIGHTS):
    # weights_only=True: the checkpoint holds only tensors and plain metadata.
    ck = torch.load(path, map_location="cpu", weights_only=True)
    bb = timm.create_model(
        ck["arch"], pretrained=False, num_classes=0, in_chans=3, global_pool="avg"
    )
    model = RaptorClassifier(bb, f_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    assert list(ck["lab"]) == LABELS, ck["lab"]
    return model.eval().to(device)


# --------------------------------------------------------------------------- #
# DICOM -> study volume
# --------------------------------------------------------------------------- #
def _order_series(series_dir):
    """Sort a series' files along the slice normal; return [(path, spacing)], median spacing."""
    recs = []
    for f in glob.glob(f"{series_dir}/*.dcm"):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                normal = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                pos = float(np.dot(np.array(ipp, float), normal))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            recs.append((pos, f, float(ps[0]) if ps is not None else 0.5))
        except Exception:
            recs.append((0.0, f, 0.5))
    recs.sort(key=lambda r: r[0])
    med_ps = float(np.median([r[2] for r in recs])) if recs else 0.5
    return [(f, ps) for _, f, ps in recs], med_ps


def _read_px(path):
    d = pydicom.dcmread(path)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def _mm_crop_resize(a, ps, crop_mm=CROP_MM, offset_mm=(0.0, 0.0)):
    """Square ``crop_mm`` crop around the image centre shifted by ``offset_mm``
    (x, y), clamped inside the image, resized to IMG."""
    h, w = a.shape
    cpx = min(int(round(crop_mm / max(ps, 0.001))), min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    if offset_mm[0] or offset_mm[1]:
        x0 = int(np.clip(x0 + round(offset_mm[0] / max(ps, 0.001)), 0, w - cpx))
        y0 = int(np.clip(y0 + round(offset_mm[1] / max(ps, 0.001)), 0, h - cpx))
    return cv2.resize(a[y0:y0 + cpx, x0:x0 + cpx], (IMG, IMG), interpolation=cv2.INTER_AREA)


def _pick_series(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane
             and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(study_uid, series_rows, series_root, crop_mm=CROP_MM, offsets_mm=None):
    """Return (volume uint8 [MAXS, IMG, IMG], valid-slice mask, series uid per slot).

    ``offsets_mm``: optional per-slot (x, y) shift of the crop centre in mm."""
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    slot_series = [None] * len(SLOTS)
    idx, used = 0, set()
    for s, (plane, fluid, k) in enumerate(SLOTS):
        start = idx
        r = _pick_series(series_rows, plane, fluid, used)
        files, med_ps = ([], 0.5)
        if r is not None:
            used.add(r["SeriesInstanceUID"])
            slot_series[s] = r["SeriesInstanceUID"]
            files, med_ps = _order_series(f"{series_root}/{study_uid}/{r['SeriesInstanceUID']}")
        if not files:
            idx = start + k
            continue
        n = len(files)
        lo, hi = int(n * SPAN_LO), max(int(n * SPAN_HI) - 1, int(n * SPAN_LO))
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs, pss = [], []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(_read_px(fp))
            except Exception:
                arrs.append(None)
            pss.append(ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if a is not None:
                aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
                off = offsets_mm[s] if offsets_mm is not None else (0.0, 0.0)
                vol[idx] = (_mm_crop_resize(aw, ps if ps > 0 else med_ps, crop_mm, off)
                            * 255).astype(np.uint8)
            idx += 1
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask, slot_series


def eval_centers(mask, k=K_EVAL):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, len(mask)))
    lo, hi = int(valid.min()), int(valid.max())
    cs = list(range(lo + 1, hi))
    if not cs:
        cs = [max(1, min((lo + hi) // 2, len(mask) - 2))]
    sel = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return np.array([cs[i] for i in sel])


def eval_windows(vol, centers):
    """Stack slices c-1, c, c+1 as RGB, resize to RES, ImageNet-normalise."""
    d = vol.shape[0]
    wins = np.empty((len(centers), 3, IMG, IMG), np.float32)
    for j, c in enumerate(centers):
        c = max(1, min(int(c), d - 2))
        wins[j] = vol[c - 1:c + 2].astype(np.float32) / 255.0
    x = F.interpolate(torch.from_numpy(wins), size=(RES, RES), mode="bilinear",
                      align_corners=False)
    return (x - _MEAN) / _STD


def load_series_table(split="train"):
    df = pd.read_csv(DATA_DIR / f"{split}_series.csv", dtype={"StudyInstanceUID": str,
                                                              "SeriesInstanceUID": str})
    return {k: g.to_dict("records") for k, g in df.groupby("StudyInstanceUID")}
