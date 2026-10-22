"""Kaggle inference: ensemble of our 2.5D attention-MIL models (src/models/mil25d.py).

Self-contained (no ``src`` import) so it runs offline in a Kaggle notebook.
Preprocessing is a copy of ``src.explain.raptor.build_study`` (the study cache
the models were trained on): 64 slices from five plane/sequence slots, 140 mm
centre crop, 2-98% windowing per series, 336 px uint8; 62 three-slice windows
resized to the model's ``img`` and ImageNet-normalised.

Members: ``<group>__fold<k>.pt`` (fp16 state dicts, written by ``--export``)
next to ``config.json`` ({group: {backbone, img, head, weight}}), flat so the
Kaggle dataset needs no sub-folders.  Optionally blended (``RAPTOR_WEIGHT``) with the public
raptor CoaTNet (dreaddevelopment/raptor-knee-maxspan, used as published).  Each group's folds are averaged, then the groups are averaged with
their weights.  A study that fails to load keeps 0.5.

Credits -- this builds directly on public work; please upvote the originals:
  Training labels: the models are trained on the mean of four public LLM report-label tables
    - stevenleehans   datasets/stevenleehans/rsna-knee-llm-report-labels (llm_labels_v4_blend)
    - pilkwang        datasets/pilkwang/rsna-knee-llm-labels (report_labels_v2)
    - lixin73         datasets/lixin73/rsna-knee-llm-report-labels-sol56 (labels_llm_gpt56sol)
    - laymond         datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels
  Label-averaging recipe and report-grouped folds:
    - goodpjw2008     datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader (make_labels.py)
  Raptor CoaTNet weights, blended in as published (CC0):
    - Dread Development  datasets/dreaddevelopment/raptor-knee-maxspan
  Model design and preprocessing (64-slice / 5-slot sampling, 140 mm crop, 3-slice windows,
  per-label attention pooling), re-implemented and trained from ImageNet weights:
    - Dread Development (dreaddevelopment)
                      code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model
                      datasets/dreaddevelopment/raptor-knee-maxspan
    - prvsiyan        code/prvsiyan/head-and-shoulders-knees-and-toes (where we studied it)
  Analysis that pointed us to label blending:
    - rabari9999      code/rabari9999/rsna-knee-lb-0-95
  Backbones: ConvNeXt (Meta AI) ImageNet weights via timm (Ross Wightman).

Kaggle:  attach the competition and the model dataset, run this file.
Local:   uv run python notebooks/kaggle/mil25d_ens/submit_mil25d.py \
             --export assets/processed/kaggle_upload/mil25d-ens        (dataset dir)
         uv run python notebooks/kaggle/mil25d_ens/submit_mil25d.py \
             --data assets/big-data/rsna-knee-abnormality-detection \
             --models assets/processed/kaggle_upload/mil25d-ens --out sub.csv
"""

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pydicom.pixels import apply_modality_lut
except ImportError:  # pydicom < 3
    from pydicom.pixel_data_handlers.util import apply_modality_lut

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
IMG = 336
CROP_MM = 140.0
SPAN_LO, SPAN_HI = 0.02, 0.98
K_EVAL = 62
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12), ("Coronal", 0, 8),
         ("Axial", -1, 12)]
MAXS = sum(k for _, _, k in SLOTS)
# Blend: our groups (config.json, sharing 1 - RAPTOR_WEIGHT) + the public raptor CoaTNet
# (dreaddevelopment/raptor-knee-maxspan, attach it as a dataset).  RAPTOR_WEIGHT = 0 -> ours only.
RAPTOR = {"file": "raptor_ft_coatnet_v5_full_swa.pt", "res": 384}
RAPTOR_WEIGHT = 0.5
BLEND = "rank"  # "rank": weighted mean of per-label rank percentiles; "prob": of probabilities
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- #
# DICOM -> study volume (copy of src.explain.raptor)
# --------------------------------------------------------------------------- #
def _order_series(series_dir):
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


def _crop_resize(a, ps):
    h, w = a.shape
    cpx = min(int(round(CROP_MM / max(ps, 0.001))), min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    return cv2.resize(a[y0:y0 + cpx, x0:x0 + cpx], (IMG, IMG), interpolation=cv2.INTER_AREA)


def _pick_series(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(uid, rows, series_root):
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    idx, used = 0, set()
    for plane, fluid, k in SLOTS:
        start = idx
        r = _pick_series(rows, plane, fluid, used)
        files, med_ps = [], 0.5
        if r is not None:
            used.add(r["SeriesInstanceUID"])
            files, med_ps = _order_series(f"{series_root}/{uid}/{r['SeriesInstanceUID']}")
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
        loq, hiq = (np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
                    if valid else (0.0, 1.0))
        for a, ps in zip(arrs, pss):
            if a is not None:
                aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
                vol[idx] = (_crop_resize(aw, ps if ps > 0 else med_ps) * 255).astype(np.uint8)
            idx += 1
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


def eval_centers(mask, k=K_EVAL):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, len(mask)))
    lo, hi = int(valid.min()), int(valid.max())
    cs = list(range(lo + 1, hi)) or [max(1, min((lo + hi) // 2, len(mask) - 2))]
    return np.array([cs[i] for i in np.linspace(0, len(cs) - 1, k).round().astype(int)])


def _load(job):
    uid, rows, root = job
    try:
        vol, mask = build_study(uid, rows, root)
        idx = np.clip(eval_centers(mask)[:, None] + np.arange(-1, 2), 0, MAXS - 1)
        return uid, vol[idx], None  # (62, 3, 336, 336) uint8
    except Exception as e:
        return uid, None, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# Models (same modules as src.explain.raptor.RaptorClassifier / mil25d)
# --------------------------------------------------------------------------- #
class MILClassifier(nn.Module):
    def __init__(self, backbone, f_dim, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(f_dim)
        self.att = nn.Sequential(nn.Linear(f_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, f_dim))
        self.clsb = nn.Parameter(torch.zeros(n))

    def pool(self, feats):
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        return torch.einsum("bkn,bkf->bnf", a, h)

    def head(self, feats):
        return (self.pool(feats) * self.clsW).sum(-1) + self.clsb


class LabelGraphClassifier(MILClassifier):
    def __init__(self, backbone, f_dim, n=12, drop=0.2, hidden=256):
        super().__init__(backbone, f_dim, n, drop)
        self.register_buffer("adj", torch.eye(n))
        self.g1 = nn.Linear(f_dim, hidden)
        self.g2 = nn.Linear(hidden, f_dim)

    def head(self, feats):
        pooled = self.pool(feats)
        pooled = pooled + self.adj @ self.g2(F.gelu(self.adj @ self.g1(pooled)))
        return (pooled * self.clsW).sum(-1) + self.clsb


def load_member(path, cfg, device):
    bb = timm.create_model(cfg["backbone"], pretrained=False, num_classes=0, global_pool="avg")
    cls = LabelGraphClassifier if cfg.get("head", "mil") == "graph" else MILClassifier
    m = cls(bb, bb.num_features)
    m.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cpu").items()})
    return m.eval().to(device).to(memory_format=torch.channels_last)


def find_file(name, models_dir):
    kaggle = [d / name for d in search_input(lambda f: name in f)] if os.path.exists("/kaggle/input") else []
    for c in [models_dir / name, *kaggle, Path("assets/big-data/public_weights/raptor-knee-maxspan") / name]:
        if Path(c).exists():
            return Path(c)
    raise RuntimeError(f"{name} not found")


def load_raptor(models_dir, device):
    ck = torch.load(find_file(RAPTOR["file"], models_dir), map_location="cpu", weights_only=False)
    assert list(ck["lab"]) == LABELS, ck["lab"]
    bb = timm.create_model(ck["arch"], pretrained=False, num_classes=0, global_pool="avg")
    m = MILClassifier(bb, bb.num_features)
    m.load_state_dict(ck["model"])
    return m.eval().to(device).to(memory_format=torch.channels_last)


@torch.no_grad()
def predict(model, x_u8, img, chunk=32):
    x = x_u8.float().div_(255)
    if x.shape[-1] != img:
        x = F.interpolate(x, size=(img, img), mode="bilinear", align_corners=False)
    x = ((x - _MEAN.to(x.device)) / _STD.to(x.device)).contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
        f = torch.cat([model.backbone(c) for c in x.split(chunk)])
    return torch.sigmoid(model.head(f.float()[None]))[0].cpu().numpy()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def search_input(pred, root="/kaggle/input"):
    """Directories under /kaggle/input (DICOM trees pruned) whose file names satisfy ``pred``."""
    hits = []
    for d, sub, files in os.walk(root):
        sub[:] = [x for x in sub if not x.endswith(("_series", "_images"))]
        if pred(files):
            hits.append(Path(d))
    return hits


def find_competition():
    for c in ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
              "/kaggle/input/rsna-knee-abnormality-detection"]:
        if os.path.exists(f"{c}/test.csv"):
            return Path(c)
    hits = search_input(lambda f: "test.csv" in f and "sample_submission.csv" in f)
    if hits:
        return hits[0]
    raise RuntimeError("competition data not found under /kaggle/input")


def find_models():
    hits = search_input(lambda f: "config.json" in f and any("__fold" in x for x in f))
    if not hits:
        raise RuntimeError("model config.json not found under /kaggle/input")
    return hits[0]


def run(data, models_dir, out, split="test", ids=None, workers=None):
    t0 = time.time()
    print(f"torch {torch.__version__} | timm {timm.__version__} | pydicom {pydicom.__version__} | "
          f"cpus {os.cpu_count()}", flush=True)
    # Fork the DICOM workers before CUDA is initialised in this process.
    pool = Pool(workers or os.cpu_count())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((models_dir / "config.json").read_text())
    own = sum(c["weight"] for c in cfg.values())
    for c in cfg.values():  # our groups share 1 - RAPTOR_WEIGHT
        c["weight"] *= (1 - RAPTOR_WEIGHT) / own
    members = []  # (group, img, model)
    for g, c in cfg.items():
        for p in sorted(models_dir.glob(f"{g}__fold*.pt")):
            members.append((g, c["img"], load_member(p, c, dev)))
    if RAPTOR_WEIGHT > 0:
        cfg["raptor"] = {"img": RAPTOR["res"], "weight": RAPTOR_WEIGHT}
        print(f"loaded ours | {time.time() - t0:.0f}s", flush=True)
        members.append(("raptor", RAPTOR["res"], load_raptor(models_dir, dev)))
    groups = list(cfg)
    print(f"device {dev} | {len(members)} members "
          f"{ {g: sum(m[0] == g for m in members) for g in groups} } | weights "
          f"{ {g: round(cfg[g]['weight'], 3) for g in groups} } | blend {BLEND} | "
          f"{time.time() - t0:.0f}s", flush=True)

    ser = pd.read_csv(data / f"{split}_series.csv", dtype=str)
    series = {k: g.to_dict("records") for k, g in ser.groupby("StudyInstanceUID")}
    if ids is None:
        ids = pd.read_csv(data / f"{split}.csv", dtype={"StudyInstanceUID": str}).StudyInstanceUID.tolist()
    root = str(data / f"{split}_series")
    gp = np.full((len(groups), len(ids), len(LABELS)), 0.5, np.float32)  # per-group probabilities
    ok = np.zeros(len(ids), bool)
    jobs = [(u, series.get(u, []), root) for u in ids]
    with pool:
        for i, (uid, x, err) in enumerate(pool.imap(_load, jobs, chunksize=1)):
            if x is None:
                print(f"  FALLBACK 0.5 {uid}: {err}", flush=True)
            else:
                x = torch.from_numpy(x).to(dev, non_blocking=True)
                for j, g in enumerate(groups):
                    gp[j, i] = np.mean([predict(m, x, img) for gg, img, m in members if gg == g], 0)
                ok[i] = True
            if (i + 1) % 50 == 0 or i + 1 == len(ids):
                el = time.time() - t0
                print(f"  {i + 1}/{len(ids)} | {el:.0f}s | {el / (i + 1):.2f} s/study", flush=True)

    w = np.array([cfg[g]["weight"] for g in groups])[:, None, None]
    if BLEND == "rank" and ok.sum() > 1:  # per-label rank percentiles over the studies that loaded
        r = np.full_like(gp, 0.5)
        r[:, ok] = gp[:, ok].argsort(1).argsort(1) / (ok.sum() - 1)
        final = (w * r).sum(0) / w.sum()
    else:
        final = (w * gp).sum(0) / w.sum()
    sub = pd.DataFrame(final.astype(np.float32), columns=LABELS)
    sub.insert(0, "StudyInstanceUID", ids)
    sample = data / "sample_submission.csv"
    if split == "test" and sample.exists():
        sub = sub[list(pd.read_csv(sample, nrows=0).columns)]
    assert np.isfinite(sub[LABELS].to_numpy()).all()
    sub.to_csv(out, index=False)
    pd.concat({g: pd.DataFrame(gp[j], index=ids, columns=LABELS) for j, g in enumerate(groups)},
              names=["group", "StudyInstanceUID"]).to_csv(Path(out).with_name("group_probs.csv"))
    print(f"wrote {out}: {sub.shape}, fallbacks {(~ok).sum()} | {time.time() - t0:.0f}s", flush=True)
    return sub


# --------------------------------------------------------------------------- #
# Export (local): fp16 member weights + config for the Kaggle dataset
# --------------------------------------------------------------------------- #
ENSEMBLE = {  # group -> (run dir, weight)
    "convnext_tiny": ("assets/processed/mil_runs/convnext_tiny_cons4_5fold", 0.5),
    "convnext_small_ema": ("assets/processed/mil_runs/convnext_small_ema_5fold", 0.5),
}


def export_models(dest):
    dest = Path(dest)
    cfg = {}
    for g, (run_dir, w) in ENSEMBLE.items():
        dest.mkdir(parents=True, exist_ok=True)
        for p in sorted(Path(run_dir).glob("fold*/best.pt")):
            ck = torch.load(p, map_location="cpu", weights_only=True)
            a = ck["args"]
            cfg[g] = {"backbone": a["backbone"], "img": a["img"], "head": a.get("head", "mil"),
                      "weight": w}
            sd = {k: v.half() if v.is_floating_point() else v for k, v in ck["model"].items()}
            torch.save(sd, dest / f"{g}__{p.parent.name}.pt")
    (dest / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg, indent=1))


def main():
    on_kaggle = os.path.exists("/kaggle/input")
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=None, help="write member weights to this models dir and exit")
    ap.add_argument("--data", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", default=None, help="csv with StudyInstanceUID column (local checks)")
    ap.add_argument("--out", default="/kaggle/working/submission.csv" if on_kaggle else "submission.csv")
    ap.add_argument("--workers", type=int, default=None)
    a, _ = ap.parse_known_args()
    if a.export:
        return export_models(a.export)
    t0 = time.time()
    data = Path(a.data) if a.data else find_competition()
    models_dir = Path(a.models) if a.models else find_models()
    print(f"data {data} | models {models_dir} | {time.time() - t0:.0f}s", flush=True)
    ids = pd.read_csv(a.ids, dtype=str).StudyInstanceUID.tolist() if a.ids else None
    run(data, models_dir, a.out, a.split, ids, a.workers)


if __name__ == "__main__":
    main()
