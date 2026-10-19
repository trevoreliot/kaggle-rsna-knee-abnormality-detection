"""Does the pseudo-box YOLO help classification?  Two tests on the YOLO val studies
(held out from YOLO training; raptor itself was trained on all train studies, so
raptor numbers are in-sample and only comparable with each other).

A  detector as classifier: per study and class, max detection confidence over
   all slices -> AUC; and a rank-average blend with raptor's probability.
B  detector as cropper: rebuild each study with raptor's crop re-centred per
   plane slot on the confidence-weighted centre of YOLO detections (and with a
   zoomed 112 mm crop), re-run raptor, compare AUC with the fixed centre crop.

References: gold (manual, 58 studies) and LLM soft labels (>= 0.7 positive,
<= 0.3 negative, rest dropped) on all val studies.

Usage:
    uv run python -m src.explain.yolo_eval \
        --weights assets/processed/yolo_runs/yolo11s_v1/weights/best.pt
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from src.explain import raptor
from src.explain.raptor_gradcam import _auc
from src.explain.yolo_dataset import CLASSES, OUT_DIR as YOLO_DIR, _rgb

SRC = Path("assets/processed/gradcam_raptor/train_fast")
MM_PER_PX = raptor.CROP_MM / raptor.IMG
MAX_SHIFT_MM = 35.0


def val_studies(yolo_dir=YOLO_DIR):
    return sorted({p.stem.rsplit("_", 1)[0] for p in (Path(yolo_dir) / "images" / "val").glob("*.png")})


def detect(model, studies, src=SRC, conf=0.01, batch=128):
    """All detections on every valid slice of each study's fixed-crop volume."""
    rows = []
    for i, uid in enumerate(studies):
        z = np.load(src / "studies" / f"{uid}.npz")
        vol, mask = z["vol"], z["mask"]
        slices = np.flatnonzero(mask)
        imgs = [_rgb(vol, mask, s)[..., ::-1] for s in slices]  # numpy input is BGR
        for b in range(0, len(imgs), batch):
            res = model.predict(imgs[b:b + batch], imgsz=384, conf=conf, half=True, verbose=False)
            for s, r in zip(slices[b:b + batch], res):
                for (cx, cy, w, h), c, k in zip(r.boxes.xywh.tolist(), r.boxes.conf.tolist(),
                                                r.boxes.cls.tolist()):
                    rows.append((uid, int(s), int(raptor.SLOT_OF_SLICE[s]), CLASSES[int(k)],
                                 c, cx, cy, w, h))
        if (i + 1) % 100 == 0:
            print(f"  detect {i + 1}/{len(studies)}")
    return pd.DataFrame(rows, columns=["study", "slice", "slot", "label", "conf",
                                       "cx", "cy", "w", "h"])


def slot_offsets(det, min_conf=0.25):
    """Per study: per-slot (x, y) mm shift towards the weighted detection centre."""
    out = {}
    d = det[det.conf >= min_conf]
    for uid, g in d.groupby("study"):
        offs = [(0.0, 0.0)] * len(raptor.SLOTS)
        for s, gs in g.groupby("slot"):
            wx = np.average(gs.cx, weights=gs.conf) - raptor.IMG / 2
            wy = np.average(gs.cy, weights=gs.conf) - raptor.IMG / 2
            offs[s] = tuple(float(np.clip(v * MM_PER_PX, -MAX_SHIFT_MM, MAX_SHIFT_MM))
                            for v in (wx, wy))
        out[uid] = offs
    return out


def _build(args):
    uid, rows, crop_mm, offs = args
    try:
        vol, mask, _ = raptor.build_study(uid, rows, str(raptor.DATA_DIR / "train_series"),
                                          crop_mm=crop_mm, offsets_mm=offs)
        return uid, vol, mask
    except Exception as e:
        print(f"  skip {uid}: {type(e).__name__}: {e}")
        return uid, None, None


@torch.no_grad()
def raptor_probs(model, vol, mask, device, chunk=16):
    x = raptor.eval_windows(vol, raptor.eval_centers(mask)).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = torch.cat([model.backbone(c) for c in x.split(chunk)]).float()
    return torch.sigmoid(model.head(f[None]))[0].cpu().numpy()


def rerun_raptor(studies, crop_mm, offsets, device, workers=6):
    model = raptor.load_model(device)
    series = raptor.load_series_table("train")
    jobs = [(u, series.get(u, []), crop_mm, offsets.get(u) if offsets else None) for u in studies]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for i, (uid, vol, mask) in enumerate(pool.map(_build, jobs)):
            if vol is not None:
                rows += [(uid, l, p) for l, p in zip(raptor.LABELS, raptor_probs(model, vol, mask, device))]
            if (i + 1) % 100 == 0:
                print(f"  raptor crop={crop_mm:g} {'yolo' if offsets else 'centre'} {i + 1}/{len(studies)}")
    return pd.DataFrame(rows, columns=["study", "label", "prob"])


def auc_table(preds, scores):
    """scores: {name: df(study, label, score)} -> AUC per label / reference / score."""
    ref = preds[["study", "label", "gold", "llm"]]
    out = []
    for name, sc in scores.items():
        m = ref.merge(sc, on=["study", "label"])
        for label, g in m.groupby("label"):
            llm = g[(g.llm >= 0.7) | (g.llm <= 0.3)]
            out.append((name, label, _auc(g.gold.dropna().astype(int), g.loc[g.gold.notna(), "score"]),
                        _auc((llm.llm >= 0.7).astype(int), llm.score), int(g.gold.notna().sum()),
                        int((llm.llm >= 0.7).sum())))
    return pd.DataFrame(out, columns=["score", "label", "auc_gold", "auc_llm", "n_gold", "n_llm_pos"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=None, help="default: <weights run dir>/eval")
    ap.add_argument("--limit", type=int, default=0, help="first N val studies (smoke test)")
    a = ap.parse_args()
    src = Path(a.src)
    out = Path(a.out or Path(a.weights).parents[1] / "eval")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    studies = val_studies()
    if a.limit:
        studies = studies[:a.limit]
    preds = pd.read_csv(src / "preds.csv")
    preds = preds[preds.study.isin(studies)]
    print(f"{len(studies)} val studies ({preds.loc[preds.gold.notna(), 'study'].nunique()} gold)")

    det_path = out / "detections.csv"
    if det_path.exists() and not a.limit:
        det = pd.read_csv(det_path)
    else:
        det = detect(YOLO(a.weights), studies, src)
        det.to_csv(det_path, index=False)

    # A: detector as classifier
    base = preds[["study", "label", "prob"]].rename(columns={"prob": "score"})
    yolo = (det.groupby(["study", "label"]).conf.max().reindex(
        pd.MultiIndex.from_product([studies, CLASSES], names=["study", "label"]), fill_value=0)
        .rename("score").reset_index())
    rk = lambda d: d.assign(score=d.groupby("label").score.rank(pct=True))
    blend = rk(base).merge(rk(yolo), on=["study", "label"], suffixes=("_r", "_y"))
    blend["score"] = (blend.score_r + blend.score_y) / 2
    scores = {"raptor (centre 140)": base, "yolo max conf": yolo,
              "raptor+yolo rank avg": blend[["study", "label", "score"]]}

    # B: detector as cropper
    offs = slot_offsets(det)
    print(f"crop shift (mm, median |x|,|y| over study-slots with detections): "
          f"{np.median([abs(o[0]) for v in offs.values() for o in v if o != (0.0, 0.0)]):.1f}, "
          f"{np.median([abs(o[1]) for v in offs.values() for o in v if o != (0.0, 0.0)]):.1f}")
    for name, crop_mm, use in [("raptor yolo-centred 140", 140.0, True),
                               ("raptor centre 112", 112.0, False),
                               ("raptor yolo-centred 112", 112.0, True)]:
        p = out / f"probs_{name.replace(' ', '_')}.csv"
        if p.exists() and not a.limit:
            df = pd.read_csv(p)
        else:
            df = rerun_raptor(studies, crop_mm, offs if use else None, device)
            df.to_csv(p, index=False)
        scores[name] = df.rename(columns={"prob": "score"})

    t = auc_table(preds, scores)
    t = t[t.label.isin(CLASSES) | t.score.str.startswith("raptor")]
    t.to_csv(out / "auc.csv", index=False)
    cls = t[t.label.isin(CLASSES)]
    summ = cls.groupby("score", sort=False)[["auc_gold", "auc_llm"]].mean()
    summ["auc_llm_all12"] = t.groupby("score", sort=False).auc_llm.mean()
    print("\nmacro AUC over the 10 YOLO classes (all 12 for raptor rows in last column):")
    print(summ.round(3).to_string())
    print("\nper label, LLM-reference AUC:")
    print(cls.pivot(index="label", columns="score", values="auc_llm")
          .reindex(CLASSES)[list(scores)].round(3).to_string())
    (out / "summary.json").write_text(json.dumps(summ.round(4).to_dict(), indent=1))


if __name__ == "__main__":
    main()
