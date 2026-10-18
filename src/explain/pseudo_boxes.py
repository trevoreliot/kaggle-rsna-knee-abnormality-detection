"""Two-level pseudo-boxes from raptor Grad-CAM output.

Input is an output directory of ``src.explain.raptor_gradcam`` that holds a
region map (default ``L3``, compact 12x12 Grad-CAM) and a point map (default
``LC1+L3``, the most faithful but speckled LayerCAM ensemble).

For every positive (study, label):
  0. clean-up: zero the spurious (0, 0) Grad-CAM cell and keep maps on tissue;
  1. windows: the label's top-attention windows whose region map is not weak
     relative to the study-wide peak for that label;
  2. region box: region map upsampled to the 336 px crop, thresholded at a
     fraction of the window max; box of the connected blob with most mass;
  3. point + tight box: the point map, lightly smoothed, peaked inside the
     (dilated) region; tight box = top-mass blob around that peak, with a
     minimum side so small structures still get a usable box;
  4. export: ``BOX_LEVEL`` picks region or tight box per label (or drops the
     label) into the final ``x0..y1`` and YOLO ``cx, cy, w, h`` columns.

Coordinates are in the 336 px / 140 mm centre-crop space the model sees
(``raptor.IMG``; 0.417 mm per px), indexed by the study volume slice.

``--eval`` scores boxes against the model: blur inside all of a label's boxes
and compare the logit drop with same-size boxes at random positions on the
knee in the same windows.

Usage:
    uv run python -m src.explain.pseudo_boxes --src assets/processed/gradcam_raptor/gold_layercam --eval
"""

import argparse
import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms.functional import gaussian_blur

from src.explain import raptor
from src.explain.raptor_gradcam import _CSS, _auc, _logits, _overlay

MM_PER_PX = raptor.CROP_MM / raptor.IMG

# Which box each label exports: "r" region, "t" tight, None = no box.  Chosen on
# the 58 gold studies (gold_layercam, box-vs-random-box deletion win rate):
# large findings keep the region box, small structures the tight box, and
# labels whose best box barely beats random boxes on the knee are left out.
BOX_LEVEL = {
    "ACL": None,               # r 57% / t 52%: boxes land posterior, not in the notch
    "MCL": "t",                # t 89% / r 78%
    "Medial Meniscus": None,   # t 62% / r 46%: often right visually, not reliable yet
    "Lateral Meniscus": "t",   # t 87% / r 70%
    "Medial OA": "t",          # t 87% / r 80%
    "Lateral OA": "r",         # r 82% / t 73%
    "PF OA": "t",              # t 81% / r 48%
    "Effusion": "r",           # r 89% / t 77%
    "Synovitis": "r",          # r 93% / t 85%
    "Baker's": "r",            # r 92% / t 83%
    "Contusion": "t",          # t 84% / r 74%
    "Fracture": "t",           # t 100% / r 100%; tighter wins the tie
}

# Optional per-label restriction of the slots (indices into raptor.SLOTS) boxes
# may come from.  Labels not listed use every slot.
BOX_SLOTS = {}


# --------------------------------------------------------------------------- #
# Box extraction
# --------------------------------------------------------------------------- #
def _up(cam, size=raptor.IMG):
    return cv2.resize(cam.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)


def _best_blob(mask, weight):
    """Connected component of ``mask`` with the largest summed ``weight``."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    mass = np.bincount(lab.ravel(), weights=weight.ravel(), minlength=n)[1:]
    k = int(np.argmax(mass)) + 1
    return lab == k, stats[k]


def _box_of(stats):
    x, y, w, h = stats[:4]
    return int(x), int(y), int(x + w), int(y + h)


def region_box(cam, rel_thr):
    """Region box on one window's upsampled region map."""
    m = cam.max()
    if m <= 0:
        return None
    blob = _best_blob(cam >= rel_thr * m, cam)
    return None if blob is None else (blob[0], _box_of(blob[1]))


def point_box(fine, region_mask, rel_thr, sigma_px, dilate_px, min_side_px):
    """Peak and tight box of the point map inside the dilated region."""
    size = fine.shape[0]
    f = cv2.GaussianBlur(fine, (0, 0), sigma_px)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
    allowed = cv2.dilate(region_mask.astype(np.uint8), k) > 0
    f = np.where(allowed, f, 0)
    if f.max() <= 0:
        return None
    py, px = np.unravel_index(int(np.argmax(f)), f.shape)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        (f >= rel_thr * f.max()).astype(np.uint8), connectivity=8)
    x0, y0, x1, y1 = _box_of(stats[lab[py, px]])
    # Grow to the minimum side around the box centre, clipped to the image.
    for lo, hi in ((0, 2), (1, 3)):
        b = [x0, y0, x1, y1]
        if b[hi] - b[lo] < min_side_px:
            c = (b[lo] + b[hi]) / 2
            b[lo] = int(round(min(max(c - min_side_px / 2, 0), size - min_side_px)))
            b[hi] = b[lo] + min_side_px
        x0, y0, x1, y1 = b
    return (int(px), int(py)), (x0, y0, x1, y1)


def suppress_corner(cams):
    """Zero the top-left 1/12 x 1/12 of every map (windows, h, w).

    CoaTNet Grad-CAM puts a spurious hot cell at (0, 0) of the 12x12 grid: 8% of
    top-window peaks on the gold studies land exactly there (0.7% expected by
    chance), whatever the anatomy.  Likely a padding artefact.
    """
    cams = cams.copy()
    k = max(1, cams.shape[-1] // 12)
    cams[..., :k, :k] = 0
    return cams


def tissue_mask(slice_u8, dilate_mm=5.0):
    """Foreground of a crop slice, dilated so evidence at tissue edges survives."""
    blur = cv2.GaussianBlur(slice_u8, (0, 0), 3)
    fg = (blur > max(10, np.percentile(blur, 30))).astype(np.uint8)
    r = int(round(dilate_mm / MM_PER_PX))
    return cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0


def boxes_for_label(z, n, region_map, point_map, top_windows=3, min_rel_peak=0.35,
                    region_thr=0.5, point_thr=0.3, sigma_mm=3.0, dilate_mm=4.0,
                    min_side_mm=10.0, slots=None):
    """Pseudo-box rows for label index ``n`` of one study npz.

    ``slots``: optional slot indices (into ``raptor.SLOTS``) the boxes may come
    from; windows are then ranked, and the weak-window check normalised, within
    those slots only.
    """
    reg = suppress_corner(z[f"cams_{region_map}"][n].astype(np.float32))
    pnt = suppress_corner(z[f"cams_{point_map}"][n].astype(np.float32))
    attn, centers, vol = z["attn"][:, n].copy(), z["centers"], z["vol"]
    if slots is not None:
        allowed = np.isin(raptor.SLOT_OF_SLICE[centers], list(slots))
        if not allowed.any():
            return []
        attn[~allowed] = -np.inf
        study_peak = reg[allowed].max()
    else:
        study_peak = reg.max()
    if study_peak <= 0:
        return []
    rows = []
    for rank, j in enumerate(np.argsort(-attn)[:top_windows]):
        if not np.isfinite(attn[j]):
            break
        if reg[j].max() < min_rel_peak * study_peak:
            continue
        tissue = tissue_mask(vol[int(centers[j])])
        r = region_box(_up(reg[j]) * tissue, region_thr)
        if r is None:
            continue
        rmask, rbox = r
        p = point_box(_up(pnt[j]) * tissue, rmask, point_thr, sigma_mm / MM_PER_PX,
                      int(round(dilate_mm / MM_PER_PX)), int(round(min_side_mm / MM_PER_PX)))
        (px, py), tbox = p if p is not None else ((None, None), rbox)
        c = int(centers[j])
        rows.append({
            "window": int(j), "slice": c, "slot": raptor.SLOT_NAMES[raptor.SLOT_OF_SLICE[c]],
            "attn": float(attn[j]), "attn_rank": rank,
            "peak_rel": float(reg[j].max() / study_peak),
            **{f"r_{k}": v for k, v in zip(("x0", "y0", "x1", "y1"), rbox)},
            **{f"t_{k}": v for k, v in zip(("x0", "y0", "x1", "y1"), tbox)},
            "px": px, "py": py,
        })
    return rows


def positives(preds, source, llm_thr):
    """(study, label) pairs to box: gold labels where present, else LLM soft label."""
    gold_pos = preds.gold == 1
    llm_pos = preds.gold.isna() & (preds.llm >= llm_thr)
    if source == "gold":
        keep = gold_pos
    elif source == "llm":
        keep = preds.llm >= llm_thr
    else:  # gold_or_llm
        keep = gold_pos | llm_pos
    return preds[keep]


def generate(src, region_map="L3", point_map="LC1+L3", source="gold_or_llm",
             llm_thr=0.7, **kw):
    src = Path(src)
    preds = pd.read_csv(src / "preds.csv", dtype={"study": str})
    rows = []
    for study, grp in positives(preds, source, llm_thr).groupby("study", sort=False):
        z = np.load(src / "studies" / f"{study}.npz", allow_pickle=False)
        for _, p in grp.iterrows():
            n = raptor.LABELS.index(p.label)
            for r in boxes_for_label(z, n, region_map, point_map,
                                     slots=BOX_SLOTS.get(p.label), **kw):
                rows.append({"study": study, "label": p.label, "prob": p.prob,
                             "gold": p.gold, "llm": p.llm, **r})
    df = pd.DataFrame(rows)
    if len(df):
        df["level"] = df.label.map(BOX_LEVEL)
        df = df[df.level.notna()].copy()
        for k in ("x0", "y0", "x1", "y1"):
            df[k] = np.where(df.level == "r", df[f"r_{k}"], df[f"t_{k}"])
        # YOLO-normalised centre / size on the 336 px crop.
        df["cx"] = (df.x0 + df.x1) / 2 / raptor.IMG
        df["cy"] = (df.y0 + df.y1) / 2 / raptor.IMG
        df["w"] = (df.x1 - df.x0) / raptor.IMG
        df["h"] = (df.y1 - df.y0) / raptor.IMG
        for b in ("r", "t"):
            w = df[f"{b}_x1"] - df[f"{b}_x0"]
            h = df[f"{b}_y1"] - df[f"{b}_y0"]
            df[f"{b}_w_mm"], df[f"{b}_h_mm"] = w * MM_PER_PX, h * MM_PER_PX
            df[f"{b}_area_frac"] = w * h / raptor.IMG ** 2
    return df


# --------------------------------------------------------------------------- #
# Box faithfulness
# --------------------------------------------------------------------------- #
def _random_box(fg, w, h, rng, tries=50):
    size = fg.shape[0]
    for _ in range(tries):
        x0 = int(rng.integers(0, max(1, size - w)))
        y0 = int(rng.integers(0, max(1, size - h)))
        if fg[y0 + h // 2, x0 + w // 2]:
            return x0, y0, x0 + w, y0 + h
    return x0, y0, x0 + w, y0 + h


def evaluate(src, boxes, n_random=3, seed=0):
    """Per (study, label): logit drop for blurring own boxes vs random boxes."""
    src = Path(src)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = raptor.load_model(device)
    scale = raptor.RES / raptor.IMG
    rng = np.random.default_rng(seed)
    out = []
    for study, sb in boxes.groupby("study", sort=False):
        z = np.load(src / "studies" / f"{study}.npz", allow_pickle=False)
        vol = z["vol"]
        x = raptor.eval_windows(vol, z["centers"]).to(device)
        x_blur = gaussian_blur(x, kernel_size=61, sigma=15.0)
        base = _logits(model, x)

        def drop(n, boxes_by_window):
            m = torch.zeros(x.shape[0], 1, *x.shape[-2:], dtype=torch.bool, device=device)
            for j, (x0, y0, x1, y1) in boxes_by_window:
                m[j, :, int(y0 * scale):int(y1 * scale), int(x0 * scale):int(x1 * scale)] = True
            return (base[n] - _logits(model, torch.where(m, x_blur, x))[n]).item()

        for label, lb in sb.groupby("label", sort=False):
            n = raptor.LABELS.index(label)
            row = {"study": study, "label": label, "n_boxes": len(lb)}
            for b in ("r", "t"):
                own = [(int(j), tuple(int(v) for v in c)) for j, c in zip(
                    lb.window, lb[[f"{b}_x0", f"{b}_y0", f"{b}_x1", f"{b}_y1"]].to_numpy())]
                rand = []
                for _ in range(n_random):
                    rb = []
                    for j, (x0, y0, x1, y1) in own:
                        fg = tissue_mask(vol[int(z["centers"][j])], dilate_mm=0)
                        rb.append((j, _random_box(fg, x1 - x0, y1 - y0, rng)))
                    rand.append(drop(n, rb))
                row[f"{b}_drop"] = drop(n, own)
                row[f"{b}_drop_rand"] = float(np.mean(rand))
            out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _draw(slice_u8, region_cam, row, size=256):
    s = size / raptor.IMG
    gray = cv2.cvtColor(cv2.resize(slice_u8, (size, size), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_GRAY2BGR)
    img = _overlay(gray, region_cam, size)
    cv2.rectangle(img, (int(row.r_x0 * s), int(row.r_y0 * s)),
                  (int(row.r_x1 * s), int(row.r_y1 * s)), (60, 60, 255), 2)
    cv2.rectangle(img, (int(row.t_x0 * s), int(row.t_y0 * s)),
                  (int(row.t_x1 * s), int(row.t_y1 * s)), (80, 255, 80), 2)
    if not pd.isna(row.px):
        cv2.drawMarker(img, (int(row.px * s), int(row.py * s)), (255, 255, 255),
                       cv2.MARKER_CROSS, 12, 2)
    pair = np.concatenate([gray, np.full((size, 4, 3), 255, np.uint8), img], axis=1)
    ok, buf = cv2.imencode(".jpg", pair, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode()


def build_report(src, boxes, scores, region_map, out_path, per_label=8):
    src = Path(src)
    parts = [f"<title>Pseudo-boxes</title><style>{_CSS}</style>",
             "<h1>Two-level pseudo-boxes</h1>",
             f"<p class=sub>{boxes.study.nunique()} studies &middot; {len(boxes)} boxes &middot; "
             "<span style='color:#ff3c3c'>red</span>: region box (L3 Grad-CAM) &middot; "
             "<span style='color:#3cb43c'>green</span>: tight box &middot; cross: point peak "
             "&middot; overlay: region map</p>"]
    g = boxes.groupby("label")
    summ = pd.DataFrame({
        "boxes": g.size(), "pairs": g.apply(lambda d: d[["study"]].drop_duplicates().shape[0]),
        "r_w_mm": g.r_w_mm.median(), "r_h_mm": g.r_h_mm.median(),
        "t_w_mm": g.t_w_mm.median(), "t_h_mm": g.t_h_mm.median(),
    })
    if scores is not None and len(scores):
        sg = scores.groupby("label")
        summ["r_win"] = sg.apply(lambda d: (d.r_drop > d.r_drop_rand).mean())
        summ["t_win"] = sg.apply(lambda d: (d.t_drop > d.t_drop_rand).mean())
        summ["r_drop"], summ["r_rand"] = sg.r_drop.median(), sg.r_drop_rand.median()
        summ["t_drop"], summ["t_rand"] = sg.t_drop.median(), sg.t_drop_rand.median()
    summ = summ.reindex([l for l in raptor.LABELS if l in summ.index])
    cols = list(summ.columns)
    titles = {"boxes": "boxes", "pairs": "positives", "r_w_mm": "region w mm",
              "r_h_mm": "region h mm", "t_w_mm": "tight w mm", "t_h_mm": "tight h mm",
              "r_win": "region &gt; rand", "t_win": "tight &gt; rand", "r_drop": "region drop",
              "r_rand": "rand drop", "t_drop": "tight drop", "t_rand": "rand drop (t)"}

    def fmt(c, v):
        if c.endswith("win"):
            return f"{v * 100:.0f}%"
        return f"{v:.0f}" if c in ("boxes", "pairs") or c.endswith("mm") else f"{v:.2f}"

    parts.append("<h2>Summary</h2><div class=scroll><table><tr><th>Label</th>"
                 + "".join(f"<th>{titles[c]}</th>" for c in cols) + "</tr>"
                 + "".join(f"<tr><td>{html.escape(l)}</td>"
                           + "".join(f"<td>{fmt(c, r[c])}</td>" for c in cols) + "</tr>"
                           for l, r in summ.iterrows())
                 + "</table></div><p class=note>&gt; rand: share of positives where blurring "
                 "the boxes lowers the label logit more than blurring same-size boxes placed "
                 "at random on the knee in the same slices.</p>")

    for label in summ.index:
        lb = boxes[(boxes.label == label) & (boxes.attn_rank == 0)]
        lb = lb.sort_values("prob", ascending=False).head(per_label)
        parts.append(f"<h2>{html.escape(label)}</h2><div class=row style='display:flex;"
                     "flex-wrap:wrap;gap:10px'>")
        for r in lb.itertuples():
            z = np.load(src / "studies" / f"{r.study}.npz", allow_pickle=False)
            n = raptor.LABELS.index(label)
            reg = suppress_corner(z[f"cams_{region_map}"][n].astype(np.float32))
            parts.append(
                f"<figure><img src='data:image/jpeg;base64,"
                f"{_draw(z['vol'][r.slice], reg[r.window] / (reg.max() + 1e-8), r)}'>"
                f"<figcaption>{r.study[-10:]} &middot; {r.slot} &middot; prob {r.prob:.2f} "
                f"&middot; tight {r.t_w_mm:.0f}&times;{r.t_h_mm:.0f} mm</figcaption></figure>")
        parts.append("</div>")
    Path(out_path).write_text("\n".join(parts))
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="raptor_gradcam output directory")
    ap.add_argument("--region-map", default="L3")
    ap.add_argument("--point-map", default="LC1+L3")
    ap.add_argument("--positives", default="gold_or_llm", choices=["gold", "llm", "gold_or_llm"])
    ap.add_argument("--llm-thr", type=float, default=0.7)
    ap.add_argument("--top-windows", type=int, default=3)
    ap.add_argument("--region-thr", type=float, default=0.5)
    ap.add_argument("--point-thr", type=float, default=0.3)
    ap.add_argument("--min-side-mm", type=float, default=10.0)
    ap.add_argument("--eval", action="store_true", help="score boxes against the model")
    ap.add_argument("--all-labels", action="store_true",
                    help="ignore BOX_LEVEL exclusions (for re-tuning the label table)")
    ap.add_argument("--out", help="output dir (default: <src>/boxes)")
    a = ap.parse_args()

    src = Path(a.src)
    meta = json.loads((src / "meta.json").read_text())
    for m in (a.region_map, a.point_map):
        if m not in meta["maps"]:
            ap.error(f"map {m!r} not in {src} (has {meta['maps']})")
    out = Path(a.out or src / "boxes")
    out.mkdir(parents=True, exist_ok=True)
    if a.all_labels:
        BOX_LEVEL.update({k: v or "t" for k, v in BOX_LEVEL.items()})
    boxes = generate(src, a.region_map, a.point_map, a.positives, a.llm_thr,
                     top_windows=a.top_windows, region_thr=a.region_thr,
                     point_thr=a.point_thr, min_side_mm=a.min_side_mm)
    boxes.to_csv(out / "pseudo_boxes.csv", index=False)
    print(f"{len(boxes)} boxes for {boxes[['study', 'label']].drop_duplicates().shape[0]} "
          f"positives -> {out / 'pseudo_boxes.csv'}", flush=True)
    scores = None
    if a.eval:
        scores = evaluate(src, boxes)
        scores.to_csv(out / "box_scores.csv", index=False)
    print(build_report(src, boxes, scores, a.region_map, out / "boxes.html"))


if __name__ == "__main__":
    main()
