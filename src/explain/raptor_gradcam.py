"""Grad-CAM on the public raptor CoaTNet model (or our own ``mil25d`` checkpoint),
at several depths plus an ensemble.

For every study: per-label probabilities, per-label slice attention, and a
Grad-CAM map for every (label, window) at each requested CoaTNet stage
(1: 48x48, 2: 24x24, 3: 12x12).  Because the model pools windows with per-label
attention, the gradient of a label's logit already carries that attention, so
maps are comparable across windows of a study.

Maps are Grad-CAM (``L<k>``) or LayerCAM (``LC<k>``).  An ensemble such as
``L1+L2`` is the geometric mean of the per-study-normalised maps, upsampled to
the finest grid: deep layers say *what*,
shallow layers say *exactly where*, and the product keeps only agreement.

Each map is scored two ways:
  * deletion: blur the top ``--frac`` of map pixels and measure the logit drop,
    versus a control that blurs the same amount of the same windows with the
    map cells spatially shuffled.  Own > shuffled means the map is faithful.
  * focality: share of total map mass inside its top ``--frac`` pixels
    (``--frac`` would be a perfectly flat map; 1.0 a single hot spot).
  * compactness ("blob"): share of the top-attention window's top pixels in its
    largest connected component.  Faithful-but-speckled maps score high on
    deletion and low here, and cannot become pseudo-boxes.

Usage:
    uv run python -m src.explain.raptor_gradcam --studies gold --maps L3 L2 L1 L1+L2
    uv run python -m src.explain.raptor_gradcam --maps L3 LC1 LC1+L3 \\
        --reuse assets/processed/gradcam_raptor/gold_multi --out DIR
    uv run python -m src.explain.raptor_gradcam --out DIR --gallery-only
    uv run python -m src.explain.raptor_gradcam --studies all --fast \\
        --out assets/processed/gradcam_raptor/train_fast      # bulk, resumable
    uv run python -m src.explain.raptor_gradcam --studies gold --maps L2 L1 LC1 \\
        --model assets/processed/mil_runs/convnext_tiny_cons4_5fold/fold0/best.pt \\
        --reuse assets/processed/gradcam_raptor/gold_multi --out DIR   # own model

``--model``: ``raptor`` or a ``mil25d`` checkpoint.  Stage grids depend on the
backbone and input size: raptor CoaTNet at 384 px has stages 1-3 = 48/24/12;
ConvNeXt-T at 256 px has stages 0-3 = 64/32/16/8.
"""

import argparse
import base64
import html
import json
import re
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from src.explain import raptor
from src.models.registry import register_model

LLM_LABELS = (raptor.DATA_DIR / "custom_labels/slh/kneelabels-stevenleehans"
              / "llm_labels_full.csv")
OUT_DIR = Path("assets/processed/gradcam_raptor")
DEFAULT_MAPS = ["L3", "L2", "L1", "L1+L2"]
FAST_MAPS = ["L3", "LC1+L3"]  # what src.explain.pseudo_boxes uses


# --------------------------------------------------------------------------- #
# Grad-CAM engine
# --------------------------------------------------------------------------- #
def _amp():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available())


def _no_grad_trunk(model, x, stop, chunk=16):
    """Backbone stem + stages[:stop] without grad, in chunks to bound VRAM."""
    bb, outs = model.backbone, []
    with torch.no_grad(), _amp():
        for xc in x.split(chunk):
            h = bb.stem(xc)
            for stage in bb.stages[:stop]:
                h = stage(h)
            outs.append(h)
    return torch.cat(outs)


def _tail(model, h, start):
    """stages[start:] + norm + pool: stage activations -> (K, F) features."""
    bb = model.backbone
    for stage in bb.stages[start:]:
        h = stage(h)
    return _pool(bb, h)


def _pool(bb, h):
    """Final norm + global pool + head norm (CoaTNet: ``norm``; ConvNeXt: ``norm_pre``)."""
    return bb.head(bb.norm(h) if hasattr(bb, "norm") else bb.norm_pre(h))


def parse_maps(specs):
    """Map specs -> (base maps needed, CoaTNet stages needed).

    Base maps: ``L<k>`` Grad-CAM or ``LC<k>`` LayerCAM at stage k.
    Ensembles: base maps joined by ``+`` (e.g. ``LC1+L3``).
    """
    bases = []
    for spec in specs:
        for b in spec.split("+"):
            if not re.fullmatch(r"LC?[0-3]", b):
                raise ValueError(f"bad map spec {b!r} in {spec!r}")
            if b not in bases:
                bases.append(b)
    return bases, sorted({int(b[-1]) for b in bases}, reverse=True)


def gradcam(model, x, layers, chunk=8):
    """Grad-CAM and LayerCAM at each CoaTNet stage in ``layers``.

    Grad-CAM weights each channel by its spatially averaged gradient.  LayerCAM
    weights every activation by its own positive gradient, which keeps detail at
    shallow layers where averaged gradients wash out.

    The head couples windows only through attention pooling, and each window's
    features depend on that window alone, so d logit / d A_k factors exactly as
    (d logit / d feat_k) . (d feat_k / d A_k).  We get the first factor once from
    the head, then backprop chunks of windows through the backbone.  That keeps
    stage-1 (48x48) Grad-CAM inside a few GB.

    Returns probs (12,), attention (K, 12), {"L<k>"/"LC<k>": CAMs (12, K, h, w)}.
    """
    lo = min(layers)
    n_lab = len(raptor.LABELS)
    a_lo = _no_grad_trunk(model, x, lo + 1)
    with torch.no_grad(), _amp():
        feats = torch.cat([_tail(model, a, lo + 1) for a in a_lo.split(chunk)]).float()

    f = feats.requires_grad_(True)
    logits = model.head(f[None])[0]
    attn = model.attention(f[None])[0]
    g_feat = torch.stack([torch.autograd.grad(logits[n], f, retain_graph=True)[0]
                          for n in range(n_lab)])  # (12, K, F)

    names = [f"{p}{l}" for l in layers for p in ("L", "LC")]
    cams = {k: [] for k in names}
    bb = model.backbone
    for s in range(0, len(x), chunk):
        a = a_lo[s:s + chunk].float().requires_grad_(True)
        acts = {lo: a}
        with _amp():
            h = a
            for i in range(lo + 1, len(bb.stages)):
                h = bb.stages[i](h)
                acts[i] = h
            fc = _pool(bb, h).float()
        targets = [acts[l] for l in layers]
        per_label = []
        for n in range(n_lab):
            grads = torch.autograd.grad(fc, targets, grad_outputs=g_feat[n, s:s + chunk],
                                        retain_graph=n < n_lab - 1)
            maps = []
            for g, t in zip(grads, targets):
                g, t = g.float(), t.float()
                maps.append(F.relu((g.mean(dim=(2, 3), keepdim=True) * t).sum(1)).detach())
                maps.append(F.relu((F.relu(g) * t).sum(1)).detach())
            per_label.append(maps)
        for i, k in enumerate(names):
            cams[k].append(torch.stack([pl[i] for pl in per_label]))  # (12, k, h, w)
    cams = {k: torch.cat(v, dim=1).cpu().numpy() for k, v in cams.items()}
    return (torch.sigmoid(logits).detach().cpu().numpy(),
            attn.detach().cpu().numpy(), cams)


def ensemble(cams, names):
    """Geometric mean of per-label, per-study max-normalised maps on the finest grid."""
    size = max(cams[k].shape[-1] for k in names)
    logs = []
    for k in names:
        c = torch.from_numpy(cams[k].astype(np.float32))  # (12, K, h, w)
        c = c / (c.flatten(1).amax(1).view(-1, 1, 1, 1) + 1e-8)
        c = F.interpolate(c, size=(size, size), mode="bilinear", align_corners=False)
        logs.append(torch.log(c.clamp_min(1e-4)))
    return torch.exp(torch.stack(logs).mean(0)).numpy()


@torch.no_grad()
def _logits(model, x):
    h = _no_grad_trunk(model, x, len(model.backbone.stages))
    with _amp():
        feats = _pool(model.backbone, h).float()[None]
    return model.head(feats)[0]


def _top_mask(cam, frac):
    """Boolean mask of the top ``frac`` of pixels of cam (K, RES, RES), globally."""
    flat = cam.flatten()
    k = max(1, int(frac * flat.numel()))
    thr = torch.topk(flat, k).values[-1]
    return (cam >= thr) & (cam > 0)


def focality(cam, frac):
    """Share of map mass in its top ``frac`` pixels (all windows of a study)."""
    flat = np.sort(cam.astype(np.float32).ravel())[::-1]
    total = flat.sum()
    return float(flat[:max(1, int(frac * flat.size))].sum() / total) if total > 0 else np.nan


def compactness(cam, frac, size=raptor.IMG):
    """Share of the top ``frac`` pixels of one window's map in its largest blob.

    Pseudo-boxes need one contiguous region: ~1.0 is a single blob, low values
    are scattered speckle that no box can summarise.
    """
    c = cv2.resize(cam.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    k = max(1, int(frac * c.size))
    thr = np.partition(c.ravel(), -k)[-k]
    binary = ((c >= thr) & (c > 0)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return float(areas.max() / areas.sum()) if len(areas) else np.nan


def deletion_test(model, x, maps, labels, frac, attn, n_shuffle=2, seed=0):
    """Per map type and label: logit drop for own-map vs shuffled-map blurring,
    plus focality and the compactness of the top-attention window's map."""
    gen = torch.Generator().manual_seed(seed)
    x_blur = gaussian_blur(x, kernel_size=61, sigma=15.0)
    base = _logits(model, x)
    rows = []
    for name, cams in maps.items():
        for n in labels:
            cam = torch.from_numpy(cams[n].astype(np.float32)).to(x.device)  # (K, h, w)
            k, h, w = cam.shape

            def drop(c):
                up = F.interpolate(c[:, None], size=x.shape[-2:], mode="nearest")[:, 0]
                m = _top_mask(up, frac)[:, None]
                return (base[n] - _logits(model, torch.where(m, x_blur, x))[n]).item()

            shuffled = []
            for _ in range(n_shuffle):
                perm = torch.stack([torch.randperm(h * w, generator=gen)
                                    for _ in range(k)]).to(x.device)
                shuffled.append(drop(torch.gather(cam.flatten(1), 1, perm).view(k, h, w)))
            rows.append({"map": name, "label": raptor.LABELS[n],
                         "base_logit": base[n].item(), "drop_own": drop(cam),
                         "drop_shuffled": float(np.mean(shuffled)),
                         "focality": focality(cams[n], frac),
                         "compact": compactness(cams[n][int(np.argmax(attn[:, n]))], frac)})
    return rows


# --------------------------------------------------------------------------- #
# Study selection / data
# --------------------------------------------------------------------------- #
def _load_tables():
    train = pd.read_csv(raptor.DATA_DIR / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train.dropna(subset=raptor.LABELS).set_index("StudyInstanceUID")[raptor.LABELS]
    llm = pd.read_csv(LLM_LABELS, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    return train, gold, llm


def select_studies(spec, train, gold, seed=0):
    if spec == "gold":
        return list(gold.index)
    if spec == "all":
        return list(train.StudyInstanceUID)
    if Path(spec).is_file():
        return [s.strip() for s in Path(spec).read_text().split() if s.strip()]
    rest = train.loc[~train.StudyInstanceUID.isin(gold.index), "StudyInstanceUID"]
    return list(rest.sample(int(spec), random_state=seed))


def _build(args):
    uid, rows = args
    try:
        vol, mask, slot_series = raptor.build_study(
            uid, rows, str(raptor.DATA_DIR / "train_series"))
        return uid, vol, mask, slot_series, None
    except Exception as e:  # corrupt / missing DICOMs: skip the study, keep going
        return uid, None, None, None, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# Gallery
# --------------------------------------------------------------------------- #
def _overlay(gray, cam, size):
    heat = np.clip(cv2.resize(cam.astype(np.float32), (size, size),
                              interpolation=cv2.INTER_CUBIC), 0, 1)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = (0.65 * heat)[..., None]
    return (gray * (1 - alpha) + color * alpha).astype(np.uint8)


def _strip(slice_u8, cams, size=176):
    """Raw slice followed by one overlay per map, as a base64 JPEG."""
    gray = cv2.cvtColor(cv2.resize(slice_u8, (size, size), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_GRAY2BGR)
    gap = np.full((size, 4, 3), 255, np.uint8)
    tiles = [gray]
    for c in cams:
        tiles += [gap, _overlay(gray, c, size)]
    ok, buf = cv2.imencode(".jpg", np.concatenate(tiles, axis=1),
                           [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode()


def _auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    r = pd.Series(p).rank().to_numpy()
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


def summarize(out_dir):
    """Per label x map: deletion win rate, median drops, median focality."""
    dele = pd.read_csv(Path(out_dir) / "deletion.csv", dtype={"study": str})
    dele["win"] = dele.drop_own > dele.drop_shuffled
    if "compact" not in dele:  # runs made before the compactness metric existed
        dele["compact"] = np.nan
    s = dele.groupby(["label", "map"], sort=False).agg(
        n=("win", "size"), win=("win", "mean"), own=("drop_own", "median"),
        shuffled=("drop_shuffled", "median"), focality=("focality", "median"),
        compact=("compact", "median"))
    return s.reset_index()


def build_gallery(out_dir, top_windows=2, prob_show=0.5):
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text())
    maps = meta["maps"]
    preds = pd.read_csv(out_dir / "preds.csv", dtype={"study": str})
    summ = summarize(out_dir)
    parts = [f"<title>Raptor Grad-CAM</title><style>{_CSS}</style>",
             "<h1>Raptor CoaTNet &middot; Grad-CAM</h1>",
             f"<p class=sub>{preds.study.nunique()} studies &middot; each strip: slice | "
             + " | ".join(html.escape(_MAP_TITLES.get(m, m)) for m in maps)
             + " &middot; maps normalised per label across the study</p>"]

    head = "".join(f"<th colspan=4>{html.escape(_MAP_TITLES.get(m, m))}</th>" for m in maps)
    sub = "<th>own&gt;shuf</th><th>drop</th><th>focal</th><th>blob</th>" * len(maps)
    rows = []
    for lab in raptor.LABELS:
        g = preds[(preds.label == lab)].dropna(subset=["gold"])
        cells = []
        for m in maps:
            r = summ[(summ.label == lab) & (summ.map == m)]
            if r.empty:
                cells.append("<td></td>" * 4)
                continue
            r = r.iloc[0]
            cells.append(f"<td>{r.win * 100:.0f}%</td><td>{r.own:.2f}</td>"
                         f"<td>{r.focality:.2f}</td><td>{r.compact:.2f}</td>")
        rows.append(f"<tr><td>{html.escape(lab)}</td><td>{_auc(g.gold, g.prob):.3f}</td>"
                    + "".join(cells) + "</tr>")
    parts.append(
        "<h2>Summary</h2><div class=scroll><table><tr><th rowspan=2>Label</th>"
        f"<th rowspan=2>AUC (gold)</th>{head}</tr><tr>{sub}</tr>" + "".join(rows)
        + "</table></div><p class=note>own&gt;shuf: share of cases where blurring the "
        "map's top pixels lowers the logit more than blurring spatially shuffled map "
        "cells (faithfulness). drop: median logit drop. focal: share of map mass in its "
        f"top {meta['frac'] * 100:.0f}% pixels ({meta['frac']:.2f} = flat). blob: share of "
        "the top pixels of the most-attended window in its largest connected blob "
        "(1.0 = one region, what a pseudo-box needs). Gold AUC is "
        "optimistic: the public weights were likely trained on these studies.</p>")

    for study in preds.study.unique():
        z = np.load(out_dir / "studies" / f"{study}.npz", allow_pickle=False)
        vol, attn, centers = z["vol"], z["attn"], z["centers"]
        cams = {m: z[f"cams_{m}"].astype(np.float32) for m in maps}
        p = preds[preds.study == study].set_index("label")
        show = [n for n, lab in enumerate(raptor.LABELS)
                if p.loc[lab, "gold"] == 1 or p.loc[lab, "prob"] >= prob_show]
        parts.append(f"<h2 class=study>{study[-24:]}</h2>")
        if not show:
            parts.append("<p class=note>No gold-positive or predicted-positive labels.</p>")
        for n in show:
            lab = raptor.LABELS[n]
            gold = p.loc[lab, "gold"]
            gold_s = "&ndash;" if pd.isna(gold) else ("+" if gold == 1 else "&minus;")
            parts.append(
                f"<div class=lab><b>{html.escape(lab)}</b> prob {p.loc[lab, 'prob']:.2f} "
                f"&middot; gold {gold_s} &middot; LLM {p.loc[lab, 'llm']:.2f}</div>")
            gmax = {m: cams[m][n].max() + 1e-8 for m in maps}
            for j in np.argsort(-attn[:, n])[:top_windows]:
                c = int(centers[j])
                slot = raptor.SLOT_NAMES[raptor.SLOT_OF_SLICE[c]]
                strip = _strip(vol[c], [cams[m][n, j] / gmax[m] for m in maps])
                parts.append(
                    f"<figure><img src='data:image/jpeg;base64,{strip}'><figcaption>{slot} "
                    f"&middot; slice {c} &middot; attn {attn[j, n] * 100:.0f}%</figcaption>"
                    "</figure>")
    path = out_dir / "gallery.html"
    path.write_text("\n".join(parts))
    return path


_MAP_TITLES = {"L3": "L3 12x12", "L2": "L2 24x24", "L1": "L1 48x48",
               "LC3": "LayerCAM 12x12", "LC2": "LayerCAM 24x24", "LC1": "LayerCAM 48x48"}

_CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--line:#e7e5e4}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--line:#44403c}}
body{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1400px}
h1{margin:0 0 4px}.sub,.note{color:var(--mut)}h2.study{margin-top:36px;border-top:1px solid var(--line);padding-top:16px;font-family:monospace}
table{border-collapse:collapse}td,th{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right}
td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto}
.lab{margin:14px 0 6px}figure{margin:0 0 8px}figure img{display:block;max-width:100%}
figcaption{color:var(--mut);font-size:12px}
"""


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _load_reused(args):
    uid, src = args
    z = np.load(Path(src) / "studies" / f"{uid}.npz", allow_pickle=False)
    return uid, z["vol"], z["mask"], list(z["slot_series"]), None


def _bounded_map(pool, fn, jobs, ahead):
    """Like ``pool.map`` but keeps at most ``ahead`` results in flight, so fast
    DICOM workers cannot pile decoded volumes up in RAM ahead of the GPU."""
    jobs, pending = iter(jobs), deque()
    for job in jobs:
        pending.append(pool.submit(fn, job))
        if len(pending) >= ahead:
            break
    while pending:
        yield pending.popleft().result()
        for job in jobs:
            pending.append(pool.submit(fn, job))
            break


def _positive_labels(uid, gold, llm, llm_thr):
    """Label indices treated as positive: gold labels if present, else LLM >= thr."""
    if uid in gold.index:
        return [n for n, lab in enumerate(raptor.LABELS) if gold.loc[uid, lab] == 1]
    if uid in llm.index:
        return [n for n, lab in enumerate(raptor.LABELS) if llm.loc[uid, lab] >= llm_thr]
    return []


def _write_preds(out_dir, uids, gold, llm):
    """preds.csv from the saved study files (so resumed runs are complete)."""
    rows = []
    for uid in uids:
        f = out_dir / "studies" / f"{uid}.npz"
        if not f.exists():
            continue
        probs = np.load(f, allow_pickle=False)["probs"]
        g = gold.loc[uid] if uid in gold.index else None
        for n, lab in enumerate(raptor.LABELS):
            rows.append({"study": uid, "label": lab, "prob": probs[n],
                         "gold": np.nan if g is None else g[lab],
                         "llm": llm.loc[uid, lab] if uid in llm.index else np.nan})
    pd.DataFrame(rows).to_csv(out_dir / "preds.csv", index=False)
    return len(rows) // len(raptor.LABELS)


def load_explained(spec, device):
    """(model, windows fn): ``raptor`` or a ``src.models.mil25d`` checkpoint path."""
    if spec == "raptor":
        return raptor.load_model(device), raptor.eval_windows
    from src.models import mil25d
    ck = torch.load(spec, map_location="cpu", weights_only=True)
    a = ck["args"]
    model = mil25d.build_model(a["backbone"], pretrained=False)
    model.load_state_dict(ck["model"])
    img = a["img"]

    def windows(vol, centers):
        idx = np.clip(np.asarray(centers)[:, None] + np.arange(-1, 2), 0, len(vol) - 1)
        x = torch.from_numpy(vol[idx])[None]
        return mil25d.prepare(x, torch.from_numpy(np.asarray(centers))[None], img, augment=False)

    return model.eval().to(device), windows


def run(studies="gold", maps=None, frac=0.05, workers=6, out_dir=OUT_DIR,
        reuse=None, seed=0, fast=False, llm_thr=0.7, model_spec="raptor"):
    """Grad-CAM over ``studies``.

    ``reuse``: a previous output dir whose saved study volumes replace DICOM reads.
    ``fast``: bulk mode for pseudo-box generation.  Skips the deletion test and
    the gallery, defaults to the maps the box generator uses (L3, LC1+L3),
    stores maps other than L3 only for positive labels (gold, else LLM >=
    ``llm_thr``), and resumes by skipping studies already written.
    """
    out_dir = Path(out_dir)
    (out_dir / "studies").mkdir(parents=True, exist_ok=True)
    maps = list(maps or (FAST_MAPS if fast else DEFAULT_MAPS))
    _, layers = parse_maps(maps)
    (out_dir / "meta.json").write_text(json.dumps(
        {"maps": maps, "frac": frac, "fast": fast, "llm_thr": llm_thr, "model": model_spec}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, gold, llm = _load_tables()
    uids = select_studies(studies, train, gold, seed)
    todo = [u for u in uids if not (out_dir / "studies" / f"{u}.npz").exists()] if fast else uids
    model, windows = load_explained(model_spec, device)
    print(f"{len(uids)} studies ({len(uids) - len(todo)} already done) | maps {maps} | "
          f"{'fast' if fast else 'full'} | device {device}", flush=True)

    if reuse:
        loader, jobs = _load_reused, [(u, reuse) for u in todo]
    else:
        series = raptor.load_series_table("train")
        loader, jobs = _build, [(u, series.get(u, [])) for u in todo]
    del_rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (uid, vol, mask, slot_series, err) in enumerate(
                _bounded_map(pool, loader, jobs, ahead=2 * workers)):
            if err:
                print(f"  skip {uid[-12:]}: {err}", flush=True)
                continue
            centers = raptor.eval_centers(mask)
            x = windows(vol, centers).to(device)
            try:
                probs, attn, base = gradcam(model, x, layers)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM on {uid[-12:]}; skipping", flush=True)
                continue
            out = {m: base[m] if m in base else ensemble(base, m.split("+")) for m in maps}
            g = gold.loc[uid] if uid in gold.index else None
            if fast:
                keep = np.zeros(len(raptor.LABELS), bool)
                keep[_positive_labels(uid, gold, llm, llm_thr)] = True
                for m in maps:
                    if m != "L3":
                        out[m][~keep] = 0  # zeros compress to almost nothing
            else:
                test_labels = [n for n, lab in enumerate(raptor.LABELS)
                               if probs[n] >= 0.3 or (g is not None and g[lab] == 1)]
                for r in deletion_test(model, x, out, test_labels, frac, attn, seed=seed + i):
                    del_rows.append({"study": uid, **r})
            np.savez_compressed(
                out_dir / "studies" / f"{uid}.npz", vol=vol, mask=mask, centers=centers,
                probs=probs, attn=attn, slot_series=np.array([s or "" for s in slot_series]),
                **{f"cams_{k}": v.astype(np.float16) for k, v in out.items()})
            if not fast or (i + 1) % 25 == 0 or i + 1 == len(todo):
                rate = (time.time() - t0) / (i + 1)
                print(f"  [{i + 1}/{len(todo)}] {uid[-12:]} | {time.time() - t0:.0f}s | "
                      f"{rate:.1f}s/study | eta {rate * (len(todo) - i - 1) / 3600:.1f}h",
                      flush=True)

    n = _write_preds(out_dir, uids, gold, llm)
    print(f"preds.csv: {n} studies", flush=True)
    if fast:
        return out_dir
    pd.DataFrame(del_rows).to_csv(out_dir / "deletion.csv", index=False)
    summarize(out_dir).to_csv(out_dir / "summary.csv", index=False)
    path = build_gallery(out_dir)
    print(f"gallery: {path}", flush=True)
    return path


@register_model("raptor_gradcam")
class RaptorGradCamPipeline:
    def run(self):
        run()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--studies", default="gold",
                    help="'gold' (58 manual-label studies), 'all' (every train study), "
                         "an integer N (random non-gold studies), or a file of UIDs")
    ap.add_argument("--maps", nargs="+", default=None,
                    help="maps to compute: L<k> Grad-CAM or LC<k> LayerCAM at stage k "
                         "(raptor 1-3: 48/24/12; ConvNeXt-T 256 px 0-3: 64/32/16/8); join with + for a "
                         "geometric-mean ensemble, e.g. LC1+L3")
    ap.add_argument("--frac", type=float, default=0.05,
                    help="fraction of pixels blurred in the deletion test")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default="raptor", help="'raptor' or a mil25d checkpoint (best.pt)")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--reuse", help="earlier output dir to load study volumes from")
    ap.add_argument("--gallery-only", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="bulk pseudo-box mode: no deletion test or gallery, maps "
                         f"{FAST_MAPS}, fine maps only for positives, resumable")
    ap.add_argument("--llm-thr", type=float, default=0.7,
                    help="LLM soft-label threshold for positives in --fast mode")
    a = ap.parse_args()
    if a.gallery_only:
        print(build_gallery(a.out))
    else:
        run(a.studies, a.maps, a.frac, a.workers, a.out, a.reuse, fast=a.fast,
            llm_thr=a.llm_thr, model_spec=a.model)


if __name__ == "__main__":
    main()
