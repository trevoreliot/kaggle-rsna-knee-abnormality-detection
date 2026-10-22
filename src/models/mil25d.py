"""Phase 5: our own 2.5D attention-MIL classifier, trained with k-fold CV.

Same design as the public raptor model, trained from an ImageNet backbone:
  - input: study cache volumes (64 slices / 5 plane slots, 140 mm crop, 336 px);
  - a window is slices (c-1, c, c+1) as RGB; the timm backbone embeds each window;
  - per-label attention pooling over windows (``raptor.RaptorClassifier`` head).

Training samples ``--k-train`` random windows per study (attention pooling takes
any count); evaluation uses raptor's 62 evenly spaced windows.  Augmentation
runs on the GPU, one random affine + intensity jitter per study.  Horizontal
flips are applied to coronal and axial windows only: a flipped right knee looks
like a left knee with the same labels, while a flipped sagittal slice would put
the patella behind the knee.

Labels: community LLM soft labels as BCE targets, ``--labels`` picks a table
from ``src.data.labels`` (``steven`` = stevenleehans v4_blend, ``consensus4`` =
mean of four public tables).  The 58 manually labelled studies are
never trained on; they are the clean test set.  Folds split the rest by study.

Usage:
    uv run python -m src.models.mil25d --fold 0                 # one fold
    uv run python -m src.models.mil25d --fold all --epochs 12   # 5-fold CV
    uv run python -m src.models.mil25d --fold 0 --epochs 1 --limit 200   # smoke test
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.study_cache import StudyCache
from src.explain import raptor
from src.explain.raptor_gradcam import _auc
from src.models.registry import register_model

LABELS = raptor.LABELS
LABELS_DIR = Path("assets/processed")
OUT_DIR = Path("assets/processed/mil_runs")
FOLDS_CSV = Path("assets/processed/folds.csv")
N_FOLDS = 5
# Slots whose image x axis is left-right (safe to flip): Cor fluid, Cor T1/PD, Axial.
FLIP_OK = torch.tensor(np.isin(raptor.SLOT_OF_SLICE, [2, 3, 4]))


# --------------------------------------------------------------------------- #
# Labels and folds
# --------------------------------------------------------------------------- #
def load_labels(name="consensus4"):
    """(targets df indexed by study: LLM soft labels, gold df of manual 0/1 labels)."""
    tr = pd.read_csv(raptor.DATA_DIR / "train.csv")
    gold = tr.dropna(subset=LABELS, how="all").set_index("StudyInstanceUID")[LABELS]
    llm = pd.read_csv(LABELS_DIR / f"labels_{name}.csv").set_index("StudyInstanceUID")[LABELS]
    return llm, gold


def report_group(text):
    """Normalised report text: studies with identical reports share a fold."""
    return re.sub(r"\W+", "", str(text).lower())


def make_folds(uids, gold, seed=42, path=FOLDS_CSV):
    """Fold per study (-1 = gold test set), grouped by report text; written once, then reused.

    Groups are dealt largest first to the currently smallest fold (random order
    within equal sizes), so folds stay balanced.
    """
    if path.exists():
        f = pd.read_csv(path).set_index("StudyInstanceUID").fold
        if set(f.index) == set(uids):
            return f
    tr = pd.read_csv(raptor.DATA_DIR / "train.csv").set_index("StudyInstanceUID")
    groups = tr.Report.reindex(uids).map(report_group)
    rest = groups[~groups.index.isin(gold.index)]
    rng = np.random.default_rng(seed)
    members = list(rest.groupby(rest).groups.values())
    order = sorted(rng.permutation(len(members)), key=lambda i: -len(members[i]))
    f = pd.Series(-1, index=sorted(uids), name="fold")
    sizes = np.zeros(N_FOLDS, int)
    for i in order:
        k = int(np.argmin(sizes))
        f[list(members[i])] = k
        sizes[k] += len(members[i])
    f.index.name = "StudyInstanceUID"
    f.to_csv(path)
    return f


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _valid_centers(mask):
    valid = np.flatnonzero(mask)
    if len(valid) < 3:
        return np.array([1])
    return np.arange(valid.min() + 1, valid.max())


class StudyWindows(Dataset):
    """uint8 windows (K, 3, 336, 336), their centre slices, and the soft target."""

    def __init__(self, cache, uids, targets, k=None, seed=0):
        self.cache, self.uids, self.targets, self.k = cache, list(uids), targets, k
        self.seed = seed

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        vol, mask = self.cache[uid]
        if self.k is None:
            centers = raptor.eval_centers(mask)
        else:
            cs = _valid_centers(mask)
            centers = np.sort(np.random.choice(cs, self.k, replace=len(cs) < self.k))
        idx = np.clip(centers[:, None] + np.arange(-1, 2), 0, len(vol) - 1)
        y = self.targets.loc[uid].to_numpy(np.float32) if self.targets is not None else np.zeros(12, np.float32)
        return torch.from_numpy(vol[idx]), torch.from_numpy(centers), torch.from_numpy(y)


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prepare(x, centers, img, augment, gen=None):
    """uint8 (B, K, 3, H, W) on GPU -> normalised float (B*K, 3, img, img)."""
    b, k = x.shape[:2]
    x = x.flatten(0, 1).float().div_(255)
    dev = x.device
    if augment:
        ang = (torch.rand(b, device=dev) - 0.5) * 2 * math.radians(12)
        scale = 1 + (torch.rand(b, device=dev) - 0.5) * 2 * 0.12
        shift = (torch.rand(b, 2, device=dev) - 0.5) * 2 * 0.08
        flip = (torch.rand(b, device=dev) < 0.5).float()
        cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
        theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], -1),
                             torch.stack([sin, cos, shift[:, 1]], -1)], 1)  # (B, 2, 3)
        theta = theta.repeat_interleave(k, 0)
        fl = flip.repeat_interleave(k) * FLIP_OK.to(dev)[centers.flatten()].float()
        theta[:, 0, :2] *= (1 - 2 * fl)[:, None]  # mirror input x for flipped windows
        grid = F.affine_grid(theta, (b * k, 3, img, img), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        con = 1 + (torch.rand(b, device=dev) - 0.5) * 0.3
        bri = (torch.rand(b, device=dev) - 0.5) * 0.15
        x = (x * con.repeat_interleave(k)[:, None, None, None]
             + bri.repeat_interleave(k)[:, None, None, None]).clamp_(0, 1)
    elif x.shape[-1] != img:
        x = F.interpolate(x, size=(img, img), mode="bilinear", align_corners=False)
    return (x - _MEAN.to(dev)) / _STD.to(dev)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LabelGraphClassifier(raptor.RaptorClassifier):
    """Attention-MIL head plus a 2-layer GCN over the 12 per-label pooled features.

    Nodes are the label-specific attention-pooled study features; edges are
    label co-occurrence on the training fold (ML-GCN style: conditional
    probability P(j | i), thresholded, row-normalised, plus self loops).  The
    GCN is residual with a zero-initialised output layer, so training starts
    from the plain MIL head.
    """

    def __init__(self, backbone, f_dim=768, n=12, drop=0.2, hidden=256):
        super().__init__(backbone, f_dim, n, drop)
        self.register_buffer("adj", torch.eye(n))
        self.g1 = nn.Linear(f_dim, hidden)
        self.g2 = nn.Linear(hidden, f_dim)
        nn.init.zeros_(self.g2.weight)
        nn.init.zeros_(self.g2.bias)

    def head(self, feats):
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        z = F.gelu(self.adj @ self.g1(pooled))
        pooled = pooled + self.adj @ self.g2(z)
        return (pooled * self.clsW).sum(-1) + self.clsb


def label_adjacency(y, tau=0.3):
    """Row-normalised co-occurrence graph from soft labels (studies x labels)."""
    b = (np.asarray(y) >= 0.5).astype(np.float64)
    co = b.T @ b
    cond = co / np.maximum(np.diag(co)[:, None], 1)  # P(j | i)
    np.fill_diagonal(cond, 0)
    adj = (cond >= tau).astype(np.float64) + np.eye(len(cond))
    return torch.tensor(adj / adj.sum(1, keepdims=True), dtype=torch.float32)


def build_model(backbone, pretrained=True, grad_ckpt=False, drop_path=0.1, head="mil"):
    bb = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg",
                           drop_path_rate=drop_path)
    if grad_ckpt:
        bb.set_grad_checkpointing(True)
    cls = LabelGraphClassifier if head == "graph" else raptor.RaptorClassifier
    return cls(bb, f_dim=bb.num_features)


def forward(model, x, b, k):
    return model.head(model.backbone(x).view(b, k, -1).float())


@torch.no_grad()
def predict(model, loader, img, device, chunk=32):
    model.eval()
    out = []
    for xb, cb, _ in loader:
        for x, c in zip(xb, cb):  # one study at a time: K is fixed at 62 here
            x = prepare(x[None].to(device, non_blocking=True), c[None], img, augment=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = torch.cat([model.backbone(xc) for xc in x.split(chunk)])
            out.append(torch.sigmoid(model.head(f.float()[None]))[0].cpu().numpy())
    return np.stack(out)


def macro_auc(y, p, soft=True):
    """Macro AUC; soft LLM targets are binarised (>= 0.7 pos, <= 0.3 neg, rest dropped)."""
    aucs = []
    for j in range(y.shape[1]):
        t, s = y[:, j], p[:, j]
        if soft:
            keep = (t >= 0.7) | (t <= 0.3)
            t, s = (t[keep] >= 0.7).astype(int), s[keep]
        aucs.append(_auc(t, s))
    return float(np.nanmean(aucs)), aucs


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_fold(fold, a, cache, llm, gold, folds, device, val_ref=None):
    """``llm``: training targets; ``val_ref``: soft labels the val split is scored against."""
    val_ref = llm if val_ref is None else val_ref
    out = Path(a.out) / a.name / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    tr_ids = folds.index[(folds != fold) & (folds >= 0)].tolist()
    va_ids = folds.index[folds == fold].tolist()
    if a.limit:
        tr_ids, va_ids = tr_ids[:a.limit], va_ids[:max(a.limit // 4, 20)]
    go_ids = gold.index.intersection(cache.uids).tolist()
    print(f"fold {fold}: train {len(tr_ids)}, val {len(va_ids)}, gold test {len(go_ids)}")

    kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    tr_dl = DataLoader(StudyWindows(cache, tr_ids, llm, k=a.k_train), batch_size=a.batch,
                       shuffle=True, drop_last=True, **kw)
    va_dl = DataLoader(StudyWindows(cache, va_ids, llm), batch_size=4, **kw)
    go_dl = DataLoader(StudyWindows(cache, go_ids, None), batch_size=4, **kw)

    model = build_model(a.backbone, grad_ckpt=a.grad_ckpt, head=a.head)
    if a.head == "graph":
        model.adj.copy_(label_adjacency(llm.loc[tr_ids].to_numpy(), a.graph_tau))
        print("label graph edges:", {LABELS[i]: [LABELS[j] for j in np.flatnonzero(r) if j != i]
                                     for i, r in enumerate(model.adj.numpy())})
    model = model.to(device).to(memory_format=torch.channels_last)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": a.lr},
                             {"params": head, "lr": a.lr * 10}], weight_decay=a.wd)
    steps = a.epochs * len(tr_dl)
    warm = min(len(tr_dl), steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    ema = None
    if a.ema:  # evaluated and saved in place of the raw weights
        ema = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema))
    best, log = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for i, (x, c, y) in enumerate(tr_dl):
            x = prepare(x.to(device, non_blocking=True), c.to(device), a.img, augment=True)
            x = x.contiguous(memory_format=torch.channels_last)
            y = y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward(model, x, y.shape[0], a.k_train)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            tot, n = tot + loss.item() * len(y), n + len(y)
            if i % 100 == 0:
                print(f"  ep {ep} it {i}/{len(tr_dl)} loss {tot / n:.4f} "
                      f"{n / (time.time() - t0):.1f} studies/s", flush=True)
        t_train = time.time() - t0
        ev = model if ema is None else ema.module
        p_va = predict(ev, va_dl, a.img, device)
        auc_va, _ = macro_auc(val_ref.loc[va_ids].to_numpy(), p_va)
        p_go = predict(ev, go_dl, a.img, device)
        auc_go, _ = macro_auc(gold.loc[go_ids].to_numpy(), p_go, soft=False)
        row = dict(epoch=ep, loss=tot / n, val_auc_llm=auc_va, gold_auc=auc_go,
                   train_s=round(t_train), total_s=round(time.time() - t0))
        log.append(row)
        print(f"fold {fold} ep {ep}: " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                                   for k, v in row.items()), flush=True)
        pd.DataFrame(log).to_csv(out / "log.csv", index=False)
        if auc_va > best:  # select on the fold's own val split, never on gold
            best = auc_va
            torch.save({"model": ev.state_dict(), "args": vars(a), "epoch": ep}, out / "best.pt")
            pd.DataFrame(p_va, index=pd.Index(va_ids, name="StudyInstanceUID"), columns=LABELS).to_csv(out / "oof.csv")
            pd.DataFrame(p_go, index=pd.Index(go_ids, name="StudyInstanceUID"), columns=LABELS).to_csv(out / "gold.csv")
    (out / "best.json").write_text(json.dumps({"val_auc_llm": best, **log[int(np.argmax([r["val_auc_llm"] for r in log]))]}))
    return best


def summarize(run_dir, llm, gold):
    """Pool OOF predictions and average the folds' gold predictions."""
    run_dir = Path(run_dir)
    oof = pd.concat([pd.read_csv(p, index_col=0) for p in sorted(run_dir.glob("fold*/oof.csv"))])
    golds = [pd.read_csv(p, index_col=0) for p in sorted(run_dir.glob("fold*/gold.csv"))]
    g = sum(golds) / len(golds)
    auc_oof, per_oof = macro_auc(llm.loc[oof.index].to_numpy(), oof.to_numpy())
    auc_gold, per_gold = macro_auc(gold.loc[g.index].to_numpy(), g.to_numpy(), soft=False)
    t = pd.DataFrame({"oof_llm": per_oof, "gold": per_gold}, index=LABELS)
    print(f"{run_dir.name}: {len(golds)} folds | OOF macro AUC (LLM ref) {auc_oof:.4f} on {len(oof)} | "
          f"gold macro AUC (fold-mean) {auc_gold:.4f} on {len(g)}")
    print(t.round(3).to_string())
    t.to_csv(run_dir / "auc.csv")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold", default="0", help="fold index or 'all'")
    ap.add_argument("--name", default="convnext_tiny_cons4")
    ap.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--labels", default="consensus4", help="assets/processed/labels_<name>.csv")
    ap.add_argument("--eval-labels", default="consensus4", help="labels the val split is scored against")
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--k-train", type=int, default=24, help="windows sampled per study when training")
    ap.add_argument("--batch", type=int, default=4, help="studies per step")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--head", default="mil", choices=["mil", "graph"],
                    help="graph = label co-occurrence GCN on the pooled features")
    ap.add_argument("--graph-tau", type=float, default=0.3, help="edge threshold on P(j | i)")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay per step (0 = off)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="train on N studies (smoke test)")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--summarize", action="store_true", help="only pool existing fold outputs")
    return ap.parse_args(argv)


def run(a):
    torch.backends.cudnn.benchmark = True
    llm, gold = load_labels(a.labels)
    val_ref, _ = load_labels(a.eval_labels)
    if a.summarize:
        return summarize(Path(a.out) / a.name, val_ref, gold)
    cache = StudyCache("train")
    folds = make_folds(cache.uids, gold)
    device = torch.device("cuda")
    todo = range(N_FOLDS) if a.fold == "all" else [int(a.fold)]
    for k in todo:
        train_fold(k, a, cache, llm, gold, folds, device, val_ref)
    if a.fold == "all":
        summarize(Path(a.out) / a.name, val_ref, gold)


@register_model("mil25d")
class MIL25D:
    def run(self):
        run(parse_args([]))


if __name__ == "__main__":
    run(parse_args())
