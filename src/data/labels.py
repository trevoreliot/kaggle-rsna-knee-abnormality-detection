"""Soft training labels from the public LLM report-label tables.

Sources (all CC0, under ``custom_labels/``):
  steven    stevenleehans llm_labels_v4_blend   (gold macro AUC 0.893)
  pilkwang  pilkwang report_labels_v2           (0.866)
  lixin     lixin73 labels_llm_gpt56sol         (0.835)
  qwen      laymond qwen_knee_weak_labels       (0.818)
``consensus4`` is their unweighted mean clipped to [0.01, 0.99] (0.896), the
recipe of the public goodpjw2008 ConvNeXt reader (``make_labels.py`` v0).

Usage:
    uv run python -m src.data.labels            # writes assets/processed/labels_<name>.csv
"""

import pandas as pd

from src.explain import raptor
from src.explain.raptor_gradcam import _auc

LABELS = raptor.LABELS
SRC_DIR = raptor.DATA_DIR / "custom_labels"
OUT_DIR = raptor.DATA_DIR.parents[1] / "processed"


def _qwen():
    q = pd.read_csv(SRC_DIR / "laymond/qwen_knee_weak_labels.csv")
    return q[["StudyInstanceUID"] + [f"{c}__label" for c in LABELS]].set_axis(
        ["StudyInstanceUID"] + LABELS, axis=1)


SOURCES = {
    "steven": lambda: pd.read_csv(SRC_DIR / "slh/kneelabels-stevenleehans/llm_labels_v4_blend.csv"),
    "pilkwang": lambda: pd.read_csv(SRC_DIR / "pilkwang/report_labels_v2.csv"),
    "lixin": lambda: pd.read_csv(SRC_DIR / "lixin73/labels_llm_gpt56sol.csv"),
    "qwen": _qwen,
}


def load_sources(ids):
    return {k: f().set_index("StudyInstanceUID")[LABELS].astype(float).reindex(ids)
            for k, f in SOURCES.items()}


def consensus(src, names=None):
    names = names or list(src)
    num = sum(src[k].fillna(0) for k in names)
    den = sum(src[k].notna() for k in names)
    return (num / den).clip(0.01, 0.99)


def build():
    tr = pd.read_csv(raptor.DATA_DIR / "train.csv")
    gold = tr.set_index("StudyInstanceUID")[LABELS]
    is_gold = gold.notna().all(axis=1)
    src = load_sources(tr.StudyInstanceUID)
    tables = {"steven": src["steven"], "consensus4": consensus(src)}
    auc = pd.DataFrame({k: [_auc(gold.loc[is_gold, c], v.loc[is_gold, c].fillna(0.5)) for c in LABELS]
                        for k, v in {**src, "consensus4": tables["consensus4"]}.items()}, index=LABELS)
    auc.loc["MEAN"] = auc.mean()
    print("AUC vs the 58 gold studies:\n" + auc.round(3).to_string())
    for name, t in tables.items():
        path = OUT_DIR / f"labels_{name}.csv"
        t.to_csv(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    build()
