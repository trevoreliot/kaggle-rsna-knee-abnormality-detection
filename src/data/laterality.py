"""Knee laterality (left / right) per study, plus per-series orientation.

Four independent signals, cross-checked against each other:
  tag     DICOM Laterality / ImageLaterality (about half the series carry it)
  desc    SeriesDescription side markers (``LT_t2_tse_cor``, ``RT. PD_TSE_SAG``)
  report  side phrases in the radiology report (train only; multilingual)
  geom    sign of the image centre's patient-x (LPS: x < 0 is patient right).
          Weak: the scanned knee sits near isocentre, not the body midline.

Each source agrees with each other ~98% of the time on train (geom included:
patients are scanned with the knee consistently off-isocentre), so the final
side is a majority vote over the available sources, ties broken by
tag > desc > report > geom.  ``side_conf`` is the agreeing share of sources.

Per-series output also keeps ImageOrientationPatient so downstream code can
tell which image direction is medial:  for a right knee, medial is patient
+x (towards the patient's left); for a left knee, patient -x.  ``medial_sign``
gives the sign of the image column axis (or slice axis for sagittal) that
points medially.

Usage:
    uv run python -m src.data.laterality             # train
    uv run python -m src.data.laterality --split test
"""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

DATA_DIR = Path("assets/big-data/rsna-knee-abnormality-detection")
OUT_DIR = Path("assets/processed")
TAGS = ["Laterality", "ImageLaterality", "SeriesDescription", "ImagePositionPatient",
        "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns"]

_DESC_R = re.compile(r"(?:^|[^a-z])(rt|right|rechts|der|dcha)(?:[^a-z]|$)")
_DESC_L = re.compile(r"(?:^|[^a-z])(lt|left|links|izq)(?:[^a-z]|$)")
# High-precision "<side> knee" phrases; reports also mention sides of other things.
_REPORT_R = re.compile(
    r"right knee|knee,? right|rechte[rsn]? knie|knie,? re(?:chts)?\b|"
    r"rodilla derecha|joelho direito|ginocchio destro|genou droit|"
    r"δεξιο[υύ]? γονατ|desn(?:o|og|om) koljen|koljen[oa]? desn|sağ diz|kolan[oa] praw",
    re.IGNORECASE)
_REPORT_L = re.compile(
    r"left knee|knee,? left|linke[rsn]? knie|knie,? li(?:nks)?\b|"
    r"rodilla izquierda|joelho esquerdo|ginocchio sinistro|genou gauche|"
    r"αριστερο[υύ]? γονατ|lijev(?:o|og|om) koljen|koljen[oa]? lijev|sol diz|kolan[oa] lew",
    re.IGNORECASE)


def _vec(v, n):
    try:
        a = np.array([float(x) for x in v], float)
    except (TypeError, ValueError):
        return None
    return a if len(a) >= n else None


def _probe(item):
    study, series, path = item
    row = {"StudyInstanceUID": study, "SeriesInstanceUID": series}
    try:
        files = sorted(p for p in Path(path).iterdir() if p.suffix == ".dcm")
        if not files:
            return row
        h = pydicom.dcmread(files[len(files) // 2], stop_before_pixels=True, force=True)
        for t in TAGS:
            v = getattr(h, t, None)
            if v is None or v == "":
                continue
            row[t] = ("|".join(str(x) for x in v)
                      if type(v).__name__ in ("MultiValue", "list", "tuple") else str(v))
    except Exception as e:  # corrupt header: keep the series with no tags
        row["error"] = f"{type(e).__name__}: {e}"[:120]
    return row


def scan_headers(split, threads=16):
    base = DATA_DIR / f"{split}_series"
    items = [(st.name, se.name, se) for st in base.iterdir() if st.is_dir()
             for se in st.iterdir() if se.is_dir()]
    with ThreadPoolExecutor(threads) as pool:
        return pd.DataFrame(pool.map(_probe, items))


def _geom_x(r):
    ipp = _vec(str(r.get("ImagePositionPatient", "")).split("|"), 3)
    iop = _vec(str(r.get("ImageOrientationPatient", "")).split("|"), 6)
    ps = _vec(str(r.get("PixelSpacing", "")).split("|"), 2)
    try:
        rows, cols = float(r["Rows"]), float(r["Columns"])
    except (KeyError, TypeError, ValueError):
        return np.nan
    if ipp is None or iop is None or ps is None:
        return np.nan
    c = ipp + iop[:3] * ps[1] * cols / 2 + iop[3:6] * ps[0] * rows / 2
    return float(c[0])


def _vote(values):
    v = [x for x in values if x in ("L", "R")]
    if not v:
        return None
    n_r = v.count("R")
    return "R" if n_r > len(v) - n_r else "L" if n_r < len(v) - n_r else None


def _desc_side(d):
    d = str(d).lower()
    r, l = bool(_DESC_R.search(d)), bool(_DESC_L.search(d))
    return "R" if r and not l else "L" if l and not r else None


def _report_side(text):
    if not isinstance(text, str):
        return None
    r, l = len(_REPORT_R.findall(text)), len(_REPORT_L.findall(text))
    return "R" if r > l else "L" if l > r else None


def medial_sign(side, iop, plane):
    """+1 / -1: which way along the image axis points medially (None if unknown).

    Coronal / axial: the image column axis (increasing x pixel index), whose
    patient direction is ``iop[:3]``.  Sagittal: the slice-normal axis
    (increasing slice position), patient direction ``cross(iop[:3], iop[3:])``.
    """
    iop = _vec(str(iop).split("|"), 6)
    if side not in ("L", "R") or iop is None:
        return None
    axis = np.cross(iop[:3], iop[3:]) if plane == "Sagittal" else iop[:3]
    medial_x = 1.0 if side == "R" else -1.0  # LPS: +x is the patient's left
    s = np.sign(axis[0] * medial_x)
    return int(s) if s != 0 else None


def laterality(split="train", min_geom_mm=20.0):
    hdr = scan_headers(split)
    hdr["tag"] = [_vote([str(a).strip().upper()[:1], str(b).strip().upper()[:1]])
                  for a, b in zip(hdr.get("Laterality", pd.Series(index=hdr.index)),
                                  hdr.get("ImageLaterality", pd.Series(index=hdr.index)))]
    hdr["desc"] = hdr.get("SeriesDescription", pd.Series(index=hdr.index)).map(_desc_side)
    hdr["geom_x"] = hdr.apply(_geom_x, axis=1)

    g = hdr.groupby("StudyInstanceUID")
    st = pd.DataFrame({
        "tag": g.tag.agg(lambda s: _vote(list(s))),
        "desc": g.desc.agg(lambda s: _vote(list(s))),
        "geom_x": g.geom_x.median(),
    })
    st["geom"] = np.where(st.geom_x.abs() < min_geom_mm, None,
                          np.where(st.geom_x < 0, "R", "L"))
    st.loc[st.geom_x.isna(), "geom"] = None
    if split == "train":
        rep = pd.read_csv(DATA_DIR / "train.csv", usecols=["StudyInstanceUID", "Report"])
        st["report"] = rep.set_index("StudyInstanceUID").Report.map(_report_side)
    else:
        st["report"] = None

    srcs = ["tag", "desc", "report", "geom"]
    print(f"{split}: {len(st)} studies")
    for s in srcs:
        print(f"  {s:6s} coverage {st[s].notna().mean():6.1%}")
    agree = {}
    for i, a in enumerate(srcs):
        for b in srcs[i + 1:]:
            both = st[a].notna() & st[b].notna()
            if both.sum():
                agree[(a, b)] = (st.loc[both, a] == st.loc[both, b]).mean()
                print(f"  {a}~{b}: agree {agree[(a, b)]:6.1%} on {both.sum()}")

    # geom only votes if it clearly agrees with the stronger signals.
    voters = srcs if agree.get(("tag", "geom"), 0) >= 0.9 else srcs[:3]

    def decide(r):
        v = [r[s] for s in voters if r[s] in ("L", "R")]
        if not v:
            return None, np.nan, 0
        n_r = v.count("R")
        if n_r * 2 == len(v):
            side = v[0]  # tie: highest-priority source wins
        else:
            side = "R" if n_r * 2 > len(v) else "L"
        return side, v.count(side) / len(v), len(v)

    st[["side", "side_conf", "n_sources"]] = st.apply(decide, axis=1, result_type="expand")
    print(f"  voters {voters}; resolved {st.side.notna().mean():.1%}; "
          f"unanimous {(st.side_conf == 1).mean():.1%}; "
          f"sources per study {st.n_sources.value_counts().sort_index().to_dict()}")

    planes = pd.read_csv(DATA_DIR / f"{split}_series.csv",
                         dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    ser = hdr.merge(planes, on=["StudyInstanceUID", "SeriesInstanceUID"], how="left")
    ser["side"] = ser.StudyInstanceUID.map(st.side)
    ser["medial_sign"] = [medial_sign(s, i, p) for s, i, p in
                          zip(ser.side, ser.get("ImageOrientationPatient"), ser.Anatomical_Plane)]
    return st.reset_index(), ser


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    st, ser = laterality(a.split)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st.to_csv(out / f"laterality_{a.split}.csv", index=False)
    ser.to_csv(out / f"series_orientation_{a.split}.csv", index=False)
    print(f"wrote {out / f'laterality_{a.split}.csv'} and series_orientation_{a.split}.csv")


if __name__ == "__main__":
    main()
