"""Ratios, the valid-point filter, Quality and Statistics, and the text export.

Conventions taken from the vendor's own exports (checked on three of them):

* a ratio whose denominator is zero is written as 0, not inf/NaN;
* "Ratio" columns follow the ratio type: D3/G divides D3 by G, G/D3 the reverse;
* the export lists the numerator peak's three columns, then the denominator's,
  then the three ratios.

The Statistics block reports the mean and standard deviation over valid
points, as Q-Carbon does, plus the median and interquartile range. Per-pixel
ratios with a near-zero denominator are heavy tailed, so the mean of a ratio
map can sit far from its typical value.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .fitter import FitResult

QUANTITIES = ("intensity", "width", "area")


def safe_ratio(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    out = np.zeros(np.broadcast(a, b).shape)
    np.divide(a, b, out=out, where=b != 0)
    return out


def ratios(res: FitResult, cfg: Config) -> dict:
    n, d = res.peak(cfg.ratio_num), res.peak(cfg.ratio_den)
    return {q: safe_ratio(n[q], d[q]) for q in QUANTITIES}


def valid_mask(res: FitResult, cfg: Config) -> np.ndarray:
    n, d = res.peak(cfg.ratio_num), res.peak(cfg.ratio_den)
    r = ratios(res, cfg)
    peak_i = np.maximum(n["intensity"], d["intensity"])
    ok = cfg.intensity.admits(peak_i)
    ok &= cfg.ratio_i.admits(r["intensity"])
    ok &= cfg.ratio_a.admits(r["area"])
    return ok & np.isfinite(res.std_error)


def quality(res: FitResult, cfg: Config, valid=None) -> dict:
    v = valid_mask(res, cfg) if valid is None else valid
    return {
        "total_points": int(v.size),
        "valid_points": int(v.sum()),
        "std_error": float(res.std_error[v].mean()) if v.any() else float("nan"),
        "red_chi2": float(res.red_chi2[v].mean()) if v.any() else float("nan"),
    }


def _summary(x: np.ndarray) -> dict:
    if x.size == 0:
        return dict(mean=np.nan, sd=np.nan, median=np.nan, q1=np.nan, q3=np.nan, n=0)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if x.size > 1 else 0.0,
                median=float(med), q1=float(q1), q3=float(q3), n=int(x.size))


def statistics(res: FitResult, cfg: Config, valid=None) -> list[dict]:
    v = valid_mask(res, cfg) if valid is None else valid
    rows = []
    for name in (cfg.ratio_num, cfg.ratio_den):
        pk = res.peak(name)
        for q in QUANTITIES:
            rows.append({"parameter": f"{name} {q.capitalize()}", **_summary(pk[q][v])})
    r = ratios(res, cfg)
    for q in QUANTITIES:
        rows.append({"parameter": f"Ratio {q.capitalize()}", **_summary(r[q][v])})
    return rows


def export_table(res: FitResult, cfg: Config, x=None, y=None) -> str:
    """Tab-separated per-point table in the vendor's column order. With stage
    coordinates given, X and Y columns are appended (the vendor's export has
    none, which leaves its rows unplaceable)."""
    n, d = res.peak(cfg.ratio_num), res.peak(cfg.ratio_den)
    r = ratios(res, cfg)
    cols = [(f"{cfg.ratio_num} {q.capitalize()}", n[q]) for q in QUANTITIES]
    cols += [(f"{cfg.ratio_den} {q.capitalize()}", d[q]) for q in QUANTITIES]
    cols += [(f"Ratio {q.capitalize()}", r[q]) for q in QUANTITIES]
    if x is not None:
        cols += [("X (um)", x), ("Y (um)", y)]
    lines = ["\t" + "\t".join(c for c, _ in cols) + "\t"]
    arr = np.stack([v for _, v in cols], axis=1)
    for i, row in enumerate(arr, start=1):
        lines.append(f"{i}\t" + "\t".join(f"{v:.6g}" for v in row))
    return "\n".join(lines) + "\n"
