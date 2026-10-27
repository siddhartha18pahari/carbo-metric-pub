"""Spectral 3-D Gaussian splatting of a fitted map.

Reuses the differentiable Gaussian field of ``ramansp.splatting``
(github.com/siddhartha18pahari/RAMANSP: analytic gradients, prune/clone densification,
truncated 3-sigma evaluation, regression-tested). What changes here:

* the target is the *baseline-corrected* cube, using each pixel's own fitted
  baseline, so the ellipsoids describe the carbon bands and not the
  fluorescence background;
* the spatial mask is log(1 + fitted G intensity) above an Otsu threshold
  (the particles), so no primitives are spent on empty substrate;
* the optimisation loop reports progress and hands back snapshots, so the
  viewer can show the field converging;
* the field is exported in instrument units (um, um, cm-1) for the browser.

Held-out channels (every 5th) are never fitted; the reported
``resid_over_noise`` compares the error on them with the map's own noise
(second differences along the spectrum). Near 1 means the field is at the
noise floor. PSNR is reported too but confounds the representation with the
acquisition's signal-to-noise.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from .fitter import FitResult, baseline_basis
from .labspec import RamanMap

_HERE = os.path.dirname(os.path.abspath(__file__))
_RAMANSP = os.path.normpath(os.path.join(_HERE, "..", "..", "CLAUDE RAMAN SP"))


def _ramansp():
    try:
        import ramansp  # noqa: F401
    except ImportError:
        if os.path.isdir(_RAMANSP) and _RAMANSP not in sys.path:
            sys.path.insert(0, _RAMANSP)
        import ramansp  # noqa: F401
    from ramansp.containers import SpectralImage
    from ramansp.splatting.fit import SplatConfig, _Adam, reconstruction_report
    from ramansp.splatting.model import GaussianField
    from ramansp.splatting.volume import build_volume
    return SpectralImage, SplatConfig, _Adam, reconstruction_report, GaussianField, build_volume


def available() -> tuple[bool, str]:
    try:
        _ramansp()
        return True, ""
    except Exception as e:  # pragma: no cover - environment dependent
        return False, f"ramansp not importable ({e}); expected at {_RAMANSP}"


def otsu(v: np.ndarray, bins: int = 256) -> float:
    v = v[np.isfinite(v)]
    if v.size == 0 or v.min() == v.max():
        return float(v.min()) if v.size else 0.0
    h, e = np.histogram(v, bins=bins)
    c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def corrected_cube(rmap: RamanMap, fit: FitResult) -> tuple[np.ndarray, np.ndarray]:
    """Cube inside the fit range with each pixel's fitted baseline removed."""
    k = (rmap.wn >= fit.range_lo) & (rmap.wn <= fit.range_hi)
    wn = rmap.wn[k]
    ny, nx, _ = rmap.cube.shape
    Y = rmap.cube[:, :, k].reshape(ny * nx, -1).astype(np.float64)
    B = baseline_basis(wn, fit.range_lo, fit.range_hi, fit.baseline.shape[1] - 1)
    if B.shape[1]:
        Y = Y - fit.baseline @ B.T
    return Y.reshape(ny, nx, -1), wn


def export_field(field, vol, rmap: RamanMap) -> dict:
    """Ellipsoids in instrument units, ready for the WebGL viewer."""
    i0, i1, j0, j1 = vol.bbox
    xs, ys = rmap.x[j0:j1], rmap.y[i0:i1]
    sx = (xs[-1] - xs[0]) if xs.size > 1 else 1.0
    sy = (ys[-1] - ys[0]) if ys.size > 1 else 1.0
    sz = vol.wn1 - vol.wn0
    S = np.diag([sx, sy, sz])
    mu = field.mu * np.array([sx, sy, sz]) + np.array([xs[0], ys[0], vol.wn0])
    cov = np.einsum("ij,njk,kl->nil", S, field.covariances(), S)
    amp = field.amplitude * vol.scale
    order = np.argsort(-amp)
    return {
        "n": int(field.n),
        "mu": mu[order].round(4).tolist(),
        "cov": cov[order].reshape(-1, 9).round(6).tolist(),
        "amp": amp[order].round(4).tolist(),
        "wn_range": [float(vol.wn0), float(vol.wn1)],
        "bbox_um": [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
    }


def run(rmap: RamanMap, fit: FitResult, g_name: str = "G", n_gaussians: int = 1400,
        iters: int = 150, channel_bin: int = 2, seed: int = 0, progress=None,
        snapshot_every: int = 15) -> dict:
    SpectralImage, SplatConfig, Adam, reconstruction_report, GaussianField, build_volume = _ramansp()
    t0 = time.time()
    cube, wn = corrected_cube(rmap, fit)
    ny, nx, _ = cube.shape
    # Otsu on log(1 + I_G): the zeros of empty substrate pixels stay in one
    # bin instead of dragging the split (log10 of clipped zeros put 80 % of
    # one test map "on the particle"; this gives the 38 % the image shows)
    g = np.log1p(fit.peak(g_name)["intensity"].reshape(ny, nx))
    mask = g > otsu(g)
    if mask.mean() < 0.01 or mask.mean() > 0.98:
        mask = None
    image = SpectralImage(cube, wn, {"source": rmap.source})
    cfg = SplatConfig(n_gaussians=n_gaussians, iters=iters, channel_bin=channel_bin,
                      seed=seed, use_particle_mask=False, log_every=0)
    vol = build_volume(image, spatial_mask=mask, hold_out=cfg.hold_out, channel_bin=cfg.channel_bin)
    field = GaussianField.from_volume(vol, min(n_gaussians, cfg.max_gaussians), seed=seed,
                                      lam_a=cfg.lam_a, lam_s=cfg.lam_s, trunc=cfg.trunc)

    xa, ya, za = field.world_axes
    spacing = np.array([float(np.median(np.diff(a))) if a.size > 1 else 1.0 for a in (xa, ya, za)])
    lo_s = np.log(cfg.sigma_vox_min * spacing)[None, :]
    hi_s = np.log(cfg.sigma_vox_max * spacing)[None, :]
    np.clip(field.log_s, lo_s, hi_s, out=field.log_s)
    opt = Adam()
    stop_densify = int(cfg.densify_until * iters)
    history = []
    for it in range(1, iters + 1):
        loss, grad, aux = field.loss_and_grad(vol)
        field.mu = opt.step("mu", field.mu, grad.mu, cfg.lr_mu)
        field.log_s = opt.step("log_s", field.log_s, grad.log_s, cfg.lr_log_s)
        field.quat = opt.step("quat", field.quat, grad.quat, cfg.lr_quat)
        field.log_a = opt.step("log_a", field.log_a, grad.log_a, cfg.lr_log_a)
        np.clip(field.mu, -0.05, 1.05, out=field.mu)
        np.clip(field.log_s, lo_s, hi_s, out=field.log_s)
        if it % cfg.densify_every == 0 and it <= stop_densify:
            a_med = float(np.median(field.amplitude))
            field.densify(aux["pos_grad"], cfg.max_gaussians, seed=seed + it)
            field.prune(cfg.prune_a_frac * a_med)
            opt.m.clear(); opt.v.clear()
        history.append({"iter": it, "loss": float(loss), "n": int(field.n)})
        if progress:
            snap = export_field(field, vol, rmap) if (it == 1 or it % snapshot_every == 0) else None
            progress(it / iters, history[-1], snap)

    report = reconstruction_report(field, vol)
    out = export_field(field, vol, rmap)
    out.update({
        "history": history,
        "report": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                   for k, v in report.items()},
        "mask_fraction": float(mask.mean()) if mask is not None else 1.0,
        "voxels": int(vol.mask3d.sum()),
        "seconds": time.time() - t0,
        "channel_bin": channel_bin,
    })
    return out
