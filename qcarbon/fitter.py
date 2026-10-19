"""Batch peak fitting for a whole map at once.

Model per spectrum, over the configured spectral range:

    f(nu) = sum_k b_k t^k  +  sum_j A_j * phi_j(nu; c_j, w_j),   t = (nu - mid) / half

with ``phi`` a unit-height Lorentzian or Gaussian of FWHM ``w``. Every pixel
gets its own bound-constrained Levenberg-Marquardt fit (amplitude >= 0, centre
and width inside the peak table's limits), compiled with numba and run in
parallel across pixels. Parameters sitting on a bound with the gradient
pointing outward are held for that step (projected Newton), so a band that is
absent converges to amplitude 0 instead of pushing against the bound.

Starting point: centres and widths from the peak table, amplitudes and
baseline from one shared linear least-squares solve (the design matrix is the
same for every pixel at the starting shapes).

Shapes from the mean spectrum (``Config.auto_shapes``, on by default): the
mean spectrum of the carbon-bearing pixels is fitted first, and its centres
and widths become every pixel's starting shapes. A band whose centre or width
a single pixel cannot determine (standard error at the map's own noise level
above a quarter of its width, or half its width for the centre) keeps the
mean shape and only its amplitude is fitted per pixel. Without this a broad,
weak band such as D3 wanders to its bounds and soaks up background curvature
the polynomial cannot follow, so its map shows the baseline, not the carbon.

Reported per pixel, as on the instrument: peak intensity (height), FWHM and
area. Area is the line shape integrated over the spectral range, not to
infinity: a 20 cm-1 Lorentzian keeps ~2 % of its area outside 1000-1800 cm-1,
and the vendor's areas carry the same window (checked against its exports).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numba as nb
import numpy as np
from scipy.special import erf

from .config import Config

G4 = 4.0 * np.log(2.0)
LORENTZ, GAUSS = 0, 1


# --------------------------------------------------------------------------
# line shapes (numpy, for plotting and areas)
# --------------------------------------------------------------------------
def shape_eval(nu, c, w, kind: str):
    """Unit-height line shape."""
    if kind == "Lorentz":
        u = 2.0 * (nu - c) / w
        return 1.0 / (1.0 + u * u)
    v = (nu - c) / w
    return np.exp(-G4 * v * v)


def windowed_area(A, c, w, kind: str, lo: float, hi: float):
    """Integral of ``A * phi`` over ``[lo, hi]``."""
    if kind == "Lorentz":
        return A * (w / 2.0) * (np.arctan(2.0 * (hi - c) / w) - np.arctan(2.0 * (lo - c) / w))
    s = w / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return A * s * np.sqrt(np.pi / 2.0) * (erf((hi - c) / (s * np.sqrt(2.0)))
                                           - erf((lo - c) / (s * np.sqrt(2.0))))


def baseline_basis(nu, lo, hi, order):
    if order < 0:
        return np.zeros((np.size(nu), 0))
    t = (np.asarray(nu, float) - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
    return np.stack([t ** k for k in range(order + 1)], axis=1)


def modpoly_baseline(Y, B, n_iter=60):
    """Iterative modified polynomial baseline (Lieber & Mahadevan-Jansen 2003):
    fit, clip the spectrum down to the fit, refit. Used for 'fit separately'."""
    pinv = np.linalg.pinv(B)
    Z = Y.copy()
    coef = Z @ pinv.T
    for _ in range(n_iter):
        Z = np.minimum(Z, coef @ B.T)
        coef = Z @ pinv.T
    return coef


# --------------------------------------------------------------------------
# compiled core
# --------------------------------------------------------------------------
@nb.njit(cache=True, fastmath=False)
def _eval(nu, B, th, nbase, kinds, J, want_jac):
    """Model at ``th``; fills the full Jacobian ``J`` (N, M) when asked."""
    N = nu.size
    f = np.zeros(N)
    for k in range(nbase):
        for n in range(N):
            f[n] += th[k] * B[n, k]
            if want_jac:
                J[n, k] = B[n, k]
    for j in range(kinds.size):
        i = nbase + 3 * j
        A, c, w = th[i], th[i + 1], th[i + 2]
        for n in range(N):
            if kinds[j] == 0:
                u = 2.0 * (nu[n] - c) / w
                d = 1.0 + u * u
                phi = 1.0 / d
                if want_jac:
                    dphi_du = -2.0 * u / (d * d)
                    J[n, i] = phi
                    J[n, i + 1] = A * dphi_du * (-2.0 / w)
                    J[n, i + 2] = A * dphi_du * (-u / w)
            else:
                v = (nu[n] - c) / w
                phi = np.exp(-2.772588722239781 * v * v)
                if want_jac:
                    dphi_dv = -2.0 * 2.772588722239781 * v * phi
                    J[n, i] = phi
                    J[n, i + 1] = A * dphi_dv * (-1.0 / w)
                    J[n, i + 2] = A * dphi_dv * (-v / w)
            f[n] += A * phi
    return f


@nb.njit(cache=True)
def _lm_one(nu, y, B, th, lo, hi, free, nbase, kinds, max_iter, tol):
    N = nu.size
    M = th.size
    J = np.empty((N, M))
    f = _eval(nu, B, th, nbase, kinds, J, False)
    cost = 0.0
    for n in range(N):
        cost += (y[n] - f[n]) ** 2
    lam = 1e-3
    rel = 1.0
    it = 0
    while it < max_iter:
        it += 1
        f = _eval(nu, B, th, nbase, kinds, J, True)
        r = y - f
        g = J.T @ r
        H = J.T @ J
        # hold parameters that are fixed, or on a bound and pushed outward
        act = free.copy()
        for m in range(M):
            if act[m]:
                if th[m] <= lo[m] and g[m] <= 0.0:
                    act[m] = False
                elif th[m] >= hi[m] and g[m] >= 0.0:
                    act[m] = False
        idx = np.where(act)[0]
        if idx.size == 0:
            break
        Ha = H[idx][:, idx]
        ga = g[idx]
        dmax = 0.0
        for a in range(idx.size):
            dmax = max(dmax, Ha[a, a])
        accepted = False
        for _try in range(12):
            Hd = Ha.copy()
            for a in range(idx.size):
                Hd[a, a] += lam * Ha[a, a] + 1e-10 * dmax + 1e-12
            step = np.linalg.solve(Hd, ga)
            tn = th.copy()
            for a in range(idx.size):
                m = idx[a]
                v = th[m] + step[a]
                if v < lo[m]:
                    v = lo[m]
                elif v > hi[m]:
                    v = hi[m]
                tn[m] = v
            fn = _eval(nu, B, tn, nbase, kinds, J, False)
            cn = 0.0
            for n in range(N):
                cn += (y[n] - fn[n]) ** 2
            if cn < cost:
                rel = (cost - cn) / max(cost, 1e-300)
                th[:] = tn
                cost = cn
                lam = max(lam / 3.0, 1e-9)
                accepted = True
                break
            lam *= 4.0
        if not accepted or rel < tol:
            break
    return it


@nb.njit(parallel=True, cache=True)
def _lm_batch(nu, Y, B, TH, lo, hi, free, nbase, kinds, max_iter, tol, iters):
    for p in nb.prange(Y.shape[0]):
        iters[p] = _lm_one(nu, Y[p], B, TH[p], lo, hi, free, nbase, kinds, max_iter, tol)


@nb.njit(parallel=True, cache=True)
def _residual_stats(nu, Y, B, TH, nbase, kinds, base_fixed, Bfix, ssr, chi):
    N = nu.size
    J = np.empty((1, 1))
    for p in nb.prange(Y.shape[0]):
        f = _eval(nu, B, TH[p], nbase, kinds, J, False)
        s = 0.0
        c = 0.0
        for n in range(N):
            fv = f[n]
            yv = Y[p, n]
            if base_fixed.shape[0] > 0:
                b = 0.0
                for k in range(Bfix.shape[1]):
                    b += base_fixed[p, k] * Bfix[n, k]
                fv += b
                yv += b
            d = yv - fv
            s += d * d
            c += d * d / max(fv, 1.0)
        ssr[p] = s
        chi[p] = c


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------
@dataclass
class FitResult:
    names: list[str]
    shapes: list[str]
    intensity: np.ndarray      # (P, m)
    position: np.ndarray
    width: np.ndarray
    area: np.ndarray
    baseline: np.ndarray       # (P, nb) coefficients in t
    std_error: np.ndarray      # (P,) sqrt(SSR / (N - 3 * n_peaks))
    red_chi2: np.ndarray       # (P,) Poisson-weighted reduced chi-square
    n_iter: np.ndarray         # (P,)
    n_channels: int
    n_free: int
    range_lo: float
    range_hi: float
    seconds: float
    held: list[str] = None     # bands whose shape was held at the mean-spectrum value

    def peak(self, name: str) -> dict:
        j = self.names.index(name)
        return {"intensity": self.intensity[:, j], "position": self.position[:, j],
                "width": self.width[:, j], "area": self.area[:, j]}


# --------------------------------------------------------------------------
# optional smoothing before the fit
# --------------------------------------------------------------------------
def _sg_hat(window: int, order: int) -> np.ndarray:
    """(window, window) projection onto polynomials of ``order``: row h is the
    Savitzky-Golay smoothing filter, the rows either side evaluate the
    polynomial fitted to an end window (scipy's mode='interp')."""
    h = window // 2
    A = np.vander(np.arange(-h, h + 1, dtype=float), order + 1, increasing=True)
    return A @ np.linalg.pinv(A)


def smooth_spectra(cube: np.ndarray, cfg: Config) -> np.ndarray:
    """The spectra the fit sees, as float64 (ny, nx, K).

    ``cfg.smooth_window`` > 0: Savitzky-Golay along Raman shift (window and
    ``smooth_order``). ``cfg.spatial_kernel`` 3/5/7: each spectrum averaged with
    its neighbours, Gaussian weights with sigma = kernel / 4, renormalised at the
    map edges. Both are off by default (Q-Carbon fits the raw spectra). Spatial
    averaging trades resolution for noise, and neighbouring pixels then share
    data, so per-pixel statistics overstate the number of independent points.
    """
    Y = np.asarray(cube, float)
    w, K = cfg.smooth_window, Y.shape[-1]
    if w and K >= w:
        H, h = _sg_hat(w, cfg.smooth_order), w // 2
        out = np.empty_like(Y)
        out[..., h:K - h] = np.lib.stride_tricks.sliding_window_view(Y, w, axis=-1) @ H[h]
        out[..., :h] = Y[..., :w] @ H[:h].T
        out[..., K - h:] = Y[..., K - w:] @ H[h + 1:].T
        Y = out
    k = cfg.spatial_kernel
    if k:
        # Gaussian weights over the measured neighbours only: weights and
        # weighted sums are convolved separately and divided at the end, so a
        # missing point or the map edge simply drops out of the average
        r = k // 2
        g = np.exp(-0.5 * (np.arange(-r, r + 1) / (k / 4.0)) ** 2)
        ok = np.isfinite(Y).all(axis=-1)
        num, den = np.where(ok[..., None], Y, 0.0), ok.astype(float)
        for axis in (0, 1):
            n = num.shape[axis]
            nn, dd = np.zeros_like(num), np.zeros_like(den)
            for d, wgt in zip(range(-r, r + 1), g):
                t0, t1 = max(0, -d), min(n, n - d)
                if t1 <= t0:
                    continue
                tgt = [slice(None)] * 2; src = [slice(None)] * 2
                tgt[axis], src[axis] = slice(t0, t1), slice(t0 + d, t1 + d)
                nn[tuple(tgt)] += wgt * num[tuple(src)]
                dd[tuple(tgt)] += wgt * den[tuple(src)]
            num, den = nn, dd
        Y = np.where(ok[..., None], num / np.maximum(den, 1e-300)[..., None], np.nan)
    return Y


# --------------------------------------------------------------------------
# shapes from the mean spectrum
# --------------------------------------------------------------------------
def _otsu(v, n_bins=128):
    h, e = np.histogram(v, n_bins)
    p = h / max(h.sum(), 1)
    c = 0.5 * (e[1:] + e[:-1])
    w, mu = np.cumsum(p), np.cumsum(p * c)
    return float(c[np.argmax((mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12))])


def _mean_shapes(nu, Y, B, TH, lo, hi, free, nbase, kinds, names, max_iter):
    """Fit the mean spectrum of the carbon pixels (Otsu split of the summed
    starting amplitudes) and decide, band by band, whether one pixel could
    determine its shape. Returns the starting parameters, the free mask and
    the names of the bands whose shape is held."""
    signal = np.log1p(TH[:, nbase::3].sum(1))
    part = signal > _otsu(signal) if np.ptp(signal) > 0 else np.ones(signal.size, bool)
    if part.sum() < 2:
        part[:] = True
    ym = np.ascontiguousarray(Y[part].mean(0))
    th = np.ascontiguousarray(TH[part].mean(0))
    _lm_one(nu, ym, B, th, lo, hi, free, nbase, kinds, max_iter, 1e-9)

    # standard errors a single pixel would have, at the map's own noise level
    # (second differences along the spectrum: model-free)
    sigma = np.median(np.abs(np.diff(Y[part], 2, axis=1))) / (0.6745 * np.sqrt(6.0))
    J = np.empty((nu.size, th.size))
    _eval(nu, B, th, nbase, kinds, J, True)
    se = np.zeros(th.size)
    Jf = J[:, free]
    se[free] = sigma * np.sqrt(np.clip(np.diag(np.linalg.pinv(Jf.T @ Jf)), 0.0, None))

    free = free.copy()
    held = []
    for j, name in enumerate(names):
        i = nbase + 3 * j
        if not free[i + 1]:
            continue            # held by the user: keeps the table's shape
        A, w = th[i], th[i + 2]
        if A <= 1e-9 or se[i + 2] > 0.25 * w or se[i + 1] > 0.5 * w:
            free[i + 1] = free[i + 2] = False
            held.append(name)
    return th, free, held


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------
def fit_spectra(Y: np.ndarray, wn: np.ndarray, cfg: Config, progress=None,
                chunk: int = 2048) -> FitResult:
    """Fit every row of ``Y`` (P, K) on axis ``wn`` (K,). ``progress(frac)``
    is called between chunks."""
    cfg.check()
    t0 = time.time()
    k = (wn >= cfg.range_lo) & (wn <= cfg.range_hi)
    if k.sum() < 10:
        raise ValueError("fewer than 10 channels inside the spectral range")
    # points never measured (a map stopped early) are NaN: fit the rest and
    # report NaN for them, so no average inside the fit ever sees a gap
    measured = np.isfinite(np.asarray(Y)[:, k]).all(axis=1)
    if not measured.all():
        if not measured.any():
            raise ValueError("no measured spectra in this map")
        return _expand(fit_spectra(np.asarray(Y)[measured], wn, cfg, progress, chunk), measured)
    nu = np.ascontiguousarray(wn[k], np.float64)
    Yr = np.ascontiguousarray(np.asarray(Y, np.float64)[:, k])
    P, N = Yr.shape
    shapes = [p.shape for p in cfg.peaks]
    kinds = np.array([LORENTZ if s == "Lorentz" else GAUSS for s in shapes], np.int64)
    m = len(shapes)
    nbl = cfg.baseline_order + 1
    B = baseline_basis(nu, cfg.range_lo, cfg.range_hi, cfg.baseline_order)

    base_fixed = np.zeros((0, 0))
    if cfg.fit_separately and nbl:
        base_fixed = modpoly_baseline(Yr, B)
        Yr = np.ascontiguousarray(Yr - base_fixed @ B.T)
    joint_nb = 0 if base_fixed.shape[0] else nbl
    Bj = np.ascontiguousarray(B if joint_nb else np.zeros((N, 0)))

    M = joint_nb + 3 * m
    lo = np.full(M, -np.inf); hi = np.full(M, np.inf)
    free = np.ones(M, np.bool_)
    theta0 = np.zeros(M)
    for j, p in enumerate(cfg.peaks):
        i = joint_nb + 3 * j
        lo[i] = 0.0
        lo[i + 1], hi[i + 1] = p.pos_min, p.pos_max
        lo[i + 2], hi[i + 2] = p.width_min, p.width_max
        theta0[i + 1], theta0[i + 2] = p.pos, p.width
        if not p.fit or cfg.skip_fitting:
            free[i + 1] = free[i + 2] = False

    # linear start at the given shapes: one design matrix for every pixel
    def linear_start(th0):
        Phi = np.stack([shape_eval(nu, th0[joint_nb + 3 * j + 1], th0[joint_nb + 3 * j + 2], s)
                        for j, s in enumerate(shapes)], axis=1)
        coef = Yr @ np.linalg.pinv(np.hstack([Bj, Phi])).T
        amps = np.clip(coef[:, joint_nb:], 0.0, None)
        th = np.tile(th0, (P, 1))
        if joint_nb:
            th[:, :joint_nb] = (Yr - amps @ Phi.T) @ np.linalg.pinv(Bj).T
        th[:, joint_nb::3] = amps
        return np.ascontiguousarray(th)

    max_iter = min(cfg.max_iter, 20) if cfg.skip_fitting else cfg.max_iter
    TH = linear_start(theta0)
    held: list[str] = []
    if cfg.auto_shapes and not cfg.skip_fitting and P > 1:
        theta0, free, held = _mean_shapes(nu, Yr, Bj, TH, lo, hi, free, joint_nb, kinds,
                                          [p.name for p in cfg.peaks], max_iter)
        TH = linear_start(theta0)

    iters = np.zeros(P, np.int64)
    for c0 in range(0, P, chunk):
        sl = slice(c0, min(c0 + chunk, P))
        sub = np.ascontiguousarray(TH[sl])
        it = np.zeros(sub.shape[0], np.int64)
        _lm_batch(nu, np.ascontiguousarray(Yr[sl]), Bj, sub, lo, hi, free, joint_nb, kinds,
                  max_iter, 1e-7, it)
        TH[sl] = sub
        iters[sl] = it
        if progress:
            progress(sl.stop / P)

    ssr = np.zeros(P); chi = np.zeros(P)
    _residual_stats(nu, Yr, Bj, TH, joint_nb, kinds, base_fixed, np.ascontiguousarray(B), ssr, chi)
    n_free = int(free.sum()) + (nbl if base_fixed.shape[0] else 0)
    # Degrees of freedom as the instrument counts them: channels minus the
    # peak parameters (3 per band), baseline terms not subtracted. Inferred
    # from a vendor per-pixel error map, where this is the only
    # convention under which pixels with the same solution agree.
    dof = max(N - 3 * m, 1)

    A = TH[:, joint_nb::3]
    C = TH[:, joint_nb + 1::3]
    W = TH[:, joint_nb + 2::3]
    area = np.stack([windowed_area(A[:, j], C[:, j], W[:, j], shapes[j], cfg.range_lo, cfg.range_hi)
                     for j in range(m)], axis=1)
    base = base_fixed if base_fixed.shape[0] else TH[:, :joint_nb]
    return FitResult(
        names=[p.name for p in cfg.peaks], shapes=shapes,
        intensity=A.copy(), position=C.copy(), width=W.copy(), area=area,
        baseline=np.ascontiguousarray(base),
        std_error=np.sqrt(ssr / dof), red_chi2=chi / dof, n_iter=iters,
        n_channels=N, n_free=n_free, range_lo=cfg.range_lo, range_hi=cfg.range_hi,
        seconds=time.time() - t0, held=held,
    )


def _expand(r: FitResult, measured: np.ndarray) -> FitResult:
    """Results for the measured points, placed back among NaN rows."""
    def full(a):
        out = np.full((measured.size,) + a.shape[1:], np.nan)
        out[measured] = a
        return out
    for f in ("intensity", "position", "width", "area", "baseline", "std_error", "red_chi2"):
        setattr(r, f, full(getattr(r, f)))
    it = np.zeros(measured.size, np.int64); it[measured] = r.n_iter; r.n_iter = it
    return r


def evaluate(wn, result: FitResult, idx: int) -> dict:
    """Baseline, each component and the total fit for one pixel (for plots)."""
    nu = np.asarray(wn, float)
    B = baseline_basis(nu, result.range_lo, result.range_hi, result.baseline.shape[1] - 1)
    base = B @ result.baseline[idx] if B.shape[1] else np.zeros_like(nu)
    comps = [result.intensity[idx, j] * shape_eval(nu, result.position[idx, j], result.width[idx, j], s)
             for j, s in enumerate(result.shapes)]
    return {"baseline": base, "components": comps, "total": base + np.sum(comps, axis=0)}
