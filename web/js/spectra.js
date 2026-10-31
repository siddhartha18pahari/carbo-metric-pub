// Spectrum processing, as LabSpec's Processing menu: backgrounds, smoothing and derivatives,
// normalisation, cosmic rays, crop and resample, spectrum arithmetic. Each operation takes a
// Raman-shift axis nu (ascending) and intensities y and returns new arrays; applyOps runs a
// list of them in order. Pure functions (no DOM), so they run under Node for the tests and in
// the workers. Used by the Toolbox tab and by the Peak fit tab's background.
//
//   modified polynomial  Lieber & Mahadevan-Jansen (2003), improved by Zhao et al. (2007)
//   asymmetric least squares  Eilers & Boelens (2005), the second-difference penalty
//   SNIP  Ryan et al. (1988), Nucl. Instrum. Meth. B 34, 396, on the LLS transform (Morhac 1997)
//   rubber band  the lower convex hull of the spectrum
//   Savitzky-Golay  Savitzky & Golay (1964), the end windows fitted, not padded

import { pinv, pinvSym } from "./engine/core.js";
import { noiseSigma } from "./d3g-calibration.js";

export { noiseSigma };

// --------------------------------------------------------------------------
// backgrounds: each returns the background, y minus it is the corrected spectrum
// --------------------------------------------------------------------------
function vandermonde(nu, order) {
  const N = nu.length, q = order + 1, mid = (nu[0] + nu[N - 1]) / 2, half = (nu[N - 1] - nu[0]) / 2 || 1, A = new Float64Array(N * q);
  for (let n = 0; n < N; n++) { const t = (nu[n] - mid) / half; let v = 1; for (let k = 0; k < q; k++) { A[n * q + k] = v; v *= t; } }
  return { A, q };
}
// least squares polynomial through the points of z that `use` keeps, evaluated at every channel
function polyThrough(A, q, z, use, out) {
  const N = z.length, M = new Float64Array(q * q), b = new Float64Array(q);
  for (let n = 0; n < N; n++) {
    if (!use[n]) continue;
    for (let i = 0; i < q; i++) { const ai = A[n * q + i]; b[i] += ai * z[n]; for (let j = 0; j < q; j++) M[i * q + j] += ai * A[n * q + j]; }
  }
  const c = pinvSym(M, q), coef = new Float64Array(q);
  for (let i = 0; i < q; i++) { let s = 0; for (let j = 0; j < q; j++) s += c[i * q + j] * b[j]; coef[i] = s; }
  for (let n = 0; n < N; n++) { let s = 0; for (let k = 0; k < q; k++) s += coef[k] * A[n * q + k]; out[n] = s; }
  return out;
}
// the improved modified polynomial (Zhao et al. 2007, Appl. Spectrosc. 61, 1225): the polynomial
// is fitted, the bands found on the first pass (more than one residual SD above it) left out, the
// spectrum clipped at the polynomial plus that SD, and again until the SD settles. Clipping at the
// polynomial itself (Lieber's original) sinks the background by about the noise.
export function modpoly(nu, y, order = 3, iters = 100) {
  const N = y.length, { A, q } = vandermonde(nu, order), z = Float64Array.from(y), b = new Float64Array(N), use = new Uint8Array(N).fill(1);
  const devOf = () => { let s = 0, s2 = 0, c = 0; for (let n = 0; n < N; n++) if (use[n]) { const r = z[n] - b[n]; s += r; s2 += r * r; c++; } return Math.sqrt(Math.max(0, s2 / c - (s / c) ** 2)); };
  polyThrough(A, q, z, use, b);
  let dev = devOf();
  for (let n = 0; n < N; n++) if (z[n] > b[n] + dev) use[n] = 0;
  if (use.reduce((t, v) => t + v, 0) < q + 2) use.fill(1);
  for (let it = 0; it < iters; it++) {
    for (let n = 0; n < N; n++) if (z[n] > b[n] + dev) z[n] = b[n] + dev;
    polyThrough(A, q, z, use, b);
    const d = devOf();
    if (Math.abs(d - dev) <= 0.05 * dev || d === 0) { dev = d; break; }
    dev = d;
  }
  return b;
}

// (W + lambda D'D) z = W y with D the second difference: a symmetric band of width 2, solved by
// banded Cholesky; points above the background get weight p, below 1 - p
export function als(y, lambda = 1e5, p = 0.01, iters = 10) {
  const N = y.length;
  if (N < 5) return Float64Array.from(y);
  const w = new Float64Array(N).fill(1), z = new Float64Array(N);
  const d0 = new Float64Array(N), d1 = new Float64Array(N), d2 = new Float64Array(N);   // D'D: diagonal, first and second off-diagonal
  for (let n = 0; n < N; n++) d0[n] = n === 0 || n === N - 1 ? 1 : n === 1 || n === N - 2 ? 5 : 6;
  for (let n = 0; n < N - 1; n++) d1[n] = n === 0 || n === N - 2 ? -2 : -4;
  for (let n = 0; n < N - 2; n++) d2[n] = 1;
  const L0 = new Float64Array(N), L1 = new Float64Array(N), L2 = new Float64Array(N), t = new Float64Array(N);
  for (let it = 0; it < iters; it++) {
    // the band of A = W + lambda D'D, factorised as L L'
    for (let n = 0; n < N; n++) {
      const a0 = w[n] + lambda * d0[n], a1 = n >= 1 ? lambda * d1[n - 1] : 0, a2 = n >= 2 ? lambda * d2[n - 2] : 0;
      L2[n] = n >= 2 ? a2 / L0[n - 2] : 0;
      L1[n] = n >= 1 ? (a1 - (n >= 2 ? L2[n] * L1[n - 1] : 0)) / L0[n - 1] : 0;
      L0[n] = Math.sqrt(Math.max(a0 - L1[n] * L1[n] - L2[n] * L2[n], 1e-300));
    }
    for (let n = 0; n < N; n++) t[n] = (w[n] * y[n] - (n >= 1 ? L1[n] * t[n - 1] : 0) - (n >= 2 ? L2[n] * t[n - 2] : 0)) / L0[n];
    for (let n = N - 1; n >= 0; n--) z[n] = (t[n] - (n + 1 < N ? L1[n + 1] * z[n + 1] : 0) - (n + 2 < N ? L2[n + 2] * z[n + 2] : 0)) / L0[n];
    let changed = false;
    for (let n = 0; n < N; n++) { const v = y[n] > z[n] ? p : 1 - p; if (v !== w[n]) changed = true; w[n] = v; }
    if (!changed && it) break;
  }
  return z;
}

// clipping by the mean of the neighbours m channels away, m = 1 .. halfWidth, on the
// log-log-square-root transform so the background is followed over a wide intensity range
export function snip(y, halfWidth = 40) {
  const N = y.length, off = Math.min(0, ...y), v = Float64Array.from(y, x => Math.log(Math.log(Math.sqrt(x - off + 1) + 1) + 1)), u = new Float64Array(N);
  for (let m = 1; m <= halfWidth; m++) {
    u.set(v);
    for (let n = m; n < N - m; n++) { const a = (u[n - m] + u[n + m]) / 2; if (a < v[n]) v[n] = a; }
  }
  return Float64Array.from(v, x => (Math.exp(Math.exp(x) - 1) - 1) ** 2 - 1 + off);
}

// the lower convex hull of (nu, y), linear between its corners
export function rubberband(nu, y) {
  const N = y.length, hull = [];
  for (let n = 0; n < N; n++) {
    while (hull.length >= 2) {
      const a = hull[hull.length - 2], b = hull[hull.length - 1];
      if ((nu[b] - nu[a]) * (y[n] - y[a]) - (y[b] - y[a]) * (nu[n] - nu[a]) <= 0) hull.pop(); else break;
    }
    hull.push(n);
  }
  const out = new Float64Array(N);
  for (let h = 0; h < hull.length - 1; h++) {
    const a = hull[h], b = hull[h + 1];
    for (let n = a; n <= b; n++) out[n] = y[a] + (y[b] - y[a]) * (nu[n] - nu[a]) / ((nu[b] - nu[a]) || 1);
  }
  if (hull.length === 1) out.fill(y[hull[0]]);
  return out;
}

export const BASELINES = { none: "None", modpoly: "Polynomial (modified)", als: "Asymmetric least squares", snip: "SNIP", rubber: "Rubber band" };
// one of the above by name: { kind, order, lambda, p, width }
export function background(nu, y, b) {
  switch (b.kind) {
    case "modpoly": return modpoly(nu, y, b.order ?? 3);
    case "als": return als(y, b.lambda ?? 1e5, b.p ?? 0.01);
    case "snip": return snip(y, b.width ?? 40);
    case "rubber": return rubberband(nu, y);
    default: return new Float64Array(y.length);
  }
}

// --------------------------------------------------------------------------
// smoothing and derivatives
// --------------------------------------------------------------------------
// Savitzky-Golay: the local polynomial of `order` over `window` channels, evaluated (deriv 0) or
// differentiated (deriv 1, 2) at each channel; near the ends the first or last full window's
// polynomial is used at the channel's own offset. dx: the mean channel step, for derivatives
export function savgol(y, window = 9, order = 2, deriv = 0, dx = 1) {
  const N = y.length, w = Math.min(window | 1, N % 2 ? N : N - 1), h = (w - 1) >> 1, q = Math.min(order, w - 1) + 1;
  if (w < 3 || deriv >= q) return deriv ? new Float64Array(N) : Float64Array.from(y);
  const A = new Float64Array(w * q);
  for (let i = 0; i < w; i++) for (let k = 0; k < q; k++) A[i * q + k] = (i - h) ** k;
  const pA = pinv(A, w, q), out = new Float64Array(N), c = new Float64Array(q);
  const dpow = (k, t) => { let f = 1; for (let j = 0; j < deriv; j++) f *= k - j; return k < deriv ? 0 : f * t ** (k - deriv); };
  for (let n = 0; n < N; n++) {
    const s = Math.max(0, Math.min(N - w, n - h)), t = n - s - h;
    for (let k = 0; k < q; k++) { let v = 0; for (let i = 0; i < w; i++) v += pA[k * w + i] * y[s + i]; c[k] = v; }
    let v = 0; for (let k = 0; k < q; k++) v += c[k] * dpow(k, t);
    out[n] = v / dx ** deriv;
  }
  return out;
}
export function movingAverage(y, window = 5) {
  const N = y.length, h = window >> 1, out = new Float64Array(N);
  for (let n = 0; n < N; n++) { let s = 0, c = 0; for (let i = Math.max(0, n - h); i <= Math.min(N - 1, n + h); i++) { s += y[i]; c++; } out[n] = s / c; }
  return out;
}
export function medianFilter(y, window = 5) {
  const N = y.length, h = window >> 1, out = new Float64Array(N), buf = [];
  for (let n = 0; n < N; n++) {
    buf.length = 0;
    for (let i = Math.max(0, n - h); i <= Math.min(N - 1, n + h); i++) buf.push(y[i]);
    buf.sort((a, b) => a - b);
    out[n] = buf.length % 2 ? buf[buf.length >> 1] : (buf[buf.length / 2 - 1] + buf[buf.length / 2]) / 2;
  }
  return out;
}

// --------------------------------------------------------------------------
// cosmic rays in one spectrum
// --------------------------------------------------------------------------
function runningMedian(y, size) {
  const n = y.length, h = size >> 1, out = new Float64Array(n), w = new Float64Array(size);
  for (let i = 0; i < n; i++) { for (let k = -h; k <= h; k++) w[k + h] = y[Math.min(n - 1, Math.max(0, i + k))]; w.sort(); out[i] = w[h]; }
  return out;
}
// A spike is at most maxw channels wide, stands thr x the noise above the 7-channel running median,
// and carries most of its height above that median (over 60 % of its height above the 31-channel
// one). The top of a real band a few channels wide stands above the 7-channel median too, but by a
// fraction of its height, so it is kept: a spike test on the excess alone clips sharp bands such
// as silicon's 520. Flagged channels take the median of their unflagged neighbours.
export function despikeSpectrum(y0, thr = 8, maxw = 2) {
  const n = y0.length, y = Float64Array.from(y0);
  if (n < 9) return { y, spikes: 0 };
  const sig = noiseSigma(y0) + 1e-9, m7 = runningMedian(y0, 7), m31 = runningMedian(y0, Math.min(31, n % 2 ? n : n - 1)), flag = new Uint8Array(n);
  for (let i = 0; i < n; i++) { const ex = y0[i] - m7[i]; flag[i] = ex > thr * sig && ex > 0.6 * (y0[i] - m31[i]) ? 1 : 0; }
  let spikes = 0;
  for (let i = 0; i < n;) {
    if (!flag[i]) { i++; continue; }
    let j = i; while (j < n && flag[j]) j++;
    if (j - i <= maxw) {
      const nb = [];
      for (let k = Math.max(0, i - 3); k < Math.min(n, j + 3); k++) if (!flag[k]) nb.push(y0[k]);
      if (nb.length) { nb.sort((a, b) => a - b); const v = nb[nb.length >> 1]; for (let k = i; k < j; k++) y[k] = v; spikes++; }
    }
    i = j;
  }
  return { y, spikes };
}

// --------------------------------------------------------------------------
// normalisation
// --------------------------------------------------------------------------
export const NORMS = { max: "Maximum = 1", area: "Area = 1", vector: "Unit length", snv: "Standard normal variate", minmax: "Minimum 0, maximum 1", band: "Band height = 1" };
export function trapz(nu, y) { let s = 0; for (let n = 1; n < y.length; n++) s += (nu[n] - nu[n - 1]) * (y[n] + y[n - 1]) / 2; return s; }
export function normalise(nu, y, mode, { at = 1580, half = 20 } = {}) {
  const N = y.length;
  let a = 0, b = 1;                                   // out = (y - a) / b
  if (mode === "max") { b = Math.max(...y); }
  else if (mode === "area") { b = Math.abs(trapz(nu, y)); }
  else if (mode === "vector") { let s = 0; for (const v of y) s += v * v; b = Math.sqrt(s); }
  else if (mode === "snv") { let m = 0; for (const v of y) m += v / N; let s = 0; for (const v of y) s += (v - m) ** 2; a = m; b = Math.sqrt(s / Math.max(1, N - 1)); }
  else if (mode === "minmax") { a = Math.min(...y); b = Math.max(...y) - a; }
  else if (mode === "band") { let m = -Infinity; for (let n = 0; n < N; n++) if (Math.abs(nu[n] - at) <= half) m = Math.max(m, y[n]); b = m; }
  if (!(Math.abs(b) > 0) || !Number.isFinite(b)) b = 1;
  return Float64Array.from(y, v => (v - a) / b);
}

// --------------------------------------------------------------------------
// axis
// --------------------------------------------------------------------------
export function crop(nu, y, lo, hi) {
  const idx = []; for (let n = 0; n < nu.length; n++) if (nu[n] >= lo && nu[n] <= hi) idx.push(n);
  return { nu: Float64Array.from(idx, n => nu[n]), y: Float64Array.from(idx, n => y[n]) };
}
// linear interpolation onto x (both ascending); outside nu's span, NaN
export function interp(nu, y, x) {
  const out = new Float64Array(x.length), N = nu.length;
  let j = 1;
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    if (v < nu[0] || v > nu[N - 1]) { out[i] = NaN; continue; }
    while (j < N - 1 && nu[j] < v) j++;
    const t = (v - nu[j - 1]) / ((nu[j] - nu[j - 1]) || 1);
    out[i] = y[j - 1] + t * (y[j] - y[j - 1]);
  }
  return out;
}
// an even axis of `step` cm⁻¹ over nu's span
export function evenAxis(nu, step) {
  const a = Math.ceil(nu[0] / step) * step, n = Math.floor((nu[nu.length - 1] - a) / step + 1e-9) + 1;
  return Float64Array.from({ length: Math.max(0, n) }, (_, i) => a + i * step);
}
// Raman shift (cm⁻¹) and absolute wavelength (nm) for a laser line at laserNm
export const shiftToNm = (s, laserNm) => 1 / (1 / laserNm - s * 1e-7);
export const nmToShift = (nm, laserNm) => 1e7 / laserNm - 1e7 / nm;

// --------------------------------------------------------------------------
// a list of operations, in order: what the Toolbox applies to the pinned spectrum (to see it)
// and to every spectrum of the file (to open the result as a new file)
// --------------------------------------------------------------------------
export const OPS = {
  crop: { label: "Crop range", args: { lo: 100, hi: 3500 } },
  despike: { label: "Cosmic rays", args: { thr: 8, width: 2 } },
  baseline: { label: "Background", args: { kind: "modpoly", order: 3, lambda: 1e5, p: 0.01, width: 40 } },
  smooth: { label: "Smoothing", args: { kind: "savgol", window: 9, order: 2 } },
  deriv: { label: "Derivative", args: { order: 1, window: 9, poly: 2 } },
  normalise: { label: "Normalise", args: { mode: "max", at: 1580, half: 20 } },
  resample: { label: "Resample", args: { step: 1 } },
  subtract: { label: "Subtract reference", args: { scale: "fit", factor: 1 } },
  divide: { label: "Divide by reference", args: {} },
  offset: { label: "Add / multiply", args: { add: 0, mul: 1 } },
};

// one operation; ctx.ref = { nu, y }: the reference spectrum for subtract and divide
export function applyOp(nu, y, op, ctx = {}) {
  const a = { ...OPS[op.op].args, ...op.args };
  switch (op.op) {
    case "crop": { const lo = Math.min(a.lo, a.hi), hi = Math.max(a.lo, a.hi); return crop(nu, y, lo, hi); }
    case "despike": return { nu, y: despikeSpectrum(y, a.thr, a.width).y };
    case "baseline": { const b = background(nu, y, a); return { nu, y: Float64Array.from(y, (v, n) => v - b[n]), base: b }; }
    case "smooth": return { nu, y: a.kind === "median" ? medianFilter(y, a.window) : a.kind === "mean" ? movingAverage(y, a.window) : savgol(y, a.window, a.order) };
    case "deriv": { const dx = (nu[nu.length - 1] - nu[0]) / Math.max(1, nu.length - 1); return { nu, y: savgol(y, a.window, Math.max(a.poly, a.order), a.order, dx) }; }
    case "normalise": return { nu, y: normalise(nu, y, a.mode, a) };
    case "resample": { const x = evenAxis(nu, a.step); return { nu: x, y: interp(nu, y, x) }; }
    case "subtract": case "divide": {
      if (!ctx.ref) throw new Error(`${OPS[op.op].label}: choose a reference spectrum first`);
      const r = interp(ctx.ref.nu, ctx.ref.y, nu);
      if (op.op === "divide") return { nu, y: Float64Array.from(y, (v, n) => (Math.abs(r[n]) > 0 ? v / r[n] : NaN)) };
      // scaled to remove as much of the reference as least squares allows, or by a fixed factor
      let k = a.factor;
      if (a.scale === "fit") { let sy = 0, sr = 0; for (let n = 0; n < y.length; n++) if (Number.isFinite(r[n])) { sy += y[n] * r[n]; sr += r[n] * r[n]; } k = sr > 0 ? sy / sr : 0; }
      return { nu, y: Float64Array.from(y, (v, n) => v - k * r[n]), k };
    }
    case "offset": return { nu, y: Float64Array.from(y, v => v * a.mul + a.add) };
    default: throw new Error(`unknown operation ${op.op}`);
  }
}
export function applyOps(nu, y, ops, ctx = {}) {
  let cur = { nu: Float64Array.from(nu), y: Float64Array.from(y) };
  for (const op of ops) if (op.on !== false) cur = { ...cur, ...applyOp(cur.nu, cur.y, op, ctx) };
  return cur;
}
