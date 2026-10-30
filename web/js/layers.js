// Layers: the materials of a multilayer sample and the thickness of each layer, from a line scan
// or map across a cut edge, or a confocal depth profile into the film. The table is the one
// HORIBA's LabSpec 6 Layers app gives (number, from, to, thickness, material, spectrum, comment);
// the method is the published one, as the app's own is not.
//
// Each spectrum is cut to the analysis range, cleared of cosmic rays (against its neighbours),
// binned to at most 300 channels and cleared of a polynomial background by modified polyfit
// (Lieber and Mahadevan-Jansen 2003). Its materials come from one of two routes:
//  * a decomposition: the purest spectra, picked one after another by the successive projections
//    algorithm (Araujo et al. 2001) on the residuals over each spectrum's noise, until no more
//    than a handful of spectra have over twice their noise left unexplained; then refined by
//    non-negative matrix factorisation. Each is named after the closest reference loaded, if any
//    (HQI = 100 r²);
//  * reference spectra: every spectrum unmixed against them by non-negative least squares.
// Each material's amount is scaled to its plateau (1 = the pure material). Along each line across
// the layers every point goes to the material with the largest scaled amount, or to none where all
// are below one half, and runs of points form the layers. A boundary sits where the two materials'
// scaled amounts cross: for a step blurred by the laser spot both are at one half there, whatever
// their Raman cross-sections. An outer edge sits where the outermost layer's amount falls to one
// half. On a map every line gives its own stack: the stack most lines share is the sample's, its
// thicknesses are the mean over those lines, and the drift of the boundaries from line to line is
// the tilt of the layers in the map, taken out of the thickness.

import { nmf, nmfProject, wilson, quantileSorted } from "./pf-analysis.js";
import { noiseSigma } from "./d3g-calibration.js";

export const LAYER_DEFAULTS = {
  range: [150, 3600],     // cm⁻¹, cut to what was measured
  baseline: 3,            // degree of the polynomial background taken off each spectrum (0: none)
  kind: "section",        // "section": across a cut edge; "depth": a confocal depth profile
  axis: "auto",           // the direction across the layers on a map: "auto", "x" or "y"
  depthScale: 1,          // depth profiles: true depth over stage travel
  k: 0,                   // materials: 0 finds how many, else that many
  kMax: 8,
  stop: 2,                // the decomposition stops when no more than a handful of spectra have this × their noise left
  minPts: 2,              // a layer inside the stack spans at least this many points along a line
  tilt: true,             // take the tilt of the layers in a map out of their thickness
  useRefs: false,         // unmix against the reference spectra instead of decomposing
  minHQI: 80,             // a material takes a reference's name at this match or above
};
const MAX_BINS = 300, LEARN = 4000;

const median = a => quantileSorted(Float64Array.from(a).filter(Number.isFinite).sort(), 0.5);
function stats(v) {
  const a = Float64Array.from(v).filter(Number.isFinite).sort(), n = a.length;
  if (!n) return { n: 0, mean: NaN, sd: NaN, min: NaN, max: NaN, median: NaN };
  const mean = a.reduce((t, x) => t + x, 0) / n, sd = n > 1 ? Math.sqrt(a.reduce((t, x) => t + (x - mean) ** 2, 0) / (n - 1)) : NaN;
  return { n, mean, sd, min: a[0], max: a[n - 1], median: quantileSorted(a, 0.5) };
}

// --------------------------------------------------------------------------
// the spectra
// --------------------------------------------------------------------------
// modified polyfit: a polynomial fitted again and again to the spectrum clipped at the previous
// fit, so that it settles under the bands. Clipping also pulls it to the bottom of the noise, which
// would leave every spectrum a constant offset of a couple of noise SDs that no material accounts
// for; given the noise sd, it is raised by the median of what is left over the band-free channels
// (left over by less than 3 sd), as the improved modpoly does (Zhao et al. 2007). Returns a
// function of one spectrum on this axis (and its noise).
export function baselineFitter(nu, degree, iters = 30) {
  const n = nu.length, m = degree + 1;
  if (!(degree > 0)) return () => new Float64Array(n);
  const lo = nu[0], hi = nu[n - 1], X = new Float64Array(n * m);
  for (let i = 0; i < n; i++) { const t = 2 * (nu[i] - lo) / (hi - lo) - 1; let v = 1; for (let a = 0; a < m; a++) { X[i * m + a] = v; v *= t; } }
  // (XᵀX)⁻¹ Xᵀ once, by Gauss-Jordan: every spectrum shares the axis
  const A = new Float64Array(m * 2 * m);
  for (let a = 0; a < m; a++) {
    for (let b = 0; b < m; b++) { let t = 0; for (let i = 0; i < n; i++) t += X[i * m + a] * X[i * m + b]; A[a * 2 * m + b] = t; }
    A[a * 2 * m + m + a] = 1;
  }
  for (let c = 0; c < m; c++) {
    let p = c; for (let r = c + 1; r < m; r++) if (Math.abs(A[r * 2 * m + c]) > Math.abs(A[p * 2 * m + c])) p = r;
    for (let j = 0; j < 2 * m; j++) [A[c * 2 * m + j], A[p * 2 * m + j]] = [A[p * 2 * m + j], A[c * 2 * m + j]];
    const d = A[c * 2 * m + c];
    for (let j = 0; j < 2 * m; j++) A[c * 2 * m + j] /= d;
    for (let r = 0; r < m; r++) if (r !== c) { const f = A[r * 2 * m + c]; for (let j = 0; j < 2 * m; j++) A[r * 2 * m + j] -= f * A[c * 2 * m + j]; }
  }
  const Pm = new Float64Array(m * n);
  for (let a = 0; a < m; a++) for (let i = 0; i < n; i++) { let t = 0; for (let b = 0; b < m; b++) t += A[a * 2 * m + m + b] * X[i * m + b]; Pm[a * n + i] = t; }
  return (y0, sd = 0) => {
    const y = Float64Array.from(y0), c = new Float64Array(m), base = new Float64Array(n);
    let scale = 0; for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(y[i]));
    for (let it = 0; it < iters; it++) {
      for (let a = 0; a < m; a++) { let t = 0; for (let i = 0; i < n; i++) t += Pm[a * n + i] * y[i]; c[a] = t; }
      let moved = 0;
      for (let i = 0; i < n; i++) {
        let b = 0; for (let a = 0; a < m; a++) b += X[i * m + a] * c[a];
        base[i] = b;
        if (y[i] > b) { moved = Math.max(moved, y[i] - b); y[i] = b; }
      }
      if (moved <= 1e-4 * scale) break;
    }
    if (sd > 0) {
      const r = []; for (let i = 0; i < n; i++) { const v = y0[i] - base[i]; if (v < 3 * sd) r.push(v); }
      const off = median(r);
      if (Number.isFinite(off)) for (let i = 0; i < n; i++) base[i] += off;
    }
    return base;
  };
}

// The measured spectra on a grid of nx × ny points, cut to the range, cleared of cosmic rays,
// binned (w channels a bin) and cleared of their background. V holds them row by row, base their
// backgrounds, noise each one's noise per bin; rowOf maps a point to its row (-1: not measured).
//
// Cosmic rays are found against the neighbouring spectra, not along the spectrum: a polymer's
// sharp bands are as narrow as a cosmic ray at this sampling, and a running median along the
// spectrum clips their tops (differently in every spectrum). A channel counts as a cosmic ray when
// it stands more than 8 × its noise (plus a tenth for intensity changes) above the same channel in
// every measured neighbour (left, right, above, below) over at most two channels, three channels
// either side being back near the neighbours; it is set to the highest neighbour's value. A band
// that is stronger in one spectrum than its neighbours (a thin layer) shows its whole width, and is
// left alone.
export function prepare(cube, K, wn, geom, settings, onProgress = null) {
  const { nx, ny } = geom, P = nx * ny, [lo, hi] = settings.range, idx = [];
  for (let c = 0; c < K; c++) if (wn[c] >= lo && wn[c] <= hi) idx.push(c);
  idx.sort((a, b) => wn[a] - wn[b]);
  const N = idx.length;
  if (N < 20) throw new Error(`only ${N} channels between ${lo} and ${hi} cm⁻¹: widen the range`);
  const w = Math.ceil(N / MAX_BINS), nb = Math.floor(N / w), chan = Float64Array.from({ length: nb * w }, (_, i) => wn[idx[i]]);
  const nu = Float64Array.from({ length: nb }, (_, j) => { let t = 0; for (let q = 0; q < w; q++) t += chan[j * w + q]; return t / w; });
  const measured = new Uint8Array(P), rowOf = new Int32Array(P).fill(-1), list = [];
  for (let p = 0; p < P; p++) {
    let fin = true, any = false;
    for (let c = 0; c < N; c++) { const v = cube[p * K + idx[c]]; if (!Number.isFinite(v)) { fin = false; break; } if (v !== 0) any = true; }
    if (fin && any) { measured[p] = 1; rowOf[p] = list.length; list.push(p); }
  }
  const n = list.length, V = new Float32Array(n * nb), base = new Float32Array(n * nb), noise = new Float64Array(n);
  const B = baselineFitter(nu, settings.baseline), y = new Float64Array(N), b = new Float64Array(nb), top = new Float64Array(N), ex = new Float64Array(N);
  const bin = () => { for (let j = 0; j < nb; j++) { let t = 0; for (let q = 0; q < w; q++) t += y[j * w + q]; b[j] = t / w; } };
  let spikes = 0;
  for (let i = 0; i < n; i++) {
    const p = list[i], o = p * K, r = Math.floor(p / nx), c0 = p % nx;
    for (let c = 0; c < N; c++) y[c] = cube[o + idx[c]];
    bin();
    const sd = Math.max(noiseSigma(b), 1e-9) * Math.sqrt(w);      // per channel, robust to the spikes
    const nbr = [c0 > 0 ? p - 1 : -1, c0 < nx - 1 ? p + 1 : -1, r > 0 ? p - nx : -1, r < ny - 1 ? p + nx : -1].filter(q => q >= 0 && measured[q]);
    if (nbr.length) {
      top.fill(-Infinity);
      for (const q of nbr) for (let c = 0; c < N; c++) top[c] = Math.max(top[c], cube[q * K + idx[c]]);
      for (let c = 0; c < N; c++) ex[c] = y[c] - top[c];
      let hit = false;
      for (let c = 0; c < N; c++) {
        const lim = 8 * sd + 0.1 * Math.abs(top[c]);
        if (!(ex[c] > lim)) continue;
        let e = c; while (e + 1 < N && ex[e + 1] > lim) e++;
        if (e - c <= 1 && (c < 3 || ex[c - 3] < ex[c] / 4) && (e + 3 >= N || ex[e + 3] < ex[e] / 4)) { for (let q = c; q <= e; q++) y[q] = top[q]; hit = true; spikes++; }
        c = e;
      }
      if (hit) bin();
    }
    noise[i] = Math.max(noiseSigma(b), 1e-9);
    const bl = B(b, noise[i]);
    for (let j = 0; j < nb; j++) { base[i * nb + j] = bl[j]; V[i * nb + j] = b[j] - bl[j]; }
    if (onProgress && (i % 2000 === 1999)) onProgress(i + 1, n);
  }
  return { P, n, nb, w, nu, chan, list: Int32Array.from(list), rowOf, measured, V, base, noise, spikes, range: [nu[0], nu[nb - 1]], degree: settings.baseline };
}

// A reference spectrum ({ name, wn, y }) on the analysis's bins, background taken off the same way
// and scaled to unit sum; NaN where it was not measured, `cover` the share of bins it covers.
export function prepRef(ref, prep) {
  const { nb, w, chan, nu } = prep, order = Array.from(ref.wn.keys()).sort((a, b) => ref.wn[a] - ref.wn[b]);
  const rx = Float64Array.from(order, i => ref.wn[i]), ry = Float64Array.from(order, i => ref.y[i]);
  const at = v => {
    if (!(v >= rx[0] && v <= rx[rx.length - 1])) return NaN;
    let a = 0, b = rx.length - 1; while (b - a > 1) { const m = (a + b) >> 1; if (rx[m] <= v) a = m; else b = m; }
    return rx[b] > rx[a] ? ry[a] + (ry[b] - ry[a]) * (v - rx[a]) / (rx[b] - rx[a]) : ry[a];
  };
  // taken as clean: a running median along the spectrum would clip a sharp band's top
  const d = Float64Array.from(chan, at), y = new Float64Array(nb).fill(NaN);
  let first = -1, last = -1;
  for (let j = 0; j < nb; j++) {
    let t = 0, ok = true; for (let q = 0; q < w; q++) { const v = d[j * w + q]; if (!Number.isFinite(v)) { ok = false; break; } t += v; }
    if (ok) { y[j] = t / w; if (first < 0) first = j; last = j; }
  }
  if (first < 0) return { name: ref.name, y, cover: 0 };
  const sub = Float64Array.from(y.subarray(first, last + 1), v => (Number.isFinite(v) ? v : 0)), bl = baselineFitter(nu.subarray(first, last + 1), prep.degree)(sub, noiseSigma(sub));
  let sum = 0;
  for (let j = first; j <= last; j++) { y[j] = Number.isFinite(y[j]) ? Math.max(0, y[j] - bl[j - first]) : NaN; if (Number.isFinite(y[j])) sum += y[j]; }
  if (sum > 0) for (let j = first; j <= last; j++) y[j] /= sum;
  return { name: ref.name, y, cover: (last - first + 1) / nb };
}

// --------------------------------------------------------------------------
// the materials
// --------------------------------------------------------------------------
// The successive projections algorithm on the spectra over their noise: the spectrum with the most
// left unexplained (over its own noise) by those already taken is taken next. Each one taken is
// averaged with the spectra most like it (cosine), so that its own noise does not count against
// every other spectrum: projected off a single noisy spectrum, a spectrum of the same material keeps
// about √2 × its noise. With k = 0 it stops once fewer than a handful of spectra (2, or 0.2 % of
// them) have more than `stop` × their noise left, i.e. what is left is noise; `steps` records that
// count at each stage. `basis` holds the averaged spectra.
export function purest(prep, { k = 0, kMax = 8, stop = 2 } = {}) {
  const { n, nb, V, noise } = prep, want = Math.min(k || kMax, n), need = Math.max(2, Math.ceil(0.002 * n));
  const near = Math.min(10, Math.max(3, Math.floor(n / 20))), norm = new Float64Array(n), left = new Float64Array(n);
  for (let i = 0; i < n; i++) { let t = 0; for (let j = 0; j < nb; j++) t += V[i * nb + j] ** 2; norm[i] = t; left[i] = t; }
  const picks = [], steps = [], basis = [], E = [];        // E: the averaged spectra, orthonormalised
  for (;;) {
    let over = 0, best = -1, br = -1;
    for (let i = 0; i < n; i++) {
      const r = left[i] / (nb * noise[i] ** 2);
      if (r > stop * stop) over++;
      if (r > br && !picks.includes(i)) { br = r; best = i; }
    }
    steps.push({ k: picks.length, over, worst: Math.sqrt(Math.max(0, br)) });
    if (picks.length >= want || (!k && over < need) || best < 0) break;
    picks.push(best);
    // the spectra most like it, averaged
    const o = best * nb, cos = new Float64Array(n);
    for (let i = 0; i < n; i++) { let t = 0; for (let j = 0; j < nb; j++) t += V[i * nb + j] * V[o + j]; cos[i] = norm[i] > 0 ? t / Math.sqrt(norm[i] * norm[best]) : -1; }
    const top = Array.from({ length: n }, (_, i) => i).sort((a, b) => cos[b] - cos[a]).slice(0, near), b = new Float64Array(nb);
    for (const i of top) for (let j = 0; j < nb; j++) b[j] += V[i * nb + j] / near;
    basis.push(b);
    // what every spectrum has left once projected off the averaged spectra so far
    const e = Float64Array.from(b);
    for (const f of E) { let t = 0; for (let j = 0; j < nb; j++) t += e[j] * f[j]; for (let j = 0; j < nb; j++) e[j] -= t * f[j]; }
    let en = 0; for (let j = 0; j < nb; j++) en += e[j] * e[j];
    if (!(en > 0)) break;
    en = Math.sqrt(en); for (let j = 0; j < nb; j++) e[j] /= en;
    E.push(e);
    for (let i = 0; i < n; i++) { let t = 0; for (let j = 0; j < nb; j++) t += V[i * nb + j] * e[j]; left[i] = Math.max(0, left[i] - t * t); }
  }
  return { picks, steps, basis };
}

// The materials and their amounts in every spectrum: H (k rows of nb, unit sum), W (n rows of k),
// and each spectrum's residual RMS over its noise.
export function decompose(prep, settings, refs = [], onStage = null) {
  const { n, nb, V, noise } = prep;
  if (!n) throw new Error("no spectrum was measured in this range");
  let k, H, W, picks = null, steps = null, mode;
  if (settings.useRefs) {
    if (refs.length < 2) throw new Error("load at least two reference spectra to unmix against them");
    const R = refs.map(r => prepRef(r, prep)), short = R.find(r => r.cover < 0.98);
    if (short) throw new Error(`${short.name} covers only ${Math.round(100 * short.cover)} % of the range: set the range inside it`);
    mode = "refs"; k = R.length; H = new Float64Array(k * nb); R.forEach((r, a) => H.set(r.y, a * nb));
    onStage && onStage("Unmixing every spectrum");
    W = nmfProject(V, n, nb, H, k, { iters: 200 });
  } else {
    mode = "auto";
    onStage && onStage("Finding the purest spectra");
    let basis;
    ({ picks, steps, basis } = purest(prep, settings));
    k = basis.length;
    if (!k) throw new Error("no spectrum stands out from its noise: nothing to decompose");
    // learnt on the picks and an even sample of the rest, then every spectrum unmixed
    const step = Math.max(1, Math.floor(n / LEARN)), learn = [...new Set([...picks, ...Array.from({ length: Math.ceil(n / step) }, (_, i) => i * step)])];
    const VL = new Float32Array(learn.length * nb);
    learn.forEach((i, r) => VL.set(V.subarray(i * nb, (i + 1) * nb), r * nb));
    onStage && onStage("Refining the materials");
    H = nmf(VL, learn.length, nb, basis.map(b => Array.from(b, v => Math.max(v, 0))), { iters: 150 }).H;
    onStage && onStage("Unmixing every spectrum");
    W = nmfProject(V, n, nb, H, k, { iters: 100 });
  }
  const resid = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let t = 0;
    for (let j = 0; j < nb; j++) { let m = 0; for (let a = 0; a < k; a++) m += W[i * k + a] * H[a * nb + j]; t += (V[i * nb + j] - m) ** 2; }
    resid[i] = Math.sqrt(t / nb) / noise[i];
  }
  return { mode, k, H, W, picks, steps, resid };
}

// how well each material matches each reference: HQI = 100 r² (0 for anticorrelated), over the bins
// the reference covers; a material takes the best reference's name at minHQI or above
export function matchRefs(comps, prep, refs, minHQI) {
  const { k, H } = comps, nb = prep.nb, R = refs.map(r => prepRef(r, prep));
  return Array.from({ length: k }, (_, a) => {
    const hqi = R.map(r => {
      if (r.cover < 0.5) return NaN;
      let n = 0, mx = 0, my = 0;
      for (let j = 0; j < nb; j++) if (Number.isFinite(r.y[j])) { n++; mx += H[a * nb + j]; my += r.y[j]; }
      mx /= n; my /= n;
      let sxy = 0, sxx = 0, syy = 0;
      for (let j = 0; j < nb; j++) if (Number.isFinite(r.y[j])) { const dx = H[a * nb + j] - mx, dy = r.y[j] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
      const c = sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : 0;
      return c > 0 ? 100 * c * c : 0;
    });
    let best = -1; hqi.forEach((h, r) => { if (h >= minHQI && (best < 0 || h > hqi[best])) best = r; });
    return { hqi, best };
  });
}

// --------------------------------------------------------------------------
// the layers
// --------------------------------------------------------------------------
// The lines across the layers on a grid: rows (axis "x") or columns (axis "y"), each with its
// points in ascending position s; t is each line's coordinate along the layers.
export function lineGeometry(geom, axis) {
  const { nx, x, y } = geom, along = axis === "x" ? x : y, across = axis === "x" ? y : x;
  const order = Array.from({ length: along.length }, (_, j) => j).sort((a, b) => along[a] - along[b]);
  return { axis, m: along.length, L: across.length, s: Float64Array.from(order, j => along[j]), t: Float64Array.from(across),
    point: axis === "x" ? (l, j) => l * nx + order[j] : (l, j) => order[j] * nx + l };
}

// Across which axis the layers lie on a map: the one whose positions explain the larger share of
// the variation in the materials' amounts (each scaled by its 99th percentile, so each weighs alike).
export function chooseAxis(prep, comps, geom) {
  if (geom.ny === 1) return { axis: "x", share: { x: NaN, y: NaN } };
  if (geom.nx === 1) return { axis: "y", share: { x: NaN, y: NaN } };
  const { k, W } = comps, rowOf = prep.rowOf, n = prep.n, share = {};
  const scale = Array.from({ length: k }, (_, a) => quantileSorted(Float64Array.from({ length: n }, (_, i) => W[i * k + a]).sort(), 0.99) || 1);
  for (const axis of ["x", "y"]) {
    const G = lineGeometry(geom, axis);
    let ssb = 0, sst = 0;
    for (let a = 0; a < k; a++) {
      const sum = new Float64Array(G.m), cnt = new Float64Array(G.m);
      let tot = 0, tn = 0, tq = 0;
      for (let l = 0; l < G.L; l++) for (let j = 0; j < G.m; j++) {
        const i = rowOf[G.point(l, j)]; if (i < 0) continue;
        const v = W[i * k + a] / scale[a]; sum[j] += v; cnt[j]++; tot += v; tq += v * v; tn++;
      }
      if (!tn) continue;
      const M = tot / tn; sst += tq - tn * M * M;
      for (let j = 0; j < G.m; j++) if (cnt[j]) ssb += cnt[j] * (sum[j] / cnt[j] - M) ** 2;
    }
    share[axis] = sst > 0 ? ssb / sst : 0;
  }
  return { axis: share.x >= share.y ? "x" : "y", share };
}

// each line's measured points: their positions and the materials' amounts scaled by `plateau`,
// smoothed over three neighbouring points
function lineData(prep, comps, G, l, plateau, scale) {
  const { k, W } = comps, js = [];
  for (let j = 0; j < G.m; j++) if (prep.rowOf[G.point(l, j)] >= 0) js.push(j);
  const m = js.length, s = Float64Array.from(js, j => G.s[j] * scale), raw = [], us = [];
  for (let a = 0; a < k; a++) {
    const r = Float64Array.from(js, j => W[prep.rowOf[G.point(l, j)] * k + a] / plateau[a]), u = new Float64Array(m);
    for (let q = 0; q < m; q++) { let t = 0, c = 0; for (let d = -1; d <= 1; d++) if (q + d >= 0 && q + d < m) { t += r[q + d]; c++; } u[q] = t / c; }
    raw.push(r); us.push(u);
  }
  return { js, s, us };
}

// each material's plateau: the median over the lines of its highest (smoothed) amount, among the
// lines where it reaches at least half its overall highest
function plateaus(prep, comps, G) {
  const one = new Float64Array(comps.k).fill(1), tops = Array.from({ length: comps.k }, () => []);
  for (let l = 0; l < G.L; l++) {
    const D = lineData(prep, comps, G, l, one, 1);
    if (D.js.length) D.us.forEach((u, a) => tops[a].push(u.reduce((t, v) => Math.max(t, v), 0)));
  }
  return Float64Array.from(tops, t => { const top = Math.max(0, ...t); return median(t.filter(v => v >= 0.5 * top)) || top || 1; });
}

const runsOf = lab => { const r = []; for (let j = 0; j < lab.length;) { let e = j; while (e + 1 < lab.length && lab[e + 1] === lab[j]) e++; r.push({ lab: lab[j], a: j, b: e }); j = e + 1; } return r; };

// One line: each point to the material with the largest scaled amount (none below one half); runs
// inside the line shorter than minPts go to the neighbouring run whose material is the more present
// over them. Returns the runs.
function segmentLine(us, m, minPts) {
  const k = us.length, lab = new Int16Array(m);
  for (let q = 0; q < m; q++) { let best = -1, bv = 0.5; for (let a = 0; a < k; a++) if (us[a][q] >= bv) { bv = us[a][q]; best = a; } lab[q] = best; }
  const level = (c, q) => (c < 0 ? 0.5 : us[c][q]);
  for (let guard = 0; guard < m; guard++) {
    const runs = runsOf(lab), i = runs.findIndex((r, j) => j > 0 && j < runs.length - 1 && r.b - r.a + 1 < minPts);
    if (i < 0) return runs;
    const r = runs[i], score = c => { let t = 0; for (let q = r.a; q <= r.b; q++) t += level(c, q); return t; };
    const to = score(runs[i - 1].lab) >= score(runs[i + 1].lab) ? runs[i - 1].lab : runs[i + 1].lab;
    for (let q = r.a; q <= r.b; q++) lab[q] = to;
  }
  return runsOf(lab);
}

// Where two neighbouring runs meet: the crossing of their levels (a material's scaled amount, or
// one half for none), and the width of the step between 16 % and 84 % of the way (for a step blurred
// by a Gaussian, twice its SD), with the broadening by the 3-point smoothing taken out.
function boundary(us, s, r1, r2, step) {
  const lv = (c, q) => (c < 0 ? 0.5 : us[c][q]), g = q => lv(r1.lab, q) - lv(r2.lab, q), e = r1.b;
  const cross = (q, target) => { const a = g(q) - target, b = g(q + 1) - target, t = a - b > 0 ? a / (a - b) : 0.5; return s[q] + Math.min(1, Math.max(0, t)) * (s[q + 1] - s[q]); };
  const pos = cross(e, 0), thr = r1.lab >= 0 && r2.lab >= 0 ? 0.68 : 0.34;
  let lq = -1, rq = -1;
  for (let q = e; q >= r1.a; q--) if (g(q) >= thr) { lq = q; break; }
  for (let q = e + 1; q <= r2.b; q++) if (g(q) <= -thr) { rq = q; break; }
  let width = NaN;
  if (lq >= 0 && rq >= 0) {
    const w = cross(rq - 1, -thr) - cross(lq, thr), sd2 = (w / 2) ** 2 - (2 / 3) * step * step;
    width = 2 * Math.sqrt(Math.max(0, sd2));
  }
  return { pos, width };
}

// The layers: every line segmented, the stack most lines share, its thicknesses over those lines
// (tilt taken out on a map), and a material and layer per point.
export function findLayers(prep, comps, geom, settings) {
  const { k } = comps, P = prep.P;
  const pick = settings.axis === "auto" || geom.nx === 1 || geom.ny === 1 ? chooseAxis(prep, comps, geom) : { axis: settings.axis, share: null };
  const G = lineGeometry(geom, pick.axis), scale = settings.kind === "depth" ? settings.depthScale : 1, plateau = plateaus(prep, comps, G);
  const label = new Float32Array(P).fill(NaN), lines = [];
  const diffs = []; for (let j = 1; j < G.m; j++) diffs.push(Math.abs(G.s[j] - G.s[j - 1]) * scale);
  const step = median(diffs) || 0;
  for (let l = 0; l < G.L; l++) {
    const D = lineData(prep, comps, G, l, plateau, scale), m = D.js.length;
    if (!m) { lines.push({ l, layers: [], key: "", D }); continue; }
    const runs = segmentLine(D.us, m, settings.minPts), layers = [];
    for (const r of runs) for (let q = r.a; q <= r.b; q++) label[G.point(l, D.js[q])] = r.lab;
    const lead = runs[0].lab < 0 ? 1 : 0, trail = runs[runs.length - 1].lab < 0 ? 1 : 0;
    for (let i = lead; i < runs.length - trail; i++) {
      const r = runs[i], left = i > 0 ? boundary(D.us, D.s, runs[i - 1], r, step) : null, right = i < runs.length - 1 ? boundary(D.us, D.s, r, runs[i + 1], step) : null;
      const from = left ? left.pos : D.s[0], to = right ? right.pos : D.s[m - 1];
      let peak = 0; if (r.lab >= 0) for (let q = r.a; q <= r.b; q++) peak = Math.max(peak, D.us[r.lab][q]);
      layers.push({ mat: r.lab, from, to, t: to - from, hasFrom: !!left, hasTo: !!right, open: !left || !right, pts: r.b - r.a + 1,
        wFrom: left ? left.width : NaN, wTo: right ? right.width : NaN, peak: r.lab >= 0 ? peak : NaN, a: r.a, b: r.b });
    }
    lines.push({ l, layers, key: layers.map(L => L.mat).join(","), D });
  }
  // the stack most lines share
  const count = new Map();
  for (const ln of lines) if (ln.layers.length) count.set(ln.key, (count.get(ln.key) || 0) + 1);
  const ranked = [...count].sort((a, b) => b[1] - a[1] || b[0].split(",").length - a[0].split(",").length);
  const key = ranked.length ? ranked[0][0] : null, cons = lines.filter(ln => ln.layers.length && ln.key === key);
  const stack = key == null ? [] : key.split(",").map(Number), nL = stack.length;
  // the tilt of the layers in the map: one slope shared by every boundary that is not a scan edge
  let tilt = null;
  if (G.L > 1 && cons.length >= 3) {
    let sxy = 0, sxx = 0;
    const bounds = [];
    // boundary 0 is the first layer's start, boundary b its b-th layer's end
    for (let b = 0; b <= nL; b++) {
      const pts = [];
      for (const ln of cons) {
        const L = b === 0 ? ln.layers[0] : ln.layers[b - 1], ok = b === 0 ? L.hasFrom : L.hasTo;
        if (ok) pts.push([G.t[ln.l], b === 0 ? L.from : L.to]);
      }
      if (pts.length < 3) continue;
      const mt = pts.reduce((t, q) => t + q[0], 0) / pts.length, mp = pts.reduce((t, q) => t + q[1], 0) / pts.length;
      for (const [t, p] of pts) { sxy += (t - mt) * (p - mp); sxx += (t - mt) ** 2; }
      bounds.push({ pts, mt, mp });
    }
    if (bounds.length && sxx > 0) {
      const slope = sxy / sxx;
      let ss = 0, dof = 0;
      for (const B of bounds) for (const [t, p] of B.pts) { ss += (p - B.mp - slope * (t - B.mt)) ** 2; dof++; }
      dof -= bounds.length + 1;
      tilt = { slope, deg: Math.atan(slope) * 180 / Math.PI, cos: 1 / Math.sqrt(1 + slope * slope), n: bounds.length, scatter: dof > 0 ? Math.sqrt(ss / dof) : NaN };
    }
  }
  const corr = settings.tilt && tilt ? tilt.cos : 1;
  // the stack's layers over the lines that share it
  const layers = stack.map((mat, i) => {
    const vals = Float64Array.from(cons, ln => ln.layers[i].t * corr), opens = Uint8Array.from(cons, ln => (ln.layers[i].open ? 1 : 0));
    const at = f => median(cons.map(ln => f(ln.layers[i])));
    const nOpen = opens.reduce((a, v) => a + v, 0);
    return { i, mat, from: at(L => L.from), to: at(L => L.to), t: stats(vals), vals, opens, nOpen, open: nOpen > 0,
      wFrom: at(L => L.wFrom) * corr, wTo: at(L => L.wTo) * corr, peak: at(L => L.peak), pts: at(L => L.pts) };
  });
  // The resolution: the median 16-84 % width of the stack's steps. A layer reaches its pure level
  // (95 %) only when it is thicker than about 1.65 × that; a thinner one reads too thick (it is the
  // width of its blurred profile at half height), and a material seen only in such a layer sets its
  // own plateau below the pure level, so its amount cannot show it.
  const resolution = median(layers.flatMap(L => [L.wFrom, L.wTo]));
  for (const L of layers) L.thin = L.mat >= 0 && ((Number.isFinite(resolution) && L.t.mean < 1.65 * resolution) || L.peak < 0.8);
  const totals = Float64Array.from(cons, ln => ln.layers.reduce((a, L) => a + L.t, 0) * corr);
  // a layer per point, on the lines that share the stack
  const layerOf = new Int16Array(P).fill(-1);
  for (const ln of cons) ln.layers.forEach((L, i) => { for (let q = L.a; q <= L.b; q++) layerOf[G.point(ln.l, ln.D.js[q])] = i; });
  // the mean profile across the layers over those lines, each shifted by the tilt so that their steps line up
  const acc = Array.from({ length: k }, () => new Float64Array(G.m)), cnt = new Float64Array(G.m), use = cons.length ? cons : lines.filter(ln => ln.D.js.length);
  const sAt = Float64Array.from(G.s, v => v * scale), tbar = use.reduce((a, ln) => a + G.t[ln.l], 0) / (use.length || 1);
  for (const ln of use) {
    const sh = tilt ? tilt.slope * (G.t[ln.l] - tbar) : 0, xs = ln.D.s, m = xs.length;
    for (let j = 0, q = 0; j < G.m; j++) {
      const x = sAt[j] + sh;
      if (!(x >= xs[0] && x <= xs[m - 1])) continue;
      while (q < m - 2 && xs[q + 1] < x) q++;
      const f = m > 1 && xs[q + 1] > xs[q] ? (x - xs[q]) / (xs[q + 1] - xs[q]) : 0;
      cnt[j]++;
      for (let a = 0; a < k; a++) acc[a][j] += m > 1 ? ln.D.us[a][q] * (1 - f) + ln.D.us[a][q + 1] * f : ln.D.us[a][0];
    }
  }
  const profile = { s: sAt, u: acc.map(v => Float64Array.from(v, (x, j) => (cnt[j] ? x / cnt[j] : NaN))) };
  const others = ranked.slice(1).map(([kk, n]) => ({ mats: kk.split(",").map(Number), n }));
  return { axis: pick.axis, share: pick.share, L: G.L, m: G.m, scale, step, resolution, plateau, stack, layers, total: stats(totals), tilt, corr,
    nLines: lines.filter(ln => ln.layers.length).length, nCons: cons.length, others, label, layerOf, profile,
    lineT: Float64Array.from(cons, ln => G.t[ln.l]), empty: lines.filter(ln => !ln.layers.length).length };
}

// --------------------------------------------------------------------------
// limits and spectra
// --------------------------------------------------------------------------
// Each layer's thickness on every line against its limits ({ lo, hi, use_lo, use_hi } by layer
// number). A layer cut by the scan's edge has a lower bound for a thickness: it can pass a minimum
// (when the bound is above it) but no maximum, and is otherwise undetermined.
export function thicknessTest(res, limits) {
  const rows = res.layers.map((L, i) => {
    const q = limits[i], lo = q && q.use_lo && Number.isFinite(q.lo) ? q.lo : null, hi = q && q.use_hi && Number.isFinite(q.hi) ? q.hi : null;
    if (lo == null && hi == null) return null;
    let n = 0, pass = 0, undet = 0;
    L.vals.forEach((v, j) => {
      if (L.opens[j]) { if (hi == null && v >= lo) { n++; pass++; } else undet++; return; }
      n++; if ((lo == null || v >= lo) && (hi == null || v <= hi)) pass++;
    });
    const share = n ? pass / n : NaN;
    return { lo, hi, n, pass, fail: n - pass, undet, share, ci: wilson(pass, n), level: !n ? null : pass === n ? "good" : share >= 0.95 ? "warn" : "bad" };
  });
  const order = { good: 0, warn: 1, bad: 2 }, tested = rows.filter(r => r && r.level);
  return { rows, set: rows.filter(Boolean).length, level: tested.length ? tested.reduce((a, r) => (order[r.level] > order[a] ? r.level : a), "good") : null };
}

// each material's scaled amount at every point (NaN where not measured)
export function amountMap(prep, comps, plateau, a) {
  const out = new Float32Array(prep.P).fill(NaN), k = comps.k;
  for (let p = 0; p < prep.P; p++) { const i = prep.rowOf[p]; if (i >= 0) out[p] = comps.W[i * k + a] / plateau[a]; }
  return out;
}

// each layer's mean spectrum over its points, at full resolution
export function layerSpectra(cube, K, P, layerOf, nL) {
  const sum = Array.from({ length: nL }, () => new Float64Array(K)), n = new Float64Array(nL);
  for (let p = 0; p < P; p++) { const i = layerOf[p]; if (i < 0) continue; n[i]++; for (let c = 0; c < K; c++) sum[i][c] += cube[p * K + c]; }
  return sum.map((s, i) => ({ y: n[i] ? s.map(v => v / n[i]) : null, n: n[i] }));
}
