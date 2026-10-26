// The D3/G calibration: band shapes calibrated in steps and then held for every
// sample, so that I(D3)/I(G) means the same on all of them.
//
//   standards        G from a graphite spectrum, D1 to D4 from a glassy carbon one
//   -> baseline 1    each band's median position and width over an uncoated reference map
//   -> baseline 2    the same over a coated map, from baseline 1
//
// A band too weak to measure in enough spectra keeps the previous step's shape (D3 on
// both maps of the 21 Oct calibration, so its shape is glassy carbon's). On the uncoated
// map, D3 can instead be measured on the map's median spectrum (medianSpectrumBand). Each
// spectrum is cut to the Q-Carbon range, cleared of cosmic rays and counted as carbon when
// its G stands 10 times above its noise; on carbon spectra the five heights (0 to 5000
// counts) and a poly-2 background are solved by bounded linear least squares, every
// position and width held. Samples are compared on the median, against an uncoated
// reference fitted the same way, and placed on a scale from that reference (0) to glassy
// carbon (1). Pure functions (no DOM), so they run under Node for the tests.

import { lmOne, baselineBasis, pinv, pinvSym, LORENTZ, GAUSS } from "./engine/core.js";
import { rng, quantileSorted } from "./pf-analysis.js";

// the bands in the order of the calibration's tables
export const BANDS = ["D4", "D1", "D3", "G", "D2"];

// the LabSpec Q-Carbon panel the calibration was made with: windows for the free fits,
// the range, the background order and the height limit
export const QC_SETTINGS = {
  range: [1000, 1800], order: 2, hmax: 5000, snrMin: 10, present: 3, minSupport: 0.05,
  bands: {
    G: { pos: 1580, pos_min: 1570, pos_max: 1590, width: 20, width_min: 10, width_max: 35, shape: "Lorentz" },
    D1: { pos: 1350, pos_min: 1340, pos_max: 1360, width: 45, width_min: 15, width_max: 90, shape: "Lorentz" },
    D2: { pos: 1620, pos_min: 1610, pos_max: 1630, width: 25, width_min: 10, width_max: 60, shape: "Lorentz" },
    D3: { pos: 1500, pos_min: 1490, pos_max: 1510, width: 100, width_min: 40, width_max: 200, shape: "Gauss" },
    D4: { pos: 1200, pos_min: 1190, pos_max: 1210, width: 50, width_min: 20, width_max: 120, shape: "Lorentz" },
  },
};

const shapesOf = (rows, notes) => Object.fromEntries(BANDS.map((b, i) => [b, { pos: rows[i][0], fwhm: rows[i][1], support: rows[i][2], note: notes[i] }]));

// The calibration of 21 Oct 2026 (positions and FWHM in cm-1, full precision): baseline 2
// is the one to apply; the standards and baseline 1 are the steps that led to it. Made by the
// calibration's scripts, so D3 is glassy carbon's throughout (it was not refined on the
// uncoated map's median spectrum). glassy: the glassy carbon standard's I(D3)/I(G) with each
// step's shapes, the top of the uncoated-to-amorphous scale.
export const PRESET_21OCT = {
  name: "21 Oct 2026", date: "2026-10-21", settings: QC_SETTINGS,
  steps: [
    { name: "Standards", glassy: 0.44572088595161247, shapes: shapesOf([[1210, 120, 1], [1349.307394136578, 89.99999999999999, 1], [1510, 169.96501432885, 1],
      [1577.908852151934, 14.34735068279007, 1], [1610.778311420766, 46.59874406724895, 1]], ["glassy carbon", "glassy carbon", "glassy carbon", "graphite", "glassy carbon"]) },
    { name: "Baseline 1 (uncoated reference)", glassy: 0, shapes: shapesOf([[1210, 120, 0], [1341.222822005198, 46.72824880672977, 3086], [1510, 169.96501432885, 0],
      [1573.019625563872, 21.56014371767737, 3096], [1612.090968228501, 16.41613588528793, 2699]], ["kept", "median", "kept", "median", "median"]) },
    { name: "Baseline 2 (coated)", glassy: 0.39115584956883603, shapes: shapesOf([[1210, 120, 0], [1342.754542709468, 89.74826429230919, 1382], [1510, 169.96501432885, -18],
      [1571.48191932493, 22.95410611753739, 1469], [1610, 40.15700537658364, 616]], ["kept", "median", "kept", "median", "median"]) },
  ],
};

export const shapeArea = (shape, h, w) => (shape === "Lorentz" ? h * w * Math.PI / 2 : h * w * Math.sqrt(Math.PI / (4 * Math.LN2)));
function line(shape, x, c, w) {
  if (shape === "Lorentz") { const u = (x - c) / (w / 2); return 1 / (1 + u * u); }
  const v = (x - c) / w; return Math.exp(-4 * Math.LN2 * v * v);
}

function medianOf(a) {
  const s = Float64Array.from(a).sort(), n = s.length;
  return !n ? NaN : n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

// --------------------------------------------------------------------------
// one spectrum: noise, cosmic rays, smoothing for the carbon test
// --------------------------------------------------------------------------
// robust SD of the second difference (white noise: its SD is sqrt(6) sigma)
export function noiseSigma(y) {
  const n = y.length - 2; if (n < 1) return NaN;
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = y[i + 2] - 2 * y[i + 1] + y[i];
  const m = medianOf(d);
  for (let i = 0; i < n; i++) d[i] = Math.abs(d[i] - m);
  return 1.4826 * medianOf(d) / Math.sqrt(6);
}

// a running median, the window repeating the end values beyond the spectrum
function runningMedian(y, size = 7) {
  const n = y.length, h = size >> 1, out = new Float64Array(n), w = new Float64Array(size);
  for (let i = 0; i < n; i++) {
    for (let k = -h; k <= h; k++) w[k + h] = y[Math.min(n - 1, Math.max(0, i + k))];
    w.sort(); out[i] = w[h];
  }
  return out;
}

// Cosmic rays: points more than thr x noise above a 7-point running median, removed only
// when the excursion is at most maxw points wide (a Raman band is wider), and replaced by
// the median of the unflagged points around them.
export function despike(y0, thr = 8, maxw = 2) {
  const y = Float64Array.from(y0), n = y.length;
  const sig = noiseSigma(y0) + 1e-9, mf = runningMedian(y0), lim = thr * Math.max(sig, Math.sqrt(Math.max(medianOf(y0), 1)));
  const flag = new Uint8Array(n);
  for (let i = 0; i < n; i++) flag[i] = y0[i] - mf[i] > lim ? 1 : 0;
  let spikes = 0;
  for (let i = 0; i < n;) {
    if (!flag[i]) { i++; continue; }
    let j = i; while (j < n && flag[j]) j++;
    if (j - i <= maxw) {
      const nb = [];
      for (let k = Math.max(0, i - 3); k < Math.min(n, j + 3); k++) if (!flag[k]) nb.push(y0[k]);
      if (nb.length) { const v = medianOf(nb); for (let k = i; k < j; k++) y[k] = v; spikes++; }
    }
    i = j;
  }
  return { y, spikes };
}

// least-squares polynomial (ascending powers) through (x, y), for small orders
function polyfit(x, y, order) {
  const q = order + 1, A = new Float64Array(q * q), b = new Float64Array(q), x0 = x.reduce((t, v) => t + v, 0) / x.length;
  let sc = 0; for (const v of x) sc = Math.max(sc, Math.abs(v - x0)); sc = sc || 1;
  for (let n = 0; n < x.length; n++) {
    const t = (x[n] - x0) / sc, p = new Float64Array(q); let v = 1;
    for (let k = 0; k < q; k++) { p[k] = v; v *= t; }
    for (let i = 0; i < q; i++) { b[i] += p[i] * y[n]; for (let j = 0; j < q; j++) A[i * q + j] += p[i] * p[j]; }
  }
  const inv = pinvSym(A, q), c = new Float64Array(q);
  for (let i = 0; i < q; i++) for (let j = 0; j < q; j++) c[i] += inv[i * q + j] * b[j];
  return t => { const u = (t - x0) / sc; let s = 0, v = 1; for (let k = 0; k < q; k++) { s += c[k] * v; v *= u; } return s; };
}

// Savitzky-Golay smoothing as scipy's savgol_filter in its default mode: the end points from
// a polynomial fitted to the first and the last window. The fit is linear in the counts, so
// every output is a fixed weighting of its window: the fit to a unit impulse at each place.
const SG = new Map();
function sgWeights(win, order) {
  const key = `${win},${order}`;
  if (!SG.has(key)) {
    const pos = Float64Array.from({ length: win }, (_, i) => i), at = Array.from({ length: win }, () => new Float64Array(win));
    for (let j = 0; j < win; j++) {
      const f = polyfit(pos, Float64Array.from({ length: win }, (_, i) => (i === j ? 1 : 0)), order);
      for (let t = 0; t < win; t++) at[t][j] = f(t);
    }
    SG.set(key, at);               // at[t]: the weights giving the fit's value at place t of a window
  }
  return SG.get(key);
}
export function savgol(y, win = 11, order = 2) {
  const n = y.length, h = win >> 1, out = new Float64Array(n);
  if (n < win) return Float64Array.from(y);
  const at = sgWeights(win, order), mid = at[h];
  for (let i = h; i < n - h; i++) { let s = 0; for (let j = 0; j < win; j++) s += mid[j] * y[i - h + j]; out[i] = s; }
  for (let i = 0; i < h; i++) {
    let a = 0, b = 0;
    for (let j = 0; j < win; j++) { a += at[i][j] * y[j]; b += at[win - h + i][j] * y[n - win + j]; }
    out[i] = a; out[n - h + i] = b;
  }
  return out;
}

// --------------------------------------------------------------------------
// a set of spectra, ready for the calibration
// --------------------------------------------------------------------------
// Cut to the range, despiked, with each spectrum's noise and G signal-to-noise: the peak of
// the smoothed spectrum in 1540-1630 cm-1 above a straight line through 1000-1060 and
// 1740-1800 cm-1, over the noise. Spectra never measured (NaN) or flat carry NaN.
export function prepare(cube, K, wn, P, settings = QC_SETTINGS) {
  const [lo, hi] = settings.range, ks = [];
  for (let c = 0; c < K; c++) if (wn[c] >= lo && wn[c] <= hi) ks.push(c);
  const N = ks.length;
  if (N < 20) throw new Error(`fewer than 20 channels between ${lo} and ${hi} cm⁻¹`);
  const nu = Float64Array.from(ks, c => wn[c]), Y = new Float64Array(P * N), sig = new Float64Array(P).fill(NaN), snr = new Float64Array(P).fill(NaN);
  const edge = [], gw = [];
  nu.forEach((v, n) => { if (v < 1060 || v > 1740) edge.push(n); if (v > 1540 && v < 1630) gw.push(n); });
  const ex = Float64Array.from(edge, n => nu[n]);
  let spikes = 0;
  for (let p = 0; p < P; p++) {
    const y0 = Float64Array.from(ks, c => cube[p * K + c]);
    if (!y0.every(Number.isFinite)) { Y.fill(NaN, p * N, (p + 1) * N); continue; }
    const d = despike(y0); spikes += d.spikes;
    Y.set(d.y, p * N);
    const s = noiseSigma(d.y); sig[p] = s;
    if (!(s > 0) || !gw.length || edge.length < 2) continue;
    const sm = savgol(d.y), base = polyfit(ex, Float64Array.from(edge, n => sm[n]), 1);
    let top = -Infinity; for (const n of gw) top = Math.max(top, sm[n] - base(nu[n]));
    snr[p] = top / s;
  }
  const carbon = [];
  for (let p = 0; p < P; p++) if (snr[p] >= settings.snrMin) carbon.push(p);
  return { nu, N, P, Y, sig, snr, carbon, spikes, settings };
}

// --------------------------------------------------------------------------
// fixed shapes: heights and background by bounded linear least squares
// --------------------------------------------------------------------------
function cholesky(A, n) {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i * n + j];
    for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
    if (i === j) { if (!(s > 0)) return null; L[i * n + i] = Math.sqrt(s); } else L[i * n + j] = s / L[j * n + j];
  }
  return L;
}
function cholSolve(L, n, b) {
  const y = new Float64Array(n), x = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k]; y[i] = s / L[i * n + i]; }
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
  return x;
}

// Minimise |y - X x|² with the last nb variables in [0, hmax]. The problem is strictly convex,
// so the one set of bounds satisfying the optimality conditions is the solution; the sets are
// tried with the fewest bounds active first (most spectra need none or one). Every set's
// factor depends on X alone, so it is built once for all spectra.
export function boxSolver(X, N, q, nb, hmax) {
  const G = new Float64Array(q * q);
  for (let n = 0; n < N; n++) for (let i = 0; i < q; i++) { const a = X[n * q + i]; if (a) for (let j = 0; j < q; j++) G[i * q + j] += a * X[n * q + j]; }
  const nf0 = q - nb, states = [];
  for (let code = 0; code < 3 ** nb; code++) {
    const st = new Int8Array(nb); let c = code, act = 0;
    for (let j = 0; j < nb; j++) { st[j] = c % 3; c = (c / 3) | 0; if (st[j]) act++; }
    const free = []; for (let i = 0; i < nf0; i++) free.push(i); st.forEach((s, j) => { if (!s) free.push(nf0 + j); });
    const nf = free.length, Gf = new Float64Array(nf * nf);
    free.forEach((a, i) => free.forEach((b, j) => { Gf[i * nf + j] = G[a * q + b]; }));
    const L = cholesky(Gf, nf);
    if (L) states.push({ st, act, free, L });
  }
  states.sort((a, b) => a.act - b.act);
  return {
    G, q, cov: pinvSym(G, q),
    solve(bv) {
      let bmax = 0; for (const v of bv) bmax = Math.max(bmax, Math.abs(v));
      const tol = 1e-9 * (bmax || 1), x = new Float64Array(q);
      let best = null, bestCost = Infinity;
      for (const { st, free, L } of states) {
        x.fill(0); st.forEach((s, j) => { if (s === 2) x[nf0 + j] = hmax; });
        const rhs = Float64Array.from(free, a => { let s = bv[a]; for (let j = 0; j < nb; j++) if (st[j] === 2) s -= G[a * q + nf0 + j] * hmax; return s; });
        const xf = cholSolve(L, free.length, rhs);
        let ok = true;
        free.forEach((a, i) => { x[a] = xf[i]; if (a >= nf0 && (xf[i] < -1e-9 * hmax || xf[i] > hmax * (1 + 1e-12))) ok = false; });
        if (!ok) continue;
        for (let j = nf0; j < q; j++) x[j] = Math.min(hmax, Math.max(0, x[j]));
        // the gradient G x - b: non-negative on a height held at 0, non-positive at hmax
        let kkt = true, cost = 0;
        for (let i = 0; i < q; i++) {
          let g = -bv[i]; for (let j = 0; j < q; j++) g += G[i * q + j] * x[j];
          cost += x[i] * (0.5 * (g + bv[i]) - bv[i]);
          if (i >= nf0 && st[i - nf0] === 1 && g < -tol) kkt = false;
          if (i >= nf0 && st[i - nf0] === 2 && g > tol) kkt = false;
        }
        if (kkt) return x;
        if (cost < bestCost) { bestCost = cost; best = Float64Array.from(x); }
      }
      return best || x;
    },
  };
}

// Apply fixed shapes ({band: {pos, fwhm}}) to the spectra in list: per spectrum the heights,
// their standard errors (the covariance of the linear fit at its residual variance), the areas
// of each line shape and the ratios over G. A zero G height gives no ratio (NaN).
export function unmix(prep, shapes, list = prep.carbon) {
  const { nu, N, Y, settings } = prep, nbase = settings.order + 1, nb = BANDS.length, q = nbase + nb;
  const [lo, hi] = settings.range, mid = (lo + hi) / 2, half = (hi - lo) / 2, X = new Float64Array(N * q);
  for (let n = 0; n < N; n++) {
    const t = (nu[n] - mid) / half; let v = 1;
    for (let k = 0; k < nbase; k++) { X[n * q + k] = v; v *= t; }
    BANDS.forEach((b, j) => { X[n * q + nbase + j] = line(settings.bands[b].shape, nu[n], shapes[b].pos, shapes[b].fwhm); });
  }
  const box = boxSolver(X, N, q, nb, settings.hmax), M = list.length;
  const out = { list: Int32Array.from(list), h: new Float64Array(M * nb), se: new Float64Array(M * nb), area: new Float64Array(M * nb),
                rmse: new Float64Array(M), r2: new Float64Array(M), base: new Float64Array(M * nbase), bands: BANDS, nbase };
  const bv = new Float64Array(q);
  list.forEach((p, i) => {
    const y = Y.subarray(p * N, (p + 1) * N);
    bv.fill(0);
    for (let n = 0; n < N; n++) for (let k = 0; k < q; k++) bv[k] += X[n * q + k] * y[n];
    const x = box.solve(bv);
    let ssr = 0, my = 0, sst = 0;
    for (let n = 0; n < N; n++) { let f = 0; for (let k = 0; k < q; k++) f += X[n * q + k] * x[k]; ssr += (y[n] - f) ** 2; my += y[n] / N; }
    for (let n = 0; n < N; n++) sst += (y[n] - my) ** 2;
    const s2 = ssr / (N - q);
    out.rmse[i] = Math.sqrt(ssr / N); out.r2[i] = 1 - ssr / sst;
    for (let k = 0; k < nbase; k++) out.base[i * nbase + k] = x[k];
    BANDS.forEach((b, j) => {
      const h = x[nbase + j];
      out.h[i * nb + j] = h; out.se[i * nb + j] = Math.sqrt(box.cov[(nbase + j) * q + nbase + j] * s2);
      out.area[i * nb + j] = shapeArea(settings.bands[b].shape, h, shapes[b].fwhm);
    });
  });
  return withRatios(out);
}

function withRatios(out) {
  const M = out.list.length, nb = BANDS.length, j = Object.fromEntries(BANDS.map((b, i) => [b, i]));
  const H = (i, b) => out.h[i * nb + j[b]], E = (i, b) => out.se[i * nb + j[b]], A = (i, b) => out.area[i * nb + j[b]];
  const r = (a, b) => (b ? a / b : NaN);
  out.ratio = {}; for (const k of ["ID3/IG", "AD3/AG", "ID1/IG", "ID4/IG", "ID3/IG_se"]) out.ratio[k] = new Float64Array(M);
  for (let i = 0; i < M; i++) {
    const g = H(i, "G"), d3 = H(i, "D3"), v = r(d3, g);
    out.ratio["ID3/IG"][i] = v; out.ratio["AD3/AG"][i] = r(A(i, "D3"), A(i, "G"));
    out.ratio["ID1/IG"][i] = r(H(i, "D1"), g); out.ratio["ID4/IG"][i] = r(H(i, "D4"), g);
    // the ratio's error from the two heights' (a D3 at zero carries its own error over G)
    out.ratio["ID3/IG_se"][i] = d3 === 0 ? E(i, "D3") / g : Math.abs(v) * Math.sqrt((E(i, "D3") / d3) ** 2 + (E(i, "G") / g) ** 2);
  }
  return out;
}

// --------------------------------------------------------------------------
// free fits and one calibration step
// --------------------------------------------------------------------------
// The Q-Carbon fit of each spectrum in list (every band free inside its window, heights 0 to
// hmax, poly-2 background fitted with the bands), by this app's Levenberg-Marquardt. It starts
// where the calibration's own scripts start: the shapes of the step before, G at the spectrum's
// height above its 5th percentile and the other bands at 0.3 of it, the background through the
// lower 30 % of the counts. From the linear start a map fit uses, a weak band begun at zero can
// stay there (D4 of a glassy carbon standard does). Returns per spectrum and band the height,
// position and width, and each fit's R².
export function freeFits(prep, list, init, { maxIter = 200, tol = 1e-10 } = {}) {
  const { nu, N, Y, settings } = prep, nb = BANDS.length, M = list.length, nbase = settings.order + 1, q = nbase + 3 * nb;
  const B = baselineBasis(nu, settings.range[0], settings.range[1], settings.order), pB = pinv(B, N, nbase);
  const kinds = Int32Array.from(BANDS, b => (settings.bands[b].shape === "Lorentz" ? LORENTZ : GAUSS));
  const lo = new Float64Array(q).fill(-Infinity), hi = new Float64Array(q).fill(Infinity), free = new Uint8Array(q).fill(1), th0 = new Float64Array(q);
  BANDS.forEach((b, j) => {
    const w = settings.bands[b], i = nbase + 3 * j;
    lo[i] = 0; hi[i] = settings.hmax; lo[i + 1] = w.pos_min; hi[i + 1] = w.pos_max; lo[i + 2] = w.width_min; hi[i + 2] = w.width_max;
    th0[i + 1] = init[b].pos; th0[i + 2] = init[b].fwhm;
  });
  const out = { list: Int32Array.from(list), h: new Float64Array(M * nb), c: new Float64Array(M * nb), w: new Float64Array(M * nb), r2: new Float64Array(M) };
  const th = new Float64Array(q), low = new Float64Array(N);
  list.forEach((p, i) => {
    const y = Y.subarray(p * N, (p + 1) * N), s = Float64Array.from(y).sort();
    const H = Math.max(s[N - 1] - quantileSorted(s, 0.05), 1), p30 = quantileSorted(s, 0.3);
    th.set(th0);
    for (let n = 0; n < N; n++) low[n] = Math.min(y[n], p30);
    for (let k = 0; k < nbase; k++) { let v = 0; for (let n = 0; n < N; n++) v += pB[k * N + n] * low[n]; th[k] = v; }
    BANDS.forEach((b, j) => { th[nbase + 3 * j] = b === "G" ? H : 0.3 * H; });
    for (let k = nbase; k < q; k++) th[k] = Math.min(hi[k] - 1e-6, Math.max(lo[k] + 1e-6, th[k]));
    lmOne(nu, y, B, th, lo, hi, free, nbase, kinds, maxIter, tol);
    let ssr = 0, my = 0, sst = 0;
    for (let n = 0; n < N; n++) {
      let f = 0; for (let k = 0; k < nbase; k++) f += th[k] * B[n * nbase + k];
      BANDS.forEach((b, j) => { f += th[nbase + 3 * j] * line(settings.bands[b].shape, nu[n], th[nbase + 3 * j + 1], th[nbase + 3 * j + 2]); });
      ssr += (y[n] - f) ** 2; my += y[n] / N;
    }
    for (let n = 0; n < N; n++) sst += (y[n] - my) ** 2;
    out.r2[i] = 1 - ssr / sst;
    for (let j = 0; j < nb; j++) { out.h[i * nb + j] = th[nbase + 3 * j]; out.c[i * nb + j] = th[nbase + 3 * j + 1]; out.w[i * nb + j] = th[nbase + 3 * j + 2]; }
  });
  return out;
}

// One step: each band's median position and width over the spectra where it is present (height
// at least `present` x noise, the fit's R² above 0.5); a band present in fewer than minSupport of
// the spectra keeps the previous shape, its support then given as minus its count.
export function calibrate(fits, sig, prev, settings = QC_SETTINGS) {
  const nb = BANDS.length, M = fits.list.length, out = {};
  BANDS.forEach((b, j) => {
    const cs = [], ws = []; let atBound = 0;
    const w = settings.bands[b], near = (v, e) => Math.abs(v - e) <= 0.05;
    for (let i = 0; i < M; i++) {
      const s = sig[fits.list[i]], h = fits.h[i * nb + j];
      if (!(h / s >= settings.present) || !(fits.r2[i] > 0.5)) continue;
      const c = fits.c[i * nb + j], wd = fits.w[i * nb + j];
      cs.push(c); ws.push(wd);
      if (near(c, w.pos_min) || near(c, w.pos_max) || near(wd, w.width_min) || near(wd, w.width_max)) atBound++;
    }
    out[b] = cs.length >= settings.minSupport * M && cs.length
      ? { pos: medianOf(cs), fwhm: medianOf(ws), support: cs.length, atBound: atBound / cs.length, note: "median" }
      : { pos: prev[b].pos, fwhm: prev[b].fwhm, support: -cs.length, atBound: NaN, note: "kept" };
  });
  return out;
}

// The element-wise median of the spectra in list
export function medianSpectrum(prep, list) {
  const { N, Y } = prep, out = new Float64Array(N), col = new Float64Array(list.length);
  for (let n = 0; n < N; n++) { list.forEach((p, i) => { col[i] = Y[p * N + n]; }); out[n] = medianOf(col); }
  return out;
}

// A band that no single spectrum of a map carries (D3 on an uncoated map: under 3 x noise in every
// one) measured on the map's median spectrum instead, fitted freely from the step before. It counts
// as present when its height is at least `present` x the larger of the fit's residual RMS and the
// noise a median of n spectra has (sqrt(pi / 2) sigma / sqrt(n)): the median of whole counts is
// itself whole, so its second-difference noise reads 0, and the residual is mostly misfit, which a
// band must stand above. A position or width on its window limit is not set by the fit and keeps
// the step before's value. On a low-D3 map the fit puts a narrow band (about 50 cm-1) on the
// 1490 limit at under 2 x the residual: a shoulder, not the broad D3 of a coating.
export function medianSpectrumBand(prep, list, init, band, settings = prep.settings) {
  const med = medianSpectrum(prep, list), j = BANDS.indexOf(band);
  const f = freeFits({ nu: prep.nu, N: prep.N, Y: med, settings }, [0], init);
  const fitted = Object.fromEntries(BANDS.map((b, k) => [b, { pos: f.c[k], fwhm: f.w[k] }]));
  const U = unmix({ nu: prep.nu, N: prep.N, Y: med, settings }, fitted, [0]);
  const sMed = Math.sqrt(Math.PI / 2) * medianOf(list.map(p => prep.sig[p])) / Math.sqrt(list.length);
  const noise = Math.max(U.rmse[0], sMed), w = settings.bands[band], at = (v, a, b) => Math.abs(v - a) <= 0.05 || Math.abs(v - b) <= 0.05;
  const h = f.h[j], pos = f.c[j], fwhm = f.w[j], z = h / noise, present = z >= settings.present && f.r2[0] > 0.5;
  const posLim = at(pos, w.pos_min, w.pos_max), wLim = at(fwhm, w.width_min, w.width_max);
  return { n: list.length, h, pos, fwhm, z, r2: f.r2[0], noise, present, posLim, wLim,
           shape: present ? { pos: posLim ? init[band].pos : pos, fwhm: wLim ? init[band].fwhm : fwhm } : null };
}

// One calibration step's shapes with band refined on the median spectrum where the spectrum-by-
// spectrum rule kept the step before's shape. The result says what was found either way.
export function refineOnMedian(shapes, prep, list, prev, band = "D3") {
  if (shapes[band].note !== "kept") return { shapes, median: null };
  const m = medianSpectrumBand(prep, list, prev, band), out = { ...shapes };
  if (m.shape) {
    const kept = [m.posLim ? "position" : "", m.wLim ? "width" : ""].filter(Boolean);
    out[band] = { pos: m.shape.pos, fwhm: m.shape.fwhm, support: shapes[band].support, atBound: NaN,
                  note: `median spectrum${kept.length ? ` (${kept.join(" and ")} on a window limit, kept)` : ""}` };
  }
  return { shapes: out, median: m };
}

// I(D3)/I(G) of one reference spectrum ({ wn, y }) with a step's shapes held, and how well they fit it
export function standardRatio(spec, shapes, settings = QC_SETTINGS) {
  const p = prepare(Float64Array.from(spec.y), spec.y.length, spec.wn, 1, { ...settings, snrMin: -Infinity }), U = unmix(p, shapes);
  return { ratio: U.ratio["ID3/IG"][0], rn: U.rmse[0] / p.sig[0] };
}

// The first step: the standards fitted with the Q-Carbon windows from their default shapes;
// G is graphite's, D1 to D4 are glassy carbon's. Each standard: { wn, y }.
export function standards(graphite, glassy, settings = QC_SETTINGS) {
  const one = s => {
    const prep = prepare(Float64Array.from(s.y), s.y.length, s.wn, 1, { ...settings, snrMin: -Infinity });
    const init = Object.fromEntries(BANDS.map(b => [b, { pos: settings.bands[b].pos, fwhm: settings.bands[b].width }]));
    const f = freeFits(prep, [0], init);
    return Object.fromEntries(BANDS.map((b, j) => [b, { h: f.h[j], pos: f.c[j], fwhm: f.w[j] }]));
  };
  const gr = one(graphite), gl = one(glassy);
  const shapes = Object.fromEntries(BANDS.map(b => {
    const s = b === "G" ? gr[b] : gl[b];
    return [b, { pos: s.pos, fwhm: s.fwhm, support: 1, note: b === "G" ? "graphite" : "glassy carbon" }];
  }));
  return { shapes, graphite: gr, glassy: gl };
}

// --------------------------------------------------------------------------
// comparing samples on the median
// --------------------------------------------------------------------------
// values grouped into blocks of neighbouring spectra (keys: one block id per value), empty
// and non-finite values left out, in order of first appearance
export function blocksOf(values, keys) {
  const m = new Map();
  values.forEach((v, i) => { if (!Number.isFinite(v)) return; const k = keys[i]; if (!m.has(k)) m.set(k, []); m.get(k).push(v); });
  return [...m.values()];
}

// the k-th smallest of a[0..n) (quickselect; reorders a)
function select(a, n, k) {
  let l = 0, r = n - 1;
  while (r > l) {
    const piv = a[(l + r) >> 1]; let i = l, j = r;
    while (i <= j) {
      while (a[i] < piv) i++;
      while (a[j] > piv) j--;
      if (i <= j) { const t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
    }
    if (k <= j) r = j; else if (k >= i) l = i; else break;
  }
  return a[k];
}
function resampledMedian(blocks, R, buf) {
  let n = 0;
  for (let i = 0; i < blocks.length; i++) for (const v of blocks[Math.floor(R() * blocks.length)]) buf[n++] = v;
  if (!n) return NaN;
  const k = (n - 1) >> 1, lo = select(buf, n, k);
  if (n % 2) return lo;
  let hi = Infinity; for (let i = k + 1; i < n; i++) if (buf[i] < hi) hi = buf[i];
  return 0.5 * (lo + hi);
}
// room for any resample: as many values as blocks times the largest block
const room = blocks => new Float64Array(blocks.length * Math.max(1, ...blocks.map(b => b.length)));

// The median with a 95 % interval from B resamples of whole blocks: neighbouring map spectra
// share their signal, so resampling spectra one by one would make the interval too narrow.
export function blockMedian(blocks, { B = 2000, seed = 21 } = {}) {
  const all = Float64Array.from(blocks.flat()).sort(), R = rng(seed), d = new Float64Array(B), buf = room(blocks);
  for (let b = 0; b < B; b++) d[b] = resampledMedian(blocks, R, buf);
  d.sort();
  return { n: all.length, blocks: blocks.length, median: quantileSorted(all, 0.5), lo: quantileSorted(d, 0.025), hi: quantileSorted(d, 0.975) };
}

// Where a value sits between the uncoated reference (0) and glassy carbon, the amorphous
// reference (1), on I(D3)/I(G) with the same held shapes. No scale when glassy carbon does not
// read above the reference with those shapes.
export const scaleIndex = (r, ref, top) => (top > ref ? (r - ref) / (top - ref) : NaN);

// The difference of medians, sample minus reference, with its spatial block bootstrap 95 %
// interval (each resample draws the blocks of both anew), and the ratio of the medians. Given
// glassy carbon's value (top), also the sample's place on the scale, with its interval from the
// same resamples (the reference's spread moves both ends of the scale's bottom).
export function compareMedians(sBlocks, rBlocks, { B = 2000, seed = 21, top = NaN } = {}) {
  const ms = quantileSorted(Float64Array.from(sBlocks.flat()).sort(), 0.5), mr = quantileSorted(Float64Array.from(rBlocks.flat()).sort(), 0.5);
  const R = rng(seed), d = new Float64Array(B), x = new Float64Array(B), bs = room(sBlocks), br = room(rBlocks);
  for (let b = 0; b < B; b++) { const s = resampledMedian(sBlocks, R, bs), r = resampledMedian(rBlocks, R, br); d[b] = s - r; x[b] = scaleIndex(s, r, top); }
  d.sort(); x.sort();
  const out = { sample: ms, reference: mr, diff: ms - mr, lo: quantileSorted(d, 0.025), hi: quantileSorted(d, 0.975), ratio: mr ? ms / mr : NaN };
  const index = scaleIndex(ms, mr, top);
  return Number.isFinite(index) ? { ...out, index, indexLo: quantileSorted(x, 0.025), indexHi: quantileSorted(x, 0.975) } : out;
}

// block ids: 8 x 8 pixels on a map, 150 µm squares of stage position for ParticleFinder particles
export const mapBlock = (p, nx, b = 8) => `${Math.floor((p % nx) / b)},${Math.floor(Math.floor(p / nx) / b)}`;
export const stageBlock = (x, y, s = 150) => `${Math.floor(x / s)},${Math.floor(y / s)}`;

// --------------------------------------------------------------------------
// saved calibrations
// --------------------------------------------------------------------------
// This app's calibration file holds the settings and every step; the package's
// baselines_D3G.json (qcarbon_settings plus one block per baseline) reads as well.
export function toJSON(cal) {
  return JSON.stringify({ format: "carbo-metrics D3/G calibration", version: 1, name: cal.name, date: cal.date || null,
    settings: cal.settings, steps: cal.steps.map(s => ({ name: s.name, shapes: s.shapes, glassy: Number.isFinite(s.glassy) ? s.glassy : null })) }, null, 1);
}

export function fromJSON(text, fileName = "calibration") {
  const d = JSON.parse(text);
  if (d.format === "carbo-metrics D3/G calibration") {
    if (!Array.isArray(d.steps) || !d.steps.length) throw new Error(`${fileName}: no calibration steps`);
    return { name: d.name || fileName, date: d.date, settings: checkSettings(d.settings),
             steps: d.steps.map(s => ({ name: s.name, shapes: checkShapes(s.shapes, fileName), ...(Number.isFinite(s.glassy) ? { glassy: s.glassy } : {}) })) };
  }
  // the 21 Oct package layout: settings under qcarbon_settings, each baseline a block of bands
  const q = d.qcarbon_settings;
  if (!q || !q.bands) throw new Error(`${fileName}: not a D3/G calibration file`);
  const settings = checkSettings({
    range: q["range_cm-1"], order: q.baseline_poly_order, hmax: q.height_range ? q.height_range[1] : 5000,
    snrMin: QC_SETTINGS.snrMin, present: QC_SETTINGS.present, minSupport: QC_SETTINGS.minSupport,
    bands: Object.fromEntries(Object.entries(q.bands).map(([b, v]) => [b, { pos: v.position, pos_min: v.position_min, pos_max: v.position_max,
      width: v.width, width_min: v.width_min, width_max: v.width_max, shape: v.shape }])),
  });
  const steps = Object.entries(d).filter(([k, v]) => k !== "qcarbon_settings" && v && typeof v === "object" && v.G && v.G["position_cm-1"] != null)
    .map(([k, v]) => ({ name: k, shapes: checkShapes(Object.fromEntries(Object.entries(v).map(([b, s]) => [b, { pos: s["position_cm-1"], fwhm: s["FWHM_cm-1"], note: s.note || "" }])), fileName) }));
  if (!steps.length) throw new Error(`${fileName}: no baselines in this file`);
  return { name: fileName, settings, steps };
}

function checkSettings(s) {
  if (!s || !Array.isArray(s.range) || !(s.range[1] > s.range[0])) throw new Error("calibration: no valid spectral range");
  for (const b of BANDS) if (!s.bands || !s.bands[b]) throw new Error(`calibration: band ${b} missing from the settings`);
  for (const b of BANDS) if (!["Lorentz", "Gauss"].includes(s.bands[b].shape)) throw new Error(`calibration: band ${b} has no Lorentz or Gauss shape`);
  return { ...QC_SETTINGS, ...s, order: Number.isInteger(s.order) ? s.order : 2 };
}
function checkShapes(sh, fileName) {
  for (const b of BANDS) if (!sh || !sh[b] || !(sh[b].fwhm > 0) || !Number.isFinite(sh[b].pos)) throw new Error(`${fileName}: band ${b} has no position and width`);
  return sh;
}
