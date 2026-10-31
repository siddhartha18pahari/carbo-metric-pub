// Peak fitting of any spectrum, as LabSpec's Peak fitting: a list of bands, each a Gauss,
// Lorentz, Gauss-Lorentz (pseudo-Voigt), Voigt or Breit-Wigner-Fano line, over a polynomial
// background fitted with them or a background taken off first. Every parameter is free between
// bounds, fixed, or linked to the same parameter of another band (equal, at an offset, or in a
// ratio). Bounded Levenberg-Marquardt on analytic derivatives; the standard errors come from the
// covariance at the residual's variance. One spectrum (fitOne) or many (fitMany, what the workers
// run), each started from the model fitted to the spectrum in view.
//
// Voigt: the real part of the Faddeeva function w(z), z = (x + iγ) / (σ√2), computed by
// Weideman's rational series (1994, SIAM J. Numer. Anal. 31, 1497) with N = 32 terms, and its
// derivative from w'(z) = −2 z w(z) + 2i/√π. Its FWHM: Olivero & Longbothum (1977).
// Areas are each band's integral over all shifts (BWF: over the fit range, as its tail does not end).

import { pinvSym } from "./engine/core.js";
import { noiseSigma, background, savgol, despikeSpectrum } from "./spectra.js";

export const SHAPES = { gauss: "Gauss", lorentz: "Lorentz", pv: "Gauss-Lorentz", voigt: "Voigt", bwf: "Breit-Wigner-Fano" };
export const PARAMS = { gauss: ["h", "c", "w"], lorentz: ["h", "c", "w"], pv: ["h", "c", "w", "eta"], voigt: ["h", "c", "w", "wl"], bwf: ["h", "c", "w", "q"] };
export const PARAM_LABEL = { h: "Height", c: "Position", w: "Width", eta: "Lorentz share", wl: "Lorentz width", q: "Coupling 1/Q" };
export const paramLabel = (shape, k) => (shape === "voigt" && k === "w" ? "Gauss width" : PARAM_LABEL[k]);
// what each fitted band gives: height, position, FWHM, area and its extra parameter
export const OUT = ["h", "c", "fwhm", "area", "x"];
export const OUT_LABEL = { h: "Height", c: "Position", fwhm: "FWHM", area: "Area", x: "Shape" };
export const OUT_UNIT = { h: "counts", c: "cm⁻¹", fwhm: "cm⁻¹", area: "counts × cm⁻¹", x: "" };
export const STATUS = ["not measured", "fitted", "a position or width on its bound", "fit failed"];

const G4 = 4 * Math.log(2), GAREA = Math.sqrt(Math.PI / G4), SQPI = Math.sqrt(Math.PI), S2L = 2 * Math.sqrt(2 * Math.log(2));

// --------------------------------------------------------------------------
// the Faddeeva function in the upper half plane (Weideman 1994)
// --------------------------------------------------------------------------
const NW = 32, MW = 2 * NW, LW = Math.sqrt(NW / Math.SQRT2), WC = (() => {
  const M2 = 2 * MW, f = new Float64Array(M2);
  for (let i = 1; i < M2; i++) { const t = LW * Math.tan((i - MW) * Math.PI / MW / 2); f[i] = Math.exp(-t * t) * (LW * LW + t * t); }
  const g = Float64Array.from({ length: M2 }, (_, i) => f[(i + MW) % M2]), a = new Float64Array(NW);
  for (let j = 1; j <= NW; j++) { let s = 0; for (let n = 0; n < M2; n++) s += g[n] * Math.cos(2 * Math.PI * j * n / M2); a[NW - j] = s / M2; }
  return a;                                    // highest power first
})();
// w(x + iy), y >= 0: [re, im] into out
export function faddeeva(x, y, out = [0, 0]) {
  const dr = LW + y, di = -x, nr = LW - y, ni = x, dd = dr * dr + di * di;
  const zr = (nr * dr + ni * di) / dd, zi = (ni * dr - nr * di) / dd;          // Z = (L + iz) / (L − iz)
  let pr = 0, pi = 0;
  for (let k = 0; k < NW; k++) { const r = pr * zr - pi * zi + WC[k]; pi = pr * zi + pi * zr; pr = r; }
  const d2r = dr * dr - di * di, d2i = 2 * dr * di, d2 = d2r * d2r + d2i * d2i;
  out[0] = 2 * (pr * d2r + pi * d2i) / d2 + dr / dd / SQPI;
  out[1] = 2 * (pi * d2r - pr * d2i) / d2 - di / dd / SQPI;
  return out;
}

// --------------------------------------------------------------------------
// one band's shape (height 1 at its centre) and its derivatives in c, w and the extra parameter
// --------------------------------------------------------------------------
const W1 = [0, 0], W0 = [0, 0];
function band(shape, u, w, e, out) {
  if (shape === "lorentz" || shape === "pv") {
    const t = 2 * u / w, d = 1 + t * t, L = 1 / d, Lc = 4 * t / (w * d * d), Lw = 2 * t * t / (w * d * d);
    if (shape === "lorentz") { out[0] = L; out[1] = Lc; out[2] = Lw; out[3] = 0; return out; }
    const v = u / w, G = Math.exp(-G4 * v * v), Gc = G * 2 * G4 * u / (w * w), Gw = G * 2 * G4 * u * u / (w * w * w);
    out[0] = e * L + (1 - e) * G; out[1] = e * Lc + (1 - e) * Gc; out[2] = e * Lw + (1 - e) * Gw; out[3] = L - G; return out;
  }
  if (shape === "gauss") {
    const v = u / w, G = Math.exp(-G4 * v * v);
    out[0] = G; out[1] = G * 2 * G4 * u / (w * w); out[2] = G * 2 * G4 * u * u / (w * w * w); out[3] = 0; return out;
  }
  if (shape === "bwf") {
    const q = e, x = 2 * u / w + q, a = 1 + q * x, den = 1 / (1 + x * x), g = a * a * den, s = 1 + q * q, dg = 2 * a * (q - x) * den * den;
    out[0] = g / s; out[1] = dg * (-2 / w) / s; out[2] = dg * (-(x - q) / w) / s; out[3] = ((2 * a * x * den + dg) * s - g * 2 * q) / (s * s); return out;
  }
  // Voigt: w is the Gaussian FWHM, e the Lorentzian FWHM
  const sg = w / S2L, ga = e / 2, k = 1 / (sg * Math.SQRT2);
  const zr = u * k, zi = ga * k;
  faddeeva(zr, zi, W1); faddeeva(0, zi, W0);
  const R = W1[0], R0 = W0[0], s = R / R0;
  // w'(z) = −2 z w + 2i/√π
  const dr = -2 * (zr * W1[0] - zi * W1[1]), di = -2 * (zr * W1[1] + zi * W1[0]) + 2 / SQPI;
  const d0r = -2 * (-zi * W0[1]), d0i = -2 * (zi * W0[0]) + 2 / SQPI;
  const Rc = -dr * k, Rg = -di * k, R0g = -d0i * k;                 // in c, in γ
  const Rs = -(dr * zr - di * zi) / sg, R0s = -(d0r * 0 - d0i * zi) / sg;   // in σ
  out[0] = s; out[1] = Rc / R0; out[2] = (Rs - s * R0s) / R0 / S2L; out[3] = (Rg - s * R0g) / R0 / 2;
  return out;
}

// the band's FWHM and its area (over all shifts; BWF over [lo, hi])
export function fwhmOf(shape, w, e) {
  if (shape === "voigt") return 0.5346 * e + Math.sqrt(0.2166 * e * e + w * w);
  if (shape === "bwf") return Math.abs(e) < 1 ? w * (1 + e * e) / (1 - e * e) : NaN;
  return w;
}
export function areaOf(shape, h, c, w, e, lo, hi) {
  if (shape === "gauss") return h * w * GAREA;
  if (shape === "lorentz") return Math.PI * h * w / 2;
  if (shape === "pv") return h * w * (e * Math.PI / 2 + (1 - e) * GAREA);
  if (shape === "voigt") { const sg = w / S2L; return h * sg * Math.sqrt(2 * Math.PI) / faddeeva(0, e / 2 / (sg * Math.SQRT2), W0)[0]; }
  const q = e, F = x => q * q * x + (1 - q * q) * Math.atan(x) + q * Math.log(1 + x * x);
  return h / (1 + q * q) * w / 2 * (F(2 * (hi - c) / w + q) - F(2 * (lo - c) / w + q));
}

// --------------------------------------------------------------------------
// a model: { range: [lo, hi], baseline: { kind: "poly", order } (fitted with the bands) or a
// background taken off first (spectra.js), despike, peaks: [{ name, shape, h, c, w, eta|wl|q }] with
// each parameter { v, lo, hi, mode: "free" | "fixed" | "link", to, rel: "equal" | "offset" | "ratio", k } }
// --------------------------------------------------------------------------
const bnd = (v, lo, hi) => ({ v, lo, hi, mode: "free" });
export function newPeak(shape, c, h, w, name) {
  const pk = { name: name || `Band ${Math.round(c)}`, shape, h: bnd(Math.max(h, 0), 0, Infinity), c: bnd(c, c - Math.max(w, 5), c + Math.max(w, 5)) };
  const wv = shape === "voigt" ? w / 1.6376 : w;
  pk.w = bnd(wv, Math.max(0.3, wv / 5), wv * 5);
  if (shape === "pv") pk.eta = bnd(0.5, 0, 1);
  if (shape === "voigt") pk.wl = bnd(wv, 0.05, wv * 5);
  if (shape === "bwf") pk.q = bnd(-0.1, -0.5, 0.5);
  return pk;
}
// the same band with another line shape, keeping its height, position and width
export function reshape(pk, shape) {
  const fw = fwhmOf(pk.shape, pk.w.v, pk.shape === "pv" ? pk.eta.v : pk.shape === "voigt" ? pk.wl.v : pk.shape === "bwf" ? pk.q.v : 0);
  const out = newPeak(shape, pk.c.v, pk.h.v, Number.isFinite(fw) ? fw : pk.w.v, pk.name);
  out.h = { ...pk.h }; out.c = { ...pk.c };
  return out;
}
export const MODEL_DEFAULTS = { range: [100, 3500], baseline: { kind: "poly", order: 1 }, despike: true, maxIter: 200, peaks: [] };

export function checkModel(model) {
  const [lo, hi] = model.range;
  if (!(lo < hi)) throw new Error("the fit range: the minimum must be below the maximum");
  if (!model.peaks.length) throw new Error("no bands: add one, or Find peaks");
  const names = model.peaks.map(p => p.name);
  if (new Set(names).size !== names.length) throw new Error("two bands share a name");
  model.peaks.forEach((pk, j) => {
    if (!SHAPES[pk.shape]) throw new Error(`${pk.name}: unknown line shape ${pk.shape}`);
    for (const k of PARAMS[pk.shape]) {
      const P = pk[k];
      if (!P || !Number.isFinite(P.v)) throw new Error(`${pk.name}: ${paramLabel(pk.shape, k)} has no value`);
      if (P.mode === "link") {
        const t = model.peaks[P.to];
        if (!t || P.to === j) throw new Error(`${pk.name}: ${paramLabel(pk.shape, k)} is linked to no other band`);
        if (!PARAMS[t.shape].includes(k)) throw new Error(`${pk.name}: ${t.name} has no ${paramLabel(pk.shape, k)} to link to`);
      } else if (P.mode !== "fixed" && !(P.lo <= P.hi)) throw new Error(`${pk.name}: ${paramLabel(pk.shape, k)}: the minimum is above the maximum`);
    }
  });
  return model;
}

// the free parameters x and how every band parameter follows from them: p = a x[s] + b
export function compile(model, wn) {
  checkModel(model);
  const [lo, hi] = model.range, idx = [];
  for (let c = 0; c < wn.length; c++) if (wn[c] >= lo && wn[c] <= hi) idx.push(c);
  idx.sort((a, b) => wn[a] - wn[b]);
  const N = idx.length, nu = Float64Array.from(idx, c => wn[c]);
  const poly = model.baseline.kind === "poly", nb = poly ? Math.max(0, (model.baseline.order ?? 1) + 1) : 0;
  const peaks = model.peaks.map(pk => ({ name: pk.name, shape: pk.shape, keys: PARAMS[pk.shape] }));
  const xs = [], map = new Map();               // map: "j.k" -> { s, a, b }
  for (let k = 0; k < nb; k++) xs.push({ kind: "base", j: -1, key: `b${k}`, lo: -Infinity, hi: Infinity, v: 0 });
  const resolve = (j, k, seen = new Set()) => {
    const id = `${j}.${k}`;
    if (map.has(id)) return map.get(id);
    if (seen.has(id)) throw new Error(`${model.peaks[j].name}: its links go round in a circle`);
    seen.add(id);
    const P = model.peaks[j][k];
    let r;
    if (P.mode === "fixed") r = { s: -1, a: 0, b: P.v };
    else if (P.mode === "link") {
      const t = resolve(P.to, k, seen), kk = P.k ?? (P.rel === "ratio" ? 1 : 0);
      r = P.rel === "offset" ? { s: t.s, a: t.a, b: t.b + kk } : P.rel === "ratio" ? { s: t.s, a: kk * t.a, b: kk * t.b } : { ...t };
    } else { xs.push({ kind: k, j, key: id, lo: P.lo, hi: P.hi, v: Math.min(P.hi, Math.max(P.lo, P.v)) }); r = { s: xs.length - 1, a: 1, b: 0 }; }
    map.set(id, r);
    return r;
  };
  peaks.forEach((pk, j) => { pk.map = pk.keys.map(k => resolve(j, k)); });
  const nx = xs.length;
  if (N < nx + 3) throw new Error(`only ${N} channels in the fit range for ${nx} free parameters: widen the range`);
  const C = { model, N, nu, idx, nb, peaks, xs, nx, lo: Float64Array.from(xs, x => x.lo), hi: Float64Array.from(xs, x => x.hi),
    x0: Float64Array.from(xs, x => x.v), mid: (nu[0] + nu[N - 1]) / 2, half: (nu[N - 1] - nu[0]) / 2 || 1, range: [nu[0], nu[N - 1]],
    linear: Uint8Array.from(xs, x => (x.kind === "base" || x.kind === "h" ? 1 : 0)) };
  C.t = Float64Array.from(nu, v => (v - C.mid) / C.half);
  return C;
}

// each band's parameters from x: [[h, c, w, e], ...]
export function bandParams(C, x) {
  return C.peaks.map(pk => {
    const v = pk.map.map(m => (m.s < 0 ? m.b : m.a * x[m.s] + m.b));
    return [v[0], v[1], v[2], v.length > 3 ? v[3] : 0];
  });
}

// the model at every channel into f, and its Jacobian in x (N x nx, row-major) when J
const OUT4 = new Float64Array(4);
export function evaluate(C, x, f, J) {
  const { N, nu, nb, nx, t } = C;
  f.fill(0);
  if (J) J.fill(0);
  for (let n = 0; n < N; n++) {
    let v = 1, s = 0;
    for (let k = 0; k < nb; k++) { s += x[k] * v; if (J) J[n * nx + k] = v; v *= t[n]; }
    f[n] = s;
  }
  for (const pk of C.peaks) {
    const m = pk.map, val = k => (m[k].s < 0 ? m[k].b : m[k].a * x[m[k].s] + m[k].b);
    const h = val(0), c = val(1), w = val(2), e = m.length > 3 ? val(3) : 0;
    for (let n = 0; n < N; n++) {
      band(pk.shape, nu[n] - c, w, e, OUT4);
      f[n] += h * OUT4[0];
      if (!J) continue;
      const row = n * nx;
      if (m[0].s >= 0) J[row + m[0].s] += m[0].a * OUT4[0];
      if (m[1].s >= 0) J[row + m[1].s] += m[1].a * h * OUT4[1];
      if (m[2].s >= 0) J[row + m[2].s] += m[2].a * h * OUT4[2];
      if (m.length > 3 && m[3].s >= 0) J[row + m[3].s] += m[3].a * h * OUT4[3];
    }
  }
  return f;
}

// --------------------------------------------------------------------------
// the spectrum the model sees: cut to the range, cosmic rays off, a background taken off first
// --------------------------------------------------------------------------
export function prepareY(C, yCut) {
  let y = Float64Array.from(yCut);
  if (C.model.despike) y = despikeSpectrum(y).y;
  const b = C.model.baseline;
  if (b.kind !== "poly" && b.kind !== "none") {
    const base = background(C.nu, y, b);
    return { y: Float64Array.from(y, (v, n) => v - base[n]), base };
  }
  return { y, base: null };
}
export function cut(C, spectrum) { return Float64Array.from(C.idx, c => spectrum[c]); }

// --------------------------------------------------------------------------
// the fit
// --------------------------------------------------------------------------
function chol(A, b, n) {
  const L = new Float64Array(n * n), x = new Float64Array(n);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i * n + j];
    for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
    if (i === j) { if (!(s > 0)) return null; L[i * n + i] = Math.sqrt(s); } else L[i * n + j] = s / L[j * n + j];
  }
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * x[k]; x[i] = s / L[i * n + i]; }
  for (let i = n - 1; i >= 0; i--) { let s = x[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
  return x;
}
const clampX = (C, x) => { for (let i = 0; i < C.nx; i++) x[i] = Math.min(C.hi[i], Math.max(C.lo[i], x[i])); return x; };

// the background and the heights for the band shapes in x: linear least squares (bounds clipped)
export function solveLinear(C, y, x0) {
  const { N, nx, linear } = C, x = Float64Array.from(x0), lin = [];
  for (let i = 0; i < nx; i++) if (linear[i]) { lin.push(i); x[i] = 0; }
  if (!lin.length) return x;
  const f = new Float64Array(N), J = new Float64Array(N * nx);
  evaluate(C, x, f, J);
  const m = lin.length, A = new Float64Array(m * m), b = new Float64Array(m);
  for (let n = 0; n < N; n++) {
    const r = y[n] - f[n];
    for (let a = 0; a < m; a++) { const ja = J[n * nx + lin[a]]; b[a] += ja * r; for (let c = 0; c <= a; c++) A[a * m + c] += ja * J[n * nx + lin[c]]; }
  }
  let dmax = 0;
  for (let a = 0; a < m; a++) { for (let c = a + 1; c < m; c++) A[a * m + c] = A[c * m + a]; dmax = Math.max(dmax, A[a * m + a]); }
  for (let a = 0; a < m; a++) A[a * m + a] += 1e-12 * dmax + 1e-300;
  const z = chol(A, b, m);
  if (z) lin.forEach((i, a) => { x[i] = z[a]; });
  else lin.forEach(i => { x[i] = x0[i]; });
  return clampX(C, x);
}

// bounded Levenberg-Marquardt from x0; the covariance of the parameters not on a bound
export function lm(C, y, x0, maxIter = C.model.maxIter || 200) {
  const { N, nx, lo, hi } = C, x = clampX(C, Float64Array.from(x0));
  const f = new Float64Array(N), J = new Float64Array(N * nx), r = new Float64Array(N), trial = new Float64Array(nx);
  const A = new Float64Array(nx * nx), g = new Float64Array(nx), M = new Float64Array(nx * nx);
  const ssrOf = v => { let s = 0; for (let n = 0; n < N; n++) { const d = y[n] - v[n]; s += d * d; } return s; };
  evaluate(C, x, f, J);
  let ssr = ssrOf(f), lambda = 1e-3, it = 0;
  for (; it < maxIter; it++) {
    for (let n = 0; n < N; n++) r[n] = y[n] - f[n];
    A.fill(0); g.fill(0);
    for (let n = 0; n < N; n++) {
      const row = n * nx;
      for (let i = 0; i < nx; i++) { const ji = J[row + i]; if (ji === 0) continue; g[i] += ji * r[n]; for (let j = 0; j <= i; j++) A[i * nx + j] += ji * J[row + j]; }
    }
    // a parameter on a bound with the gradient pushing outwards stays there this step
    const act = [];
    for (let i = 0; i < nx; i++) { if ((x[i] <= lo[i] && g[i] <= 0) || (x[i] >= hi[i] && g[i] >= 0)) continue; act.push(i); }
    if (!act.length) break;
    const m = act.length;
    let dmax = 0;
    for (let a = 0; a < m; a++) dmax = Math.max(dmax, A[act[a] * nx + act[a]]);
    let accepted = false, gain = 0;
    for (let tries = 0; tries < 12 && !accepted; tries++) {
      for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) { const i = Math.max(act[a], act[b]), j = Math.min(act[a], act[b]); M[a * m + b] = A[i * nx + j]; }
      for (let a = 0; a < m; a++) M[a * m + a] += lambda * Math.max(A[act[a] * nx + act[a]], 1e-12 * dmax) + 1e-300;
      const d = chol(M.subarray(0, m * m), Float64Array.from(act, i => g[i]), m);
      if (!d) { lambda *= 10; continue; }
      trial.set(x); act.forEach((i, a) => { trial[i] += d[a]; });
      clampX(C, trial);
      evaluate(C, trial, r, null);
      const s2 = ssrOf(r);
      if (s2 < ssr) { gain = (ssr - s2) / (ssr || 1); x.set(trial); ssr = s2; lambda = Math.max(lambda / 3, 1e-12); accepted = true; }
      else lambda *= 4;
    }
    evaluate(C, x, f, J);
    if (!accepted || gain < 1e-10) break;
  }
  const onB = new Uint8Array(nx), act = [];
  for (let i = 0; i < nx; i++) {
    const span = Number.isFinite(hi[i] - lo[i]) ? hi[i] - lo[i] : Math.max(1, Math.abs(x[i]));
    if ((Number.isFinite(lo[i]) && x[i] - lo[i] <= 1e-6 * span) || (Number.isFinite(hi[i]) && hi[i] - x[i] <= 1e-6 * span)) onB[i] = 1; else act.push(i);
  }
  const na = act.length, AA = new Float64Array(na * na);
  for (let n = 0; n < N; n++) { const row = n * nx; for (let a = 0; a < na; a++) { const ja = J[row + act[a]]; if (ja) for (let b = 0; b < na; b++) AA[a * na + b] += ja * J[row + act[b]]; } }
  const inv = pinvSym(AA, na), s2 = ssr / Math.max(1, N - na), cov = new Float64Array(nx * nx);
  for (let a = 0; a < na; a++) for (let b = 0; b < na; b++) cov[act[a] * nx + act[b]] = s2 * inv[a * na + b];
  return { x, ssr, iter: it, cov, onB };
}

// what one fit gives: per band [h, c, fwhm, area, extra] and their standard errors, and the fit's
// quality (residual over the noise, R², reduced χ² at the noise)
export function derive(C, fit, y) {
  const { nx, peaks } = C, P = bandParams(C, fit.x), vals = [], ses = [];
  const varOf = (mi, mj) => (mi.s < 0 || mj.s < 0 ? 0 : mi.a * mj.a * fit.cov[mi.s * nx + mj.s]);
  peaks.forEach((pk, j) => {
    const [h, c, w, e] = P[j], m = pk.map, fns = [
      p => p[0], p => p[1], p => fwhmOf(pk.shape, p[2], p[3]), p => areaOf(pk.shape, p[0], p[1], p[2], p[3], C.range[0], C.range[1]), p => p[3]];
    const v = fns.map(fn => fn([h, c, w, e])), s = fns.map(fn => {
      // the delta method on the band's own parameters (numerical gradient)
      const g = [0, 0, 0, 0], p0 = [h, c, w, e];
      for (let k = 0; k < m.length; k++) {
        const d = 1e-6 * Math.max(Math.abs(p0[k]), 1e-3), up = p0.slice(), dn = p0.slice();
        up[k] += d; dn[k] -= d;
        g[k] = (fn(up) - fn(dn)) / (2 * d);
      }
      let q = 0;
      for (let a = 0; a < m.length; a++) for (let b = 0; b < m.length; b++) if (g[a] && g[b]) q += g[a] * g[b] * varOf(m[a], m[b]);
      return q > 0 ? Math.sqrt(q) : 0;
    });
    if (m.length <= 3) { v[4] = NaN; s[4] = NaN; }
    vals.push(v); ses.push(s);
  });
  const N = C.N, noise = noiseSigma(y) || 1e-9;
  let mean = 0; for (const v of y) mean += v / N;
  let tot = 0; for (const v of y) tot += (v - mean) ** 2;
  const nfree = fit.onB.reduce((t, b) => t + (b ? 0 : 1), 0);
  return { vals, ses, noise, rn: Math.sqrt(fit.ssr / N) / noise, r2: tot > 0 ? 1 - fit.ssr / tot : NaN, chi2: fit.ssr / Math.max(1, N - nfree) / (noise * noise) };
}

// the fit drawn: background and each band at every channel
export function curves(C, x, preBase = null) {
  const N = C.N, base = new Float64Array(N), total = new Float64Array(N), P = bandParams(C, x);
  for (let n = 0; n < N; n++) { let v = 1, s = 0; for (let k = 0; k < C.nb; k++) { s += x[k] * v; v *= C.t[n]; } base[n] = s + (preBase ? preBase[n] : 0); total[n] = base[n]; }
  const comps = C.peaks.map((pk, j) => {
    const [h, c, w, e] = P[j], y = new Float64Array(N);
    for (let n = 0; n < N; n++) { y[n] = h * band(pk.shape, C.nu[n] - c, w, e, OUT4)[0]; total[n] += y[n]; }
    return { name: pk.name, y, A: h, pos: c, width: fwhmOf(pk.shape, w, e) };
  });
  return { baseline: base, comps, total };
}

// one spectrum (already cut to the range): from x0 (the model's values unless given), its
// background and heights solved first; status 1 fitted, 2 a free position or width on a bound
export function fitOne(C, yCut, x0 = null) {
  const { y, base } = prepareY(C, yCut);
  const fit = lm(C, y, solveLinear(C, y, x0 || C.x0));
  if (!fit.x.every(Number.isFinite) || !Number.isFinite(fit.ssr)) return { status: 3, y, base };
  const bound = C.xs.some((v, i) => fit.onB[i] && (v.kind === "c" || v.kind === "w" || v.kind === "wl"));
  return { status: bound ? 2 : 1, fit, y, base, D: derive(C, fit, y) };
}

// many spectra (a worker's chunk; Y: n x Ncut): flat outputs
export function fitMany({ model, wn, x0 }, Y) {
  const C = compile(model, wn), N = C.N, n = Y.length / N, nP = C.peaks.length, nO = OUT.length;
  const vals = new Float64Array(n * nP * nO).fill(NaN), ses = new Float64Array(n * nP * nO).fill(NaN), xs = new Float64Array(n * C.nx).fill(NaN);
  const stats = new Float64Array(n * 4).fill(NaN), status = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const y = Y.subarray(i * N, (i + 1) * N);
    let ok = true, any = false;
    for (const v of y) { if (!Number.isFinite(v)) { ok = false; break; } if (v !== 0) any = true; }
    if (!ok || !any) { status[i] = 0; continue; }
    const r = fitOne(C, y, x0);
    status[i] = r.status;
    if (r.status === 3) continue;
    xs.set(r.fit.x, i * C.nx);
    r.D.vals.forEach((v, j) => { vals.set(v, (i * nP + j) * nO); ses.set(r.D.ses[j], (i * nP + j) * nO); });
    stats.set([r.D.rn, r.D.r2, r.D.chi2, r.fit.iter], i * 4);
  }
  return { vals, ses, xs, stats, status };
}

// --------------------------------------------------------------------------
// peak search: maxima of the smoothed spectrum that stand out by their prominence
// --------------------------------------------------------------------------
export function findPeaks(nu, y, { prominence = 5, maxPeaks = 12, minSep = 0, window = 7 } = {}) {
  const N = y.length, sigma = noiseSigma(y) || 1e-9, ys = N >= window ? savgol(y, window, 2) : Float64Array.from(y), found = [];
  for (let i = 1; i < N - 1; i++) {
    if (!(ys[i] > ys[i - 1] && ys[i] >= ys[i + 1])) continue;
    let l = i, lmin = ys[i]; while (l > 0 && ys[l - 1] <= ys[i]) { l--; lmin = Math.min(lmin, ys[l]); }
    let r = i, rmin = ys[i]; while (r < N - 1 && ys[r + 1] <= ys[i]) { r++; rmin = Math.min(rmin, ys[r]); }
    const prom = ys[i] - Math.max(lmin, rmin);
    if (prom < prominence * sigma) continue;
    // the width at half the prominence, interpolated between channels
    const half = ys[i] - prom / 2;
    let a = i; while (a > 0 && ys[a] > half) a--;
    let b = i; while (b < N - 1 && ys[b] > half) b++;
    const xa = ys[a] <= half && a < i ? nu[a] + (half - ys[a]) / ((ys[a + 1] - ys[a]) || 1) * (nu[a + 1] - nu[a]) : nu[a];
    const xb = ys[b] <= half && b > i ? nu[b] - (half - ys[b]) / ((ys[b - 1] - ys[b]) || 1) * (nu[b] - nu[b - 1]) : nu[b];
    found.push({ c: nu[i], h: prom, w: Math.max(xb - xa, 2 * Math.abs(nu[Math.min(N - 1, i + 1)] - nu[i])), prominence: prom / sigma });
  }
  found.sort((p, q) => q.h - p.h);
  const keep = [];
  for (const p of found) { if (keep.length >= maxPeaks) break; if (keep.every(q => Math.abs(q.c - p.c) >= Math.max(minSep, 0.5 * Math.min(p.w, q.w)))) keep.push(p); }
  return keep.sort((p, q) => p.c - q.c);
}
