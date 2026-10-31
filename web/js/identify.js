// Identification against a library of reference spectra, as LabSpec's IDFinder or KnowItAll's
// search: every spectrum (prepared as on the Layers and MVA tabs: range, cosmic rays, binning,
// polynomial background) compared with every reference treated the same way, by the hit quality
// index HQI = 100 r², r the correlation over the channels the reference covers (Pearson, so an
// offset and a scale do not matter). The first derivative may be compared instead, which weighs
// band shapes and ignores what is left of a background. A mixture search fits the spectrum as two
// references (non-negative, plus an offset) and keeps the pair when it explains clearly more.
// Pure functions (no DOM), so they run under Node for the tests and in a worker.

import { prepRef } from "./layers.js";

export const ID_DEFAULTS = { range: [150, 3600], baseline: 3, deriv: false, minHQI: 80, mixGain: 3 };

// each reference on the prepared channels: its values where it covers them, mean-removed and unit
// length for the correlation (derivative: differences of neighbouring covered channels)
export function refVectors(refs, prep, deriv = false) {
  return refs.map(ref => {
    const p = prepRef(ref, prep), nb = p.y.length, idx = [], val = [];
    if (deriv) { for (let j = 0; j + 1 < nb; j++) if (Number.isFinite(p.y[j]) && Number.isFinite(p.y[j + 1])) { idx.push(j); val.push(p.y[j + 1] - p.y[j]); } }
    else for (let j = 0; j < nb; j++) if (Number.isFinite(p.y[j])) { idx.push(j); val.push(p.y[j]); }
    const n = val.length, m = n ? val.reduce((a, b) => a + b, 0) / n : 0;
    let ss = 0; for (const v of val) ss += (v - m) ** 2;
    const u = Float64Array.from(val, v => (ss > 0 ? (v - m) / Math.sqrt(ss) : 0));
    return { name: ref.name, idx: Int32Array.from(idx), u, y: p.y, cover: p.cover, ok: p.cover >= 0.5 && ss > 0 };
  });
}
// the spectrum's value at a reference channel (the derivative: the difference to the next channel)
const at = (x, j, deriv) => (deriv ? x[j + 1] - x[j] : x[j]);
// HQI of one spectrum x (prepared channels) against one reference vector
export function hqi(x, R, deriv = false) {
  if (!R.ok) return NaN;
  const { idx, u } = R, n = idx.length;
  let sxu = 0, sx = 0, sxx = 0;
  for (let k = 0; k < n; k++) { const v = at(x, idx[k], deriv); sxu += v * u[k]; sx += v; sxx += v * v; }
  const var_ = sxx - sx * sx / n, r = var_ > 0 ? sxu / Math.sqrt(var_) : 0;
  return r > 0 ? 100 * r * r : 0;
}
// the references ranked for one spectrum
export function hits(x, refs, deriv = false) {
  return refs.map((R, r) => ({ r, name: R.name, hqi: hqi(x, R, deriv) })).filter(h => Number.isFinite(h.hqi)).sort((a, b) => b.hqi - a.hqi);
}
// every prepared spectrum's best two references (rows of prep.V)
export function searchAll(prep, refs, deriv = false, onProgress = null) {
  const { n, nb, V } = prep, best = new Int16Array(n).fill(-1), second = new Int16Array(n).fill(-1), h1 = new Float32Array(n), h2 = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = V.subarray(i * nb, (i + 1) * nb);
    let b1 = -1, b2 = -1, v1 = -1, v2 = -1;
    for (let r = 0; r < refs.length; r++) {
      const h = hqi(x, refs[r], deriv);
      if (!(h >= 0)) continue;
      if (h > v1) { b2 = b1; v2 = v1; b1 = r; v1 = h; } else if (h > v2) { b2 = r; v2 = h; }
    }
    best[i] = b1; second[i] = b2; h1[i] = Math.max(v1, 0); h2[i] = Math.max(v2, 0);
    if (onProgress && i % 2000 === 1999) onProgress(i + 1, n);
  }
  return { best, second, hqi: h1, hqi2: h2 };
}

// x as a·ref i + b·ref j + c (a, b ≥ 0) over the channels both cover, against the best single one:
// the pair with the highest HQI of the fitted sum, each reference's share of the fitted signal
export function mixture(x, refs, cand) {
  let best = null;
  for (let p = 0; p < cand.length; p++) for (let q = p + 1; q < cand.length; q++) {
    const A = refs[cand[p]], B = refs[cand[q]], idx = [];
    for (let j = 0; j < x.length; j++) if (Number.isFinite(A.y[j]) && Number.isFinite(B.y[j])) idx.push(j);
    if (idx.length < 10) continue;
    const fit = cols => {
      // least squares on the columns (references and a constant)
      const m = cols.length + 1, M = new Float64Array(m * m), v = new Float64Array(m);
      const col = (c, j) => (c < cols.length ? cols[c].y[j] : 1);
      for (const j of idx) for (let a = 0; a < m; a++) { v[a] += col(a, j) * x[j]; for (let b = 0; b < m; b++) M[a * m + b] += col(a, j) * col(b, j); }
      const z = solve(M, v, m);
      return z && z.slice(0, cols.length).every(c => c >= 0) ? z : null;
    };
    const z = fit([A, B]);
    if (!z) continue;
    let sx = 0, sf = 0, sxx = 0, sff = 0, sxf = 0;
    for (const j of idx) { const f = z[0] * A.y[j] + z[1] * B.y[j] + z[2]; sx += x[j]; sf += f; sxx += x[j] * x[j]; sff += f * f; sxf += x[j] * f; }
    const n = idx.length, r = (sxf - sx * sf / n) / Math.sqrt(Math.max((sxx - sx * sx / n) * (sff - sf * sf / n), 1e-300)), h = r > 0 ? 100 * r * r : 0;
    if (!best || h > best.hqi) best = { a: cand[p], b: cand[q], hqi: h, share: z[0] / (z[0] + z[1] || 1) };
  }
  return best;
}
function solve(M, v, m) {
  const A = Float64Array.from(M), b = Float64Array.from(v);
  for (let c = 0; c < m; c++) {
    let piv = c; for (let r = c + 1; r < m; r++) if (Math.abs(A[r * m + c]) > Math.abs(A[piv * m + c])) piv = r;
    if (!(Math.abs(A[piv * m + c]) > 1e-300)) return null;
    if (piv !== c) { for (let k = 0; k < m; k++) [A[c * m + k], A[piv * m + k]] = [A[piv * m + k], A[c * m + k]]; [b[c], b[piv]] = [b[piv], b[c]]; }
    for (let r = c + 1; r < m; r++) { const f = A[r * m + c] / A[c * m + c]; for (let k = c; k < m; k++) A[r * m + k] -= f * A[c * m + k]; b[r] -= f * b[c]; }
  }
  const z = new Float64Array(m);
  for (let i = m - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < m; k++) s -= A[i * m + k] * z[k]; z[i] = s / A[i * m + i]; }
  return z;
}
