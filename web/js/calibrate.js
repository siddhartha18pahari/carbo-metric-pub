// Raman-shift axis calibration, as LabSpec's: the bands of a reference material measured on the
// instrument are fitted, and the axis correction Δ(ω) = true − measured is fitted to them as an
// offset, a line or a parabola in t = (ω − mid) / 1000:
//
//   Δ(ω) = shift + stretch t + curve t²
//
// then applied to every spectrum's axis (the Toolbox step "Correct the axis"). Reference shifts:
// silicon 520.7 cm⁻¹; polystyrene and cyclohexane from ASTM E1840 (Standard Guide for Raman Shift
// Standards for Spectrometer Calibration).
// Pure functions; the band fits are peakfit.js models.

import { newPeak, MODEL_DEFAULTS, compile, cut, fitOne } from "./peakfit.js";
export { correctAxis } from "./spectra.js";

export const STANDARDS = {
  si: { label: "Silicon (520.7)", bands: [520.7] },
  ps: { label: "Polystyrene (ASTM E1840)", bands: [620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5, 1583.1, 1602.3, 2852.4, 2904.5, 3054.3] },
  chx: { label: "Cyclohexane (ASTM E1840)", bands: [384.1, 426.3, 801.3, 1028.3, 1157.6, 1266.4, 1444.4, 2664.4, 2852.9, 2923.8, 2938.3] },
  custom: { label: "Your own bands", bands: [] },
};
export const CAL_DEFAULTS = { standard: "si", custom: "", order: "auto", search: 8, half: 15, snr: 10 };
export const ORDERS = { auto: "Automatic", 0: "Offset", 1: "Line", 2: "Parabola" };

// the reference bands the spectrum covers, each fitted with its neighbours (bands closer than a
// window are fitted together); a band counts when its height is snr × its error and its position
// is off its limits. Each position may move ± search, less where a neighbour is nearer.
export function fitBands(nu, y, refs, { search = 8, half = 15, snr = 10 } = {}) {
  const lo = nu[0] + half, hi = nu[nu.length - 1] - half, inside = refs.filter(r => r > lo && r < hi).sort((a, b) => a - b);
  const groups = [];
  for (const r of inside) { const g = groups[groups.length - 1]; if (g && r - g[g.length - 1] < 2 * half + search) g.push(r); else groups.push([r]); }
  const out = refs.filter(r => !(r > lo && r < hi)).map(ref => ({ ref, used: false, why: "outside the spectrum" }));
  for (const g of groups) {
    const peaks = g.map((ref, i) => {
      const room = Math.min(search, i ? (ref - g[i - 1]) / 2 : search, i < g.length - 1 ? (g[i + 1] - ref) / 2 : search);
      // started at the highest channel within reach
      let c = ref, top = -Infinity;
      for (let n = 0; n < nu.length; n++) if (Math.abs(nu[n] - ref) <= room && y[n] > top) { top = y[n]; c = nu[n]; }
      const p = newPeak("lorentz", c, 1, 6, String(ref));
      p.c.lo = ref - room; p.c.hi = ref + room; p.w.lo = 1; p.w.hi = 40;
      return p;
    });
    const model = { ...structuredClone(MODEL_DEFAULTS), range: [g[0] - half - search, g[g.length - 1] + half + search], baseline: { kind: "poly", order: 1 }, despike: true, peaks };
    let r = null;
    try { const C = compile(model, nu); r = fitOne(C, cut(C, y)); } catch { r = null; }
    g.forEach((ref, j) => {
      if (!r || r.status === 3) { out.push({ ref, used: false, why: "the fit failed" }); return; }
      const [h, c, fw] = r.D.vals[j], [sh, sc] = r.D.ses[j], lim = peaks[j].c;
      const onLimit = c - lim.lo < 0.01 || lim.hi - c < 0.01, strong = h > 0 && h > snr * (sh || 0);
      out.push({ ref, pos: c, se: sc, h, fwhm: fw, used: strong && !onLimit, why: !strong ? `below ${snr} × its error` : onLimit ? "on its position limit" : "" });
    });
  }
  return out.sort((a, b) => a.ref - b.ref);
}

// the correction fitted to the bands used, by least squares; order "auto": an offset for one or two
// bands, a line for three or more
export function fitCorrection(bands, order = "auto") {
  const B = bands.filter(b => b.used), n = B.length;
  if (!n) throw new Error("no reference band was found in the spectrum");
  let k = order === "auto" ? (n >= 3 ? 1 : 0) : +order;
  k = Math.min(k, n - 1);
  const mid = B.reduce((t, b) => t + b.pos, 0) / n, t = B.map(b => (b.pos - mid) / 1000), d = B.map(b => b.ref - b.pos), m = k + 1;
  // normal equations (at most 3 × 3)
  const A = Array.from({ length: m }, () => new Float64Array(m)), v = new Float64Array(m);
  for (let i = 0; i < n; i++) for (let a = 0; a < m; a++) { v[a] += t[i] ** a * d[i]; for (let b = 0; b < m; b++) A[a][b] += t[i] ** (a + b); }
  const inv = invert(A), c = Array.from({ length: m }, (_, a) => inv[a].reduce((s, x, b) => s + x * v[b], 0));
  const resid = B.map((b, i) => d[i] - c.reduce((s, x, a) => s + x * t[i] ** a, 0)), dof = n - m;
  const s2 = dof > 0 ? resid.reduce((s, r) => s + r * r, 0) / dof : NaN;
  const se = c.map((_, a) => (dof > 0 ? Math.sqrt(s2 * inv[a][a]) : NaN));
  const coef = { shift: c[0], stretch: c[1] || 0, curve: c[2] || 0, mid };
  return { order: k, n, coef, se, rms: Math.sqrt(resid.reduce((s, r) => s + r * r, 0) / n), dof, resid: Object.fromEntries(B.map((b, i) => [b.ref, resid[i]])) };
}
function invert(A) {
  const m = A.length, M = A.map((r, i) => [...r, ...Array.from({ length: m }, (_, j) => (i === j ? 1 : 0))]);
  for (let i = 0; i < m; i++) {
    let p = i; for (let r = i + 1; r < m; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
    [M[i], M[p]] = [M[p], M[i]];
    const d = M[i][i]; if (!d) throw new Error("the reference bands do not determine the correction");
    for (let j = 0; j < 2 * m; j++) M[i][j] /= d;
    for (let r = 0; r < m; r++) if (r !== i) { const f = M[r][i]; for (let j = 0; j < 2 * m; j++) M[r][j] -= f * M[i][j]; }
  }
  return M.map(r => r.slice(m));
}
// the reference list of a standard (custom: numbers in any separator)
export function referenceBands(standard, custom = "") {
  if (standard !== "custom") return STANDARDS[standard].bands.slice();
  return (custom.match(/-?\d+(?:\.\d+)?/g) || []).map(Number).filter(v => v > 0).sort((a, b) => a - b);
}
