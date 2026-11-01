// Carbon nanotubes: every spectrum fitted in up to three windows, then read as:
//
//   RBM       radial breathing modes (100 to 400 cm⁻¹), one Lorentzian for each band the peak search
//             finds in the mean spectrum; each band's diameter from ω = A / d + B (d in nm):
//             A = 248, B = 0 for isolated tubes on SiO2 (Jorio et al. 2001, Phys. Rev. Lett. 86, 1118),
//             A = 234, B = 10 for bundles (Milnera et al. 2000, Phys. Rev. Lett. 84, 1324),
//             A = 223.5, B = 12.5 for HiPco tubes in SDS (Bachilo et al. 2002, Science 298, 2361).
//             Only tubes in resonance with the laser show, so these are the diameters this laser sees.
//   D and G   Lorentzian D, G⁺ and D′, and G⁻ as a Breit-Wigner-Fano line: G⁻ is Lorentzian in
//             semiconducting tubes and broadened to a BWF in metallic ones (Brown et al. 2001, Phys.
//             Rev. B 63, 155414), so a coupling 1/Q clearly below zero marks metallic tubes. The
//             G⁺/G⁻ splitting gives a diameter, ωG⁻ = ωG⁺ − C / d², C = 47.7 cm⁻¹ nm² for
//             semiconducting and 79.5 for metallic tubes (Jorio et al. 2002, Phys. Rev. B 65, 155412).
//             I(G)/I(D), the G band's maximum (G⁺ and G⁻ together) over D's height, is the usual
//             purity measure of single-wall tubes; I(D)/I(G) is reported for multi-wall ones.
//             The window is also fitted with one Lorentzian G (multi-wall tubes, other carbons), and
//             each spectrum keeps the split model only where it lowers the BIC by 10 or more.
//   2D (G′)   one Lorentzian: its position and I(2D)/I(G).
// Pure functions; the fits are peakfit.js models.

import { newPeak, MODEL_DEFAULTS, findPeaks } from "./peakfit.js";
import { background } from "./spectra.js";

export const RBM_RELATIONS = {
  isolated: { label: "Isolated, on SiO₂ (248 / d)", A: 248, B: 0 },
  bundles: { label: "Bundles (234 / d + 10)", A: 234, B: 10 },
  sds: { label: "In SDS (223.5 / d + 12.5)", A: 223.5, B: 12.5 },
  custom: { label: "Your own A / d + B", A: 248, B: 0 },
};
export const CNT_DEFAULTS = {
  laser: 532,
  winR: [100, 400], winG: [1200, 1700], win2: [2500, 2850],
  relation: "isolated", A: 248, B: 0,
  cS: 47.7, cM: 79.5,                       // G⁺/G⁻ splitting constants, cm⁻¹ nm²
  snr: 5,                                   // a band counts when its height is this many times its error
  prom: 6,                                  // RBM search: prominence over the noise
  qMin: 0.05,                               // metallic: 1/Q below −qMin and 3 errors below zero
};
export const TYPES = ["Metallic G⁻", "Semiconducting G⁻", "One G band", "No G band"];

const lor = (name, c, h, w, cr, wr) => {
  const p = newPeak("lorentz", c, h, w, name);
  p.c.lo = c - cr; p.c.hi = c + cr; p.w.lo = wr[0]; p.w.hi = wr[1];
  return p;
};
// the RBM bands for a fit: the maxima of the mean spectrum in the RBM window, searched with a SNIP
// background (40 cm⁻¹ wide) taken off, since the laser filter's edge tilts this window steeply
export function rbmPeaks(nu, y, s) {
  const idx = []; for (let n = 0; n < nu.length; n++) if (nu[n] >= s.winR[0] && nu[n] <= s.winR[1]) idx.push(n);
  if (idx.length < 15) return [];
  const x = Float64Array.from(idx, n => nu[n]), v = Float64Array.from(idx, n => y[n]);
  const step = (x[x.length - 1] - x[0]) / (x.length - 1), b = background(x, v, { kind: "snip", width: Math.max(6, Math.min(60, Math.round(40 / step))) });
  return findPeaks(x, Float64Array.from(v, (t, n) => t - b[n]), { prominence: s.prom, maxPeaks: 12, minSep: 4 }).filter(p => p.c > s.winR[0] + 5 && p.c < s.winR[1] - 5);
}
// the three models; rbm = the bands rbmPeaks found (no RBM model without them)
export function models(s, rbm = []) {
  const eV = 1239.84 / s.laser, d = 1350 - 50 * (eV - 2.33), d2 = 2680 - 100 * (eV - 2.33);
  const base = { ...structuredClone(MODEL_DEFAULTS), baseline: { kind: "poly", order: 1 }, despike: true };
  const gm = newPeak("bwf", 1570, 200, 25, "G⁻");
  gm.c.lo = 1520; gm.c.hi = 1584; gm.w.lo = 4; gm.w.hi = 80; gm.q = { v: 0, lo: -0.5, hi: 0.3, mode: "free" };
  const out = {
    mG: { ...base, range: [...s.winG], peaks: [lor("D", d, 50, 30, 40, [8, 120]), gm, lor("G⁺", 1591, 1000, 12, 12, [4, 40]), lor("D′", 1620, 20, 10, 15, [4, 40])] },
    mG1: { ...base, range: [...s.winG], peaks: [lor("D", d, 50, 30, 40, [8, 120]), lor("G", 1584, 1000, 20, 25, [4, 120]), lor("D′", 1620, 20, 10, 15, [4, 40])] },
    m2: { ...base, range: [...s.win2], peaks: [lor("2D", d2, 300, 50, 90, [12, 150])] },
    mR: null,
  };
  if (rbm.length) {
    // a degree-5 background for the filter edge's tail; no cosmic-ray step, as an RBM on a coarse axis
    // can be as narrow as a cosmic ray
    out.mR = { ...base, range: [...s.winR], baseline: { kind: "poly", order: 5 }, despike: false,
      peaks: rbm.map(p => lor(`RBM ${Math.round(p.c)}`, p.c, p.h, Math.min(Math.max(p.w, 3), 15), 4, [1.5, 25])) };
  }
  return out;
}
export const diameterOf = (w, s) => (w > s.B ? s.A / (w - s.B) : NaN);

// what each spectrum gives (the keys of the maps, the CSV and the report)
export const QUANT = [
  { key: "type", label: "G⁻ line shape", unit: "" },
  { key: "nrbm", label: "RBM bands", unit: "" },
  { key: "prbm", label: "Strongest RBM", unit: "cm⁻¹" },
  { key: "drbm", label: "Its diameter", unit: "nm" },
  { key: "dmean", label: "Mean RBM diameter", unit: "nm" },
  { key: "igd", label: "I(G)/I(D)", unit: "" },
  { key: "idg", label: "I(D)/I(G)", unit: "" },
  { key: "pgp", label: "Pos(G⁺)", unit: "cm⁻¹" },
  { key: "wgp", label: "FWHM(G⁺)", unit: "cm⁻¹" },
  { key: "pgm", label: "Pos(G⁻)", unit: "cm⁻¹" },
  { key: "wgm", label: "FWHM(G⁻)", unit: "cm⁻¹" },
  { key: "q", label: "G⁻ coupling 1/Q", unit: "" },
  { key: "dg", label: "Diameter from G⁺ − G⁻", unit: "nm" },
  { key: "p2d", label: "Pos(2D)", unit: "cm⁻¹" },
  { key: "i2dg", label: "I(2D)/I(G)", unit: "" },
  { key: "rn", label: "Residual / noise", unit: "" },
];

// the height of a Lorentzian (q = 0) or BWF band at u from its position, FWHM fw
function shapeAt(u, fw, q) {
  const w = fw * (1 - q * q) / (1 + q * q), x = 2 * u / w + q, a = 1 + q * x;
  return a * a / ((1 + x * x) * (1 + q * q));
}
// the maximum of G⁻ + G⁺ (each { h, c, w (FWHM), q }) and the profile's FWHM: { h, c, w }
export function gMax(gm, gp) {
  const bands = [gm, gp].filter(b => b && b.h > 0 && Number.isFinite(b.c) && b.w > 0);
  if (!bands.length) return { h: NaN, c: NaN, w: NaN };
  const lo = Math.min(...bands.map(b => b.c - 3 * b.w)), hi = Math.max(...bands.map(b => b.c + 3 * b.w)), M = 2400;
  const x = Float64Array.from({ length: M + 1 }, (_, k) => lo + (hi - lo) * k / M), v = Float64Array.from(x, t => bands.reduce((s, b) => s + b.h * shapeAt(t - b.c, b.w, b.q || 0), 0));
  let top = 0; for (let k = 1; k <= M; k++) if (v[k] > v[top]) top = k;
  const half = v[top] / 2, cross = (k, d) => { while (k + d >= 0 && k + d <= M && v[k + d] > half) k += d; const j = k + d; return j < 0 || j > M ? NaN : x[k] + (x[j] - x[k]) * (v[k] - half) / (v[k] - v[j]); };
  return { h: v[top], c: x[top], w: cross(top, 1) - cross(top, -1) };
}

// the one-G model preferred: the split adds four parameters (G⁻'s height, position, width and
// coupling) and must lower the BIC by `margin`; rn2, rn1 the two fits' residuals over the same noise
// on N channels, so BIC(one) − BIC(split) = 2 N ln(rn1 / rn2) − 4 ln N
export const preferOne = (rn2, rn1, N, margin = 10) => !(2 * N * Math.log(rn1 / rn2) - 4 * Math.log(N) >= margin);

// one spectrum: R the RBM bands, G the D, G⁻, G⁺, D′ bands of the split model, T the 2D band, each
// { h, sh, c, w, q, sq }; one the D, G, D′ bands of the one-G model where it is preferred (else null);
// rn the largest residual over noise of the fits used. split = false reads every G as one band: the
// caller passes it where the spectra cover the RBM window and show no RBM, since a G band of two
// phases (graphite with amorphous carbon) also splits into two components without any nanotube
export function readOne(R, G, T, rn, s, one = null, split = true) {
  const out = Object.fromEntries(QUANT.map(q => [q.key, NaN]));
  const has = b => b && b.h > 0 && b.h > s.snr * (b.sh || 0);
  out.rn = rn;
  // RBM
  if (R) {
    const on = R.filter(has);
    out.nrbm = on.length;
    if (on.length) {
      const top = on.reduce((a, b) => (b.h > a.h ? b : a));
      out.prbm = top.c; out.drbm = diameterOf(top.c, s);
      let sw = 0, sd = 0; for (const b of on) { const d = diameterOf(b.c, s); if (Number.isFinite(d)) { sw += b.h; sd += b.h * d; } }
      out.dmean = sw > 0 ? sd / sw : NaN;
    }
  }
  if (one) {
    const [D, G1] = one;
    if (!has(G1)) { out.type = 3; return out; }
    out.type = 2; out.pgp = G1.c; out.wgp = G1.w;
    if (has(D)) { out.igd = G1.h / D.h; out.idg = D.h / G1.h; } else out.idg = 0;
    if (has(T)) { out.p2d = T.c; out.i2dg = T.h / G1.h; }
    return out;
  }
  const [D, Gm, Gp] = G || [];
  const gm = has(Gm) ? Gm : null, gp = has(Gp) ? Gp : null;
  if (!gm && !gp) { out.type = 3; return out; }
  const top = gMax(gm, gp), iG = top.h;
  // two G components count as G⁻ and G⁺ only when they are apart by over half the narrower width
  // (one broad G band, as in multi-wall tubes, is otherwise split into two overlapping halves)
  if (split && gm && gp && gp.c - gm.c > 0.5 * Math.min(gm.w, gp.w)) {
    out.pgp = gp.c; out.wgp = gp.w; out.pgm = gm.c; out.wgm = gm.w; out.q = gm.q;
    const metallic = gm.q < -s.qMin && gm.q < -3 * (gm.sq || 0);
    out.type = metallic ? 0 : 1;
    const split = gp.c - gm.c;
    if (split > 0) out.dg = Math.sqrt((metallic ? s.cM : s.cS) / split);
  } else {
    // one G band: the maximum and FWHM of the two components together
    out.type = 2; out.pgp = top.c; out.wgp = top.w;
  }
  if (has(D)) { out.igd = iG / D.h; out.idg = D.h / iG; } else out.idg = 0;
  if (has(T)) { out.p2d = T.c; out.i2dg = T.h / iG; }
  return out;
}
