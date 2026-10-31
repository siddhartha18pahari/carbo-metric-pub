// Silicon: the 520 cm⁻¹ band of crystalline silicon fitted in every spectrum (a Lorentzian, or a
// Voigt where the instrument's own width matters), with, for thin films, the grain-boundary band
// near 510 and the amorphous band near 480 (Gaussians). From the fit:
//
//   shift   Δω = ω − ω0 from the unstressed position ω0 (520.7 cm⁻¹ by default, or measured)
//   stress  σ = k Δω. For uniaxial stress k = −434 MPa per cm⁻¹ (De Wolf 1996, Semicond. Sci.
//           Technol. 11, 139): a shift up is compression. A biaxial or general stress state needs
//           its own k (it depends on the stress state and the crystal orientation), so k is editable.
//   crystalline fraction  Xc = (A_c + A_gb) / (A_c + A_gb + y A_a), y the ratio of the amorphous to
//           the crystalline Raman cross-section (about 0.8 to 1; editable)
//   calibration check  the band's position against the reference within a tolerance, logged per
//           day, with the log's mean and 3 SD limits (a Shewhart chart)
// Pure functions; the fits are peakfit.js models.

import { newPeak, MODEL_DEFAULTS } from "./peakfit.js";

export const SI_DEFAULTS = {
  model: "single", shape: "lorentz", range: [440, 600],
  w0: 520.7, k: -434, y: 0.8,
  tol: 0.5, fwhmMax: 6,                   // calibration: |ω − ω0| ≤ tol, FWHM ≤ fwhmMax (cm⁻¹)
  snr: 5,
};
export function model(s) {
  // started at the usual position whatever ω0 is, so a new ω0 changes the reading, not the fit
  const c = newPeak(s.shape, 520.7, 1000, 4, "c-Si");
  c.c.lo = 505; c.c.hi = 536;
  const peaks = [c];
  if (s.model === "film") {
    const gb = newPeak("gauss", 510, 100, 12, "Grain boundary"), a = newPeak("gauss", 480, 100, 60, "a-Si");
    gb.c.lo = 495; gb.c.hi = 515; gb.w.lo = 5; gb.w.hi = 30;
    a.c.lo = 460; a.c.hi = 495; a.w.lo = 30; a.w.hi = 120;
    peaks.unshift(a, gb);
  }
  return { ...structuredClone(MODEL_DEFAULTS), range: [...s.range], baseline: { kind: "poly", order: 1 }, despike: true, peaks };
}
export const QUANT = [
  { key: "pos", label: "Si position", unit: "cm⁻¹" },
  { key: "shift", label: "Shift Δω", unit: "cm⁻¹" },
  { key: "stress", label: "Stress", unit: "MPa" },
  { key: "fwhm", label: "Si FWHM", unit: "cm⁻¹" },
  { key: "h", label: "Si height", unit: "counts" },
  { key: "xc", label: "Crystalline fraction", unit: "%" },
  { key: "rn", label: "Residual / noise", unit: "" },
];
// one spectrum: bands [{ h, sh, c, sc, w, a }] in the model's order (film: a-Si, grain boundary, c-Si)
export function readOne(bands, rn, s) {
  const out = Object.fromEntries(QUANT.map(q => [q.key, NaN])), c = bands[bands.length - 1];
  out.rn = rn;
  if (!(c && c.h > s.snr * (c.sh || 0) && c.h > 0)) return out;
  out.pos = c.c; out.shift = c.c - s.w0; out.stress = s.k * out.shift; out.fwhm = c.w; out.h = c.h;
  out.sPos = c.sc; out.sStress = Math.abs(s.k) * c.sc;
  if (s.model === "film") {
    const [a, gb] = bands, A = v => (v && v.a > 0 ? v.a : 0), cr = A(c) + A(gb);
    out.xc = cr + s.y * A(a) > 0 ? 100 * cr / (cr + s.y * A(a)) : NaN;
  }
  return out;
}
// a calibration measurement against the limits
export function calCheck(pos, fwhm, s) {
  const d = pos - s.w0, okPos = Math.abs(d) <= s.tol, okW = !(fwhm > s.fwhmMax);
  return { d, okPos, okW, pass: okPos && okW };
}
// a log of checks: mean, SD and the points outside mean ± 3 SD (from the first `base` entries, all by default)
export function controlLimits(values, base = values.length) {
  const v = values.slice(0, base).filter(Number.isFinite), n = v.length;
  if (n < 2) return { n, mean: n ? v[0] : NaN, sd: NaN, lo: NaN, hi: NaN, out: [] };
  const mean = v.reduce((a, b) => a + b, 0) / n, sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1));
  return { n, mean, sd, lo: mean - 3 * sd, hi: mean + 3 * sd, out: values.map((x, i) => (Math.abs(x - mean) > 3 * sd ? i : -1)).filter(i => i >= 0) };
}
