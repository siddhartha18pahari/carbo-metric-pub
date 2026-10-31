// Graphene: every spectrum fitted with Lorentzian D, G and D′ bands (with a straight background)
// and, in its own window, a single Lorentzian 2D band, then read as:
//
//   layers   from the 2D band: a single narrow Lorentzian is a monolayer (FWHM about 24 to 30
//            cm⁻¹, Ferrari et al. 2006, Phys. Rev. Lett. 97, 187401); a bilayer's 2D has four
//            components and reads as one band about 50 cm⁻¹ wide; thicker graphite is wider still.
//            I(2D)/I(G) is reported but not used: doping and the substrate change it.
//   defects  point-defect distance and density from I(D)/I(G) (Cançado et al. 2011, Nano Lett.
//            11, 3190): L_D² [nm²] = 1.8e-9 λ⁴ / (I(D)/I(G)), n_D [cm⁻²] = 1.8e22 / λ⁴ × I(D)/I(G),
//            λ in nm, valid for L_D ≥ 10 nm; their nature from I(D)/I(D′) (Eckmann et al. 2012,
//            Nano Lett. 12, 3925): about 13 for sp³ sites, 7 for vacancies, 3.5 for boundaries.
//   strain and doping  from the positions of G and 2D (Lee et al. 2012, Nat. Commun. 3, 1024): the
//            shift from the charge-neutral, unstrained point (1581.6, 2676.9) cm⁻¹ (at 514.5 nm)
//            split along strain (Δω2D/ΔωG = 2.2) and hole doping (0.70); strain from G's share at
//            −69.1 cm⁻¹ per % (biaxial). The doping is given as G's shift: turning it into a carrier
//            density needs a calibration the paper takes from gated graphene.
// Pure functions; the fits are peakfit.js models.

import { newPeak, MODEL_DEFAULTS } from "./peakfit.js";

export const GRAPHENE_DEFAULTS = {
  laser: 532,
  win1: [1200, 1700], win2: [2500, 2900],
  mono: 35, bi: 60,                       // FWHM(2D) limits, cm⁻¹: monolayer up to mono, bilayer or few up to bi
  snr: 5,                                 // a band counts when its height is this many times its error
  g0: 1581.6, d0: 2676.9,                 // the charge-neutral, unstrained point at 514.5 nm
  slopeT: 2.2, slopeH: 0.70, sensG: -69.1,
};
export const LAYERS = ["Monolayer", "Bilayer or few-layer", "Multilayer", "No graphene bands"];
export const DEFECT_TYPES = [["sp³ sites", 13], ["vacancies", 7], ["boundaries", 3.5]];

const lor = (name, c, h, w, cr, wr) => {
  const p = newPeak("lorentz", c, h, w, name);
  p.c.lo = c - cr; p.c.hi = c + cr; p.w.lo = wr[0]; p.w.hi = wr[1];
  return p;
};
// the two models, D and 2D positions following the laser (D disperses about 50, 2D about 100 cm⁻¹ per eV)
export function models(s) {
  const eV = 1239.84 / s.laser, d = 1350 - 50 * (eV - 2.33), d2 = 2680 - 100 * (eV - 2.33);
  const base = { ...structuredClone(MODEL_DEFAULTS), baseline: { kind: "poly", order: 1 }, despike: true };
  return {
    m1: { ...base, range: [...s.win1], peaks: [lor("D", d, 50, 25, 40, [8, 120]), lor("G", 1582, 500, 15, 30, [5, 80]), lor("D′", 1620, 20, 10, 20, [4, 40])] },
    m2: { ...base, range: [...s.win2], peaks: [lor("2D", d2, 800, 30, 90, [12, 150])] },
  };
}

// what each spectrum gives (the keys of the maps, the CSV and the report)
export const QUANT = [
  { key: "layer", label: "Layers", unit: "" },
  { key: "i2dg", label: "I(2D)/I(G)", unit: "" },
  { key: "w2d", label: "FWHM(2D)", unit: "cm⁻¹" },
  { key: "p2d", label: "Pos(2D)", unit: "cm⁻¹" },
  { key: "pg", label: "Pos(G)", unit: "cm⁻¹" },
  { key: "wg", label: "FWHM(G)", unit: "cm⁻¹" },
  { key: "idg", label: "I(D)/I(G)", unit: "" },
  { key: "adg", label: "A(D)/A(G)", unit: "" },
  { key: "ld", label: "Defect distance L_D", unit: "nm" },
  { key: "nd", label: "Defect density n_D", unit: "10¹⁰ cm⁻²" },
  { key: "iddp", label: "I(D)/I(D′)", unit: "" },
  { key: "eps", label: "Strain ε", unit: "%" },
  { key: "dop", label: "G shift from doping", unit: "cm⁻¹" },
  { key: "rn", label: "Residual / noise", unit: "" },
];

// one spectrum's reading from its two fits: f1 = { h, se, c, w, a } per band of m1 (D, G, D′), f2 the 2D's,
// each value [height, position, FWHM, area] with its error; rn the larger residual over noise
export function readOne(f1, f2, rn, s) {
  const out = Object.fromEntries(QUANT.map(q => [q.key, NaN]));
  const has = b => b && b.h > s.snr * (b.sh || 0) && b.h > 0;
  const [D, G, Dp] = f1 || [], T = f2;
  out.rn = rn;
  if (!has(G)) { out.layer = 3; return out; }
  out.pg = G.c; out.wg = G.w;
  if (has(T)) {
    out.i2dg = T.h / G.h; out.w2d = T.w; out.p2d = T.c;
    out.layer = T.w <= s.mono ? 0 : T.w <= s.bi ? 1 : 2;
    // Lee et al.: (ΔωG, Δω2D) = a (1, slopeT) + b (1, slopeH)
    const dg = G.c - s.g0, d2 = T.c - s.d0, a = (d2 - s.slopeH * dg) / (s.slopeT - s.slopeH);
    out.eps = a / s.sensG; out.dop = dg - a;
  } else out.layer = 3;
  if (has(D)) {
    out.idg = D.h / G.h; out.adg = G.a > 0 ? D.a / G.a : NaN;
    const l4 = s.laser ** 4, ld = Math.sqrt(1.8e-9 * l4 / out.idg);
    if (ld >= 10) { out.ld = ld; out.nd = 1.8e22 / l4 * out.idg / 1e10; }
    if (has(Dp)) out.iddp = D.h / Dp.h;
  } else out.idg = 0;
  return out;
}
// the defect type nearest to I(D)/I(D′) on a log scale
export function defectType(r) {
  if (!(r > 0)) return null;
  let best = null;
  for (const [name, v] of DEFECT_TYPES) { const d = Math.abs(Math.log(r / v)); if (!best || d < best.d) best = { name, v, d }; }
  return best;
}
