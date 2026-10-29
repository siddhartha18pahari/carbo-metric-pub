// DLC (diamond-like carbon) coating analysis. Every spectrum is fitted with the two-band
// model of Ferrari & Robertson (2000): a Lorentzian D band and a Breit-Wigner-Fano (BWF) G
// band over a straight background, all fitted together. From the fit:
//
//   I(D)/I(G)  ratio of the band heights (G's height is the BWF maximum); defects and disorder
//   Pos(G)     the BWF maximum, FWHM(G) the BWF's own full width at half maximum
//   m/I(G)     the background slope over G's height. In hydrogenated DLC (a-C:H) the slope is
//              the photoluminescence hydrogen brings: H [at.%] = 21.7 + 16.6 log10(m/I(G) [µm])
//              (Casiraghi et al. 2005, at 514.5 nm, for about 20 to 45 at.% H)
//   La         sp2 cluster size in amorphous carbon (stage 2 of the three-stage model):
//              I(D)/I(G) = C'(λ) La², C'(514 nm) = 0.0055 Å⁻² (Ferrari & Robertson 2000)
//
// The quality test checks every spectrum against limits; homogeneity sets the spread of each
// parameter against its own fit errors. Pure functions (no DOM), so they run under Node for
// the tests.

import { pinvSym } from "./engine/core.js";
import { noiseSigma, despike } from "./d3g-calibration.js";
import { wilson, heterogeneity, gridMoran, moranI, quantileSorted } from "./pf-analysis.js";

export const DLC_DEFAULTS = {
  range: [800, 2000],
  D: { pos_min: 1250, pos_max: 1450, width_min: 50, width_max: 450 },
  G: { pos_min: 1480, pos_max: 1620, width_min: 20, width_max: 260 },
  q: [-0.5, 0],                   // BWF coupling 1/Q: negative, a tail towards low shift
  snrMin: 10, maxIter: 80,
  hA: 21.7, hB: 16.6,             // H [at.%] = hA + hB log10(m/I(G) [µm])
  hRange: [20, 45],               // where that calibration holds
  laC: 0.0055,                    // C' in Å⁻²
  laMax: 2,                       // nm: stage 2 (amorphous carbon) only
  graphiteFwhm: 50,               // cm⁻¹: a G narrower than this (median) is graphite's, not DLC's
};

// the parameter vector: background (constant, slope over the half range), D, G, coupling
const B0 = 0, B1 = 1, HD = 2, CD = 3, WD = 4, HG = 5, CG = 6, WG = 7, QQ = 8, NP = 9;

// what each spectrum gives; se: has a standard error from the fit
export const DLC_PARAMS = [
  { key: "idig", label: "I(D)/I(G)", unit: "", se: true, qc: true },
  { key: "adag", label: "A(D)/A(G)", unit: "" },
  { key: "gpos", label: "G position", unit: "cm⁻¹", se: true, qc: true },
  { key: "gfwhm", label: "G FWHM", unit: "cm⁻¹", se: true, qc: true },
  { key: "gh", label: "G height", unit: "counts" },
  { key: "dpos", label: "D position", unit: "cm⁻¹", se: true },
  { key: "dfwhm", label: "D FWHM", unit: "cm⁻¹", se: true },
  { key: "dh", label: "D height", unit: "counts" },
  { key: "q", label: "BWF coupling 1/Q", unit: "" },
  { key: "slope", label: "Background slope m", unit: "counts/cm⁻¹" },
  { key: "mig", label: "m/I(G)", unit: "µm", se: true, qc: true },
  { key: "h", label: "Hydrogen content", unit: "at.%", se: true, qc: true },
  { key: "la", label: "sp² cluster size La", unit: "nm", se: true, qc: true },
  { key: "snr", label: "G signal-to-noise", unit: "" },
  { key: "res", label: "Residual / noise", unit: "" },
];
export const SE_KEYS = DLC_PARAMS.filter(p => p.se).map(p => p.key);
const NC = DLC_PARAMS.length, NS = SE_KEYS.length;

// per spectrum: 0 not measured, 1 fitted, 2 G too weak, 3 G position or width on a bound, 4 no fit
export const STATUS = ["not measured", "fitted", "G below the signal floor", "G on a fit bound", "fit failed"];
// flags: which parameters ended on a bound
export const FLAG = { gpos: 1, gwidth: 2, dpos: 4, dwidth: 8, q: 16 };

// --------------------------------------------------------------------------
// line shapes
// --------------------------------------------------------------------------
// BWF with its maximum h at c: x = 2 (ω − c) / w + q, value h (1 + q x)² / ((1 + q²)(1 + x²))
export function bwf(nu, h, c, w, q) {
  const x = 2 * (nu - c) / w + q, a = 1 + q * x;
  return h * a * a / ((1 + q * q) * (1 + x * x));
}
// its full width at half maximum: w (1 + q²) / (1 − q²), for |q| < 1
export const bwfFwhm = (w, q) => (Math.abs(q) < 1 ? w * (1 + q * q) / (1 - q * q) : NaN);
// its integral from a to b
export function bwfArea(h, c, w, q, a, b) {
  const F = x => q * q * x + (1 - q * q) * Math.atan(x) + q * Math.log(1 + x * x);
  const xa = 2 * (a - c) / w + q, xb = 2 * (b - c) / w + q;
  return h / (1 + q * q) * w / 2 * (F(xb) - F(xa));
}
export const lorentzArea = (h, c, w, a, b) => h * w / 2 * (Math.atan(2 * (b - c) / w) - Math.atan(2 * (a - c) / w));

// --------------------------------------------------------------------------
// the spectra: cut to the fit range, in ascending Raman shift
// --------------------------------------------------------------------------
export function prepare(cube, K, wn, P, settings) {
  const [lo, hi] = settings.range, idx = [];
  for (let c = 0; c < K; c++) if (wn[c] >= lo && wn[c] <= hi) idx.push(c);
  idx.sort((a, b) => wn[a] - wn[b]);
  const N = idx.length;
  if (N < 20) throw new Error(`only ${N} channels between ${lo} and ${hi} cm⁻¹: widen the fit range`);
  const nu = Float64Array.from(idx, c => wn[c]), Y = new Float32Array(P * N), measured = new Uint8Array(P);
  for (let p = 0; p < P; p++) {
    let any = false, fin = true;
    for (let n = 0; n < N; n++) { const v = cube[p * K + idx[n]]; Y[p * N + n] = v; if (!Number.isFinite(v)) fin = false; else if (v !== 0) any = true; }
    measured[p] = fin && any ? 1 : 0;           // stopped maps leave NaN rows, unmeasured particles zeros
  }
  return { nu, N, Y, P, measured, idx, range: [nu[0], nu[N - 1]] };
}

// the mean of the measured spectra, cleared of cosmic rays
export function meanSpectrum(prep) {
  const { N, Y, P, measured } = prep, m = new Float64Array(N);
  let n = 0;
  for (let p = 0; p < P; p++) if (measured[p]) { n++; for (let c = 0; c < N; c++) m[c] += Y[p * N + c]; }
  if (!n) return null;
  for (let c = 0; c < N; c++) m[c] /= n;
  return { y: despike(m).y, n };
}

// --------------------------------------------------------------------------
// small dense algebra
// --------------------------------------------------------------------------
function cholSolve(A, b, n) {
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

// --------------------------------------------------------------------------
// the fitter: one per channel grid, its buffers reused for every spectrum
// --------------------------------------------------------------------------
export function makeFitter(nu, settings) {
  const N = nu.length, mid = (nu[0] + nu[N - 1]) / 2, half = (nu[N - 1] - nu[0]) / 2 || 1, S = settings;
  const lo = Float64Array.of(-Infinity, -Infinity, 0, S.D.pos_min, S.D.width_min, 0, S.G.pos_min, S.G.width_min, S.q[0]);
  const hi = Float64Array.of(Infinity, Infinity, Infinity, S.D.pos_max, S.D.width_max, Infinity, S.G.pos_max, S.G.width_max, S.q[1]);
  const t = Float64Array.from(nu, v => (v - mid) / half);
  const f = new Float64Array(N), J = new Float64Array(N * NP), r = new Float64Array(N);
  const clamp = th => { for (let k = 0; k < NP; k++) th[k] = Math.min(hi[k], Math.max(lo[k], th[k])); return th; };

  // the model at every channel into out, and its Jacobian (N x 9) when jac
  function evaluate(th, out, jac) {
    const hd = th[HD], cd = th[CD], wd = th[WD], hg = th[HG], cg = th[CG], wg = th[WG], q = th[QQ], s = 1 + q * q, i0 = hg / s;
    for (let n = 0; n < N; n++) {
      const ud = 2 * (nu[n] - cd) / wd, ld = 1 / (1 + ud * ud);
      const u = 2 * (nu[n] - cg) / wg, x = u + q, a = 1 + q * x, den = 1 / (1 + x * x), g = a * a * den;
      out[n] = th[B0] + th[B1] * t[n] + hd * ld + i0 * g;
      if (!jac) continue;
      const k = n * NP, dld = 2 * hd * ld * ld * ud, dgdx = 2 * a * (q - x) * den * den;
      J[k] = 1; J[k + 1] = t[n];
      J[k + HD] = ld; J[k + CD] = 2 * dld / wd; J[k + WD] = dld * ud / wd;
      J[k + HG] = g / s; J[k + CG] = -2 * i0 * dgdx / wg; J[k + WG] = -u * i0 * dgdx / wg;
      J[k + QQ] = -2 * q * hg / (s * s) * g + i0 * (2 * x * a * den + dgdx);
    }
  }
  const ssrOf = (y, out) => { let s = 0; for (let n = 0; n < N; n++) { const d = y[n] - out[n]; s += d * d; } return s; };

  // the background and the two heights for fixed band shapes: linear least squares, D's height
  // dropped when it comes out negative
  function heights(y, th) {
    const cols = [], q = th[QQ], s = 1 + q * q, c2 = new Float64Array(N), c3 = new Float64Array(N);
    for (let n = 0; n < N; n++) {
      const ud = 2 * (nu[n] - th[CD]) / th[WD]; c2[n] = 1 / (1 + ud * ud);
      const x = 2 * (nu[n] - th[CG]) / th[WG] + q, a = 1 + q * x; c3[n] = a * a / (s * (1 + x * x));
    }
    const solve = withD => {
      cols.length = 0; cols.push(null, t); if (withD) cols.push(c2); cols.push(c3);
      const m = cols.length, A = new Float64Array(m * m), b = new Float64Array(m), col = (j, n) => (j === 0 ? 1 : cols[j][n]);
      for (let n = 0; n < N; n++) for (let i = 0; i < m; i++) { const ci = col(i, n); b[i] += ci * y[n]; for (let j = 0; j <= i; j++) A[i * m + j] += ci * col(j, n); }
      for (let i = 0; i < m; i++) { for (let j = i + 1; j < m; j++) A[i * m + j] = A[j * m + i]; A[i * m + i] *= 1 + 1e-12; }
      return cholSolve(A, b, m);
    };
    let z = solve(true), out = Float64Array.from(th);
    if (z && z[2] >= 0) { out[B0] = z[0]; out[B1] = z[1]; out[HD] = z[2]; out[HG] = Math.max(z[3], 1e-6); return out; }
    z = solve(false);
    if (z) { out[B0] = z[0]; out[B1] = z[1]; out[HD] = 0; out[HG] = Math.max(z[2], 1e-6); }
    return out;
  }

  // a start read off the spectrum: a line through its two ends, G at the highest point in its
  // window and its half width on the high side, D likewise once G's tail is taken off
  function guess(y) {
    const e = Math.max(3, Math.round(0.05 * N)), med = a => { const s = Float64Array.from(a).sort(); return s[s.length >> 1]; };
    const y0 = med(y.subarray(0, e)), y1 = med(y.subarray(N - e)), t0 = (t[0] + t[e - 1]) / 2, t1 = (t[N - e] + t[N - 1]) / 2;
    const slope = (y1 - y0) / (t1 - t0 || 1), base = n => y0 + slope * (t[n] - t0), z = Float64Array.from(y, (v, n) => v - base(n));
    const peak = (v, a, b) => { let k = -1; for (let n = 0; n < N; n++) if (nu[n] >= a && nu[n] <= b && (k < 0 || v[n] > v[k])) k = n; return k; };
    const mid2 = (a, b) => (a + b) / 2, th = new Float64Array(NP);
    let kg = peak(z, S.G.pos_min, S.G.pos_max); if (kg < 0) kg = N >> 1;
    const hg = Math.max(z[kg], 1e-6);
    let kr = kg; while (kr < N - 1 && z[kr] > hg / 2) kr++;
    const wg = kr > kg && z[kr] <= hg / 2 ? 2 * (nu[kr] - nu[kg]) : mid2(S.G.width_min, S.G.width_max);
    th[CG] = nu[kg]; th[WG] = wg; th[HG] = hg; th[QQ] = Math.min(S.q[1], Math.max(S.q[0], -0.1));
    const zd = Float64Array.from(z, (v, n) => v - hg / (1 + (2 * (nu[n] - th[CG]) / th[WG]) ** 2));
    let kd = peak(zd, S.D.pos_min, S.D.pos_max); if (kd < 0) kd = 0;
    const hd = Math.max(zd[kd], 0.05 * hg);
    let kl = kd; while (kl > 0 && zd[kl] > hd / 2) kl--;
    th[CD] = nu[kd]; th[WD] = kl < kd && zd[kl] <= hd / 2 ? 2 * (nu[kd] - nu[kl]) : mid2(S.D.width_min, S.D.width_max); th[HD] = hd;
    th[B0] = y0 + slope * (0 - t0); th[B1] = slope;
    return clamp(th);
  }

  // bounded Levenberg-Marquardt from th0; the covariance of the parameters not on a bound
  function lm(y, th0, maxIter = S.maxIter) {
    const th = clamp(Float64Array.from(th0)), fr = [];
    for (let k = 0; k < NP; k++) if (hi[k] > lo[k]) fr.push(k);
    const m = fr.length, A = new Float64Array(m * m), g = new Float64Array(m), M = new Float64Array(m * m), trial = new Float64Array(NP);
    evaluate(th, f, true);
    let ssr = ssrOf(y, f), lambda = 1e-3, it = 0;
    for (; it < maxIter; it++) {
      for (let n = 0; n < N; n++) r[n] = y[n] - f[n];
      A.fill(0); g.fill(0);
      for (let n = 0; n < N; n++) {
        const k = n * NP;
        for (let i = 0; i < m; i++) { const ji = J[k + fr[i]]; g[i] += ji * r[n]; for (let j = 0; j <= i; j++) A[i * m + j] += ji * J[k + fr[j]]; }
      }
      for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) A[i * m + j] = A[j * m + i];
      let dmax = 0; for (let i = 0; i < m; i++) dmax = Math.max(dmax, A[i * m + i]);
      let accepted = false, gain = 0;
      for (let tries = 0; tries < 12 && !accepted; tries++) {
        M.set(A);
        for (let i = 0; i < m; i++) M[i * m + i] += lambda * Math.max(A[i * m + i], 1e-12 * dmax);
        const d = cholSolve(M, g, m);
        if (!d) { lambda *= 10; continue; }
        trial.set(th); for (let i = 0; i < m; i++) trial[fr[i]] += d[i];
        clamp(trial);
        evaluate(trial, r, false);
        const s2 = ssrOf(y, r);
        if (s2 < ssr) { gain = (ssr - s2) / (ssr || 1); th.set(trial); ssr = s2; lambda = Math.max(lambda / 3, 1e-12); accepted = true; }
        else lambda *= 4;
      }
      evaluate(th, f, true);
      if (!accepted || gain < 1e-10) break;
    }
    // which ended on a bound, and the covariance of the rest at the residual's variance
    const onB = new Uint8Array(NP), act = [];
    for (const k of fr) {
      const span = Number.isFinite(hi[k] - lo[k]) ? hi[k] - lo[k] : 1;
      if ((Number.isFinite(lo[k]) && th[k] - lo[k] <= 1e-6 * span) || (Number.isFinite(hi[k]) && hi[k] - th[k] <= 1e-6 * span)) onB[k] = 1; else act.push(k);
    }
    const na = act.length, AA = new Float64Array(na * na);
    for (let n = 0; n < N; n++) { const k = n * NP; for (let i = 0; i < na; i++) for (let j = 0; j < na; j++) AA[i * na + j] += J[k + act[i]] * J[k + act[j]]; }
    const inv = pinvSym(AA, na), s2 = ssr / Math.max(1, N - na), cov = new Float64Array(NP * NP);
    for (let i = 0; i < na; i++) for (let j = 0; j < na; j++) cov[act[i] * NP + act[j]] = s2 * inv[i * na + j];
    return { th, ssr, iter: it, cov, onB };
  }

  return { nu, N, mid, half, lo, hi, J, evaluate, heights, guess, lm, clamp };
}

// fit one spectrum: from the given band shapes (the mean spectrum's) with heights solved for
// this spectrum, and again from its own start when G ends on a bound; the lower residual wins.
// Without shapes (the mean spectrum itself), from its own start at three couplings.
export function fitOne(F, y, shapes) {
  let best = null;
  const tryFrom = th0 => { const r = F.lm(y, th0); if (Number.isFinite(r.ssr) && (!best || r.ssr < best.ssr)) best = r; };
  if (shapes) tryFrom(F.heights(y, shapes));
  if (!best || best.onB[CG] || best.onB[WG]) {
    const g = F.guess(y);
    for (const q0 of shapes ? [g[QQ]] : [-0.1, 0, -0.3]) { g[QQ] = q0; tryFrom(F.heights(y, F.clamp(Float64Array.from(g)))); }
  }
  return best;
}

// --------------------------------------------------------------------------
// what a fit gives
// --------------------------------------------------------------------------
const LN10 = Math.LN10;
function se(cov, grad) {
  let v = 0;
  for (const [i, gi] of grad) for (const [j, gj] of grad) v += gi * gj * cov[i * NP + j];
  return v > 0 ? Math.sqrt(v) : v === 0 ? 0 : NaN;
}

// the parameters of one fit (DLC_PARAMS order) and the standard errors of SE_KEYS
export function derive(F, fit, noise, settings) {
  const th = fit.th, cov = fit.cov, S = settings, [a, b] = [F.nu[0], F.nu[F.N - 1]];
  const hd = th[HD], hg = th[HG], q = th[QQ], m = th[B1] / F.half;
  const idig = hd / hg, mig = m / hg * 1e4;
  const h = mig > 0 ? S.hA + S.hB * Math.log10(mig) : NaN;
  const la = idig >= 0 ? 0.1 * Math.sqrt(idig / S.laC) : NaN;
  const ad = lorentzArea(hd, th[CD], th[WD], a, b), ag = bwfArea(hg, th[CG], th[WG], q, a, b);
  const v = [idig, ag > 0 ? ad / ag : NaN, th[CG], bwfFwhm(th[WG], q), hg, th[CD], th[WD], hd, q, m, mig, h, la, hg / noise,
    Math.sqrt(fit.ssr / F.N) / noise];
  const sIdig = se(cov, [[HD, 1 / hg], [HG, -hd / (hg * hg)]]);
  const sFw = se(cov, [[WG, (1 + q * q) / (1 - q * q)], [QQ, th[WG] * 4 * q / (1 - q * q) ** 2]]);
  const sMig = se(cov, [[B1, 1e4 / (F.half * hg)], [HG, -mig / hg]]);
  const sH = mig > 0 ? se(cov, [[B1, S.hB / (LN10 * th[B1])], [HG, -S.hB / (LN10 * hg)]]) : NaN;
  const s = [sIdig, Math.sqrt(cov[CG * NP + CG]), sFw, Math.sqrt(cov[CD * NP + CD]), Math.sqrt(cov[WD * NP + WD]), sMig, sH,
    idig > 0 ? la / 2 * sIdig / idig : NaN];
  let flags = 0;
  if (fit.onB[CG]) flags |= FLAG.gpos;
  if (fit.onB[WG]) flags |= FLAG.gwidth;
  if (fit.onB[CD]) flags |= FLAG.dpos;
  if (fit.onB[WD]) flags |= FLAG.dwidth;
  if (fit.onB[QQ]) flags |= FLAG.q;
  return { v, s, flags };
}

// the fit drawn as components (for the cursor spectrum): background, D, G, total
export function components(F, th) {
  const N = F.N, base = new Float64Array(N), d = new Float64Array(N), g = new Float64Array(N), total = new Float64Array(N);
  for (let n = 0; n < N; n++) {
    const t = (F.nu[n] - F.mid) / F.half, ud = 2 * (F.nu[n] - th[CD]) / th[WD];
    base[n] = th[B0] + th[B1] * t; d[n] = th[HD] / (1 + ud * ud); g[n] = bwf(F.nu[n], th[HG], th[CG], th[WG], th[QQ]);
    total[n] = base[n] + d[n] + g[n];
  }
  return { base, d, g, total };
}

// --------------------------------------------------------------------------
// many spectra (a worker's chunk): the mean spectrum's shapes as the start
// --------------------------------------------------------------------------
export function fitSpectra({ nu, N, Y, settings, shapes }, list) {
  const F = makeFitter(nu, settings), M = list.length;
  const cols = new Float64Array(M * NC).fill(NaN), ses = new Float64Array(M * NS).fill(NaN), th = new Float64Array(M * NP).fill(NaN);
  const status = new Uint8Array(M), flags = new Uint8Array(M), y = new Float64Array(N);
  list.forEach((p, i) => {
    for (let n = 0; n < N; n++) y[n] = Y[p * N + n];
    const yy = despike(y).y, noise = noiseSigma(yy) || 1e-9, fit = fitOne(F, yy, shapes);
    if (!fit || !fit.th.every(Number.isFinite)) { status[i] = 4; return; }
    const D = derive(F, fit, noise, settings);
    cols.set(D.v, i * NC); ses.set(D.s, i * NS); th.set(fit.th, i * NP); flags[i] = D.flags;
    status[i] = D.v[13] < settings.snrMin ? 2 : D.flags & (FLAG.gpos | FLAG.gwidth) ? 3 : 1;
  });
  return { cols, ses, th, status, flags };
}

// the band shapes every spectrum starts from: the mean spectrum's fit
export function fitMean(prep, settings) {
  const mean = meanSpectrum(prep);
  if (!mean) throw new Error("no measured spectra");
  const F = makeFitter(prep.nu, settings), fit = fitOne(F, mean.y, null);
  if (!fit) throw new Error("the mean spectrum could not be fitted");
  return { F, y: mean.y, n: mean.n, fit, derived: derive(F, fit, noiseSigma(mean.y) || 1e-9, settings) };
}

// --------------------------------------------------------------------------
// the quality test, statistics and homogeneity
// --------------------------------------------------------------------------
const inLimits = (v, L) => (!L.use_lo || v >= L.lo) && (!L.use_hi || v <= L.hi);

// every fitted spectrum against the limits in use: 1 pass, 2 fail, 0 not judged (not fitted, or
// a tested value undefined, e.g. hydrogen where the background falls)
export function qcTest(col, status, limits) {
  const use = limits.filter(L => (L.use_lo && Number.isFinite(L.lo)) || (L.use_hi && Number.isFinite(L.hi)));
  const P = status.length, pass = new Uint8Array(P), byKey = Object.fromEntries(use.map(L => [L.key, 0]));
  let nPass = 0, nFail = 0, undef = 0;
  if (!use.length) return { pass, use, n: 0, nPass, nFail, undef, byKey, share: NaN, ci: [NaN, NaN] };
  for (let p = 0; p < P; p++) {
    if (status[p] !== 1) continue;
    let ok = true, judged = true;
    for (const L of use) {
      const v = col(L.key)[p];
      if (!Number.isFinite(v)) { judged = false; break; }
      if (!inLimits(v, L)) { ok = false; byKey[L.key]++; }
    }
    if (!judged) { undef++; continue; }
    pass[p] = ok ? 1 : 2; ok ? nPass++ : nFail++;
  }
  const n = nPass + nFail;
  return { pass, use, n, nPass, nFail, undef, byKey, share: n ? nPass / n : NaN, ci: wilson(nPass, n) };
}

// one parameter over the fitted spectra: summary, the share inside its limits (Wilson 95 %)
// and Ppk, min(USL − mean, mean − LSL) / 3 SD with the overall SD
export function paramStats(v, ok, L) {
  const a = []; for (let p = 0; p < v.length; p++) if (ok[p] && Number.isFinite(v[p])) a.push(v[p]);
  const n = a.length, s = Float64Array.from(a).sort();
  if (!n) return { n: 0 };
  let sum = 0; for (const x of a) sum += x;
  const mean = sum / n; let ss = 0; for (const x of a) ss += (x - mean) ** 2;
  const sd = n > 1 ? Math.sqrt(ss / (n - 1)) : NaN;
  const out = { n, mean, sd, cv: sd / Math.abs(mean), median: quantileSorted(s, 0.5), q1: quantileSorted(s, 0.25), q3: quantileSorted(s, 0.75),
    min: s[0], max: s[n - 1] };
  if (L && ((L.use_lo && Number.isFinite(L.lo)) || (L.use_hi && Number.isFinite(L.hi)))) {
    let k = 0; for (const x of a) if (inLimits(x, L)) k++;
    const sides = [];
    if (L.use_hi && Number.isFinite(L.hi)) sides.push((L.hi - mean) / (3 * sd));
    if (L.use_lo && Number.isFinite(L.lo)) sides.push((mean - L.lo) / (3 * sd));
    Object.assign(out, { inSpec: k / n, inSpecCI: wilson(k, n), ppk: Math.min(...sides),
      tolerance: L.use_lo && L.use_hi ? L.hi - L.lo : NaN });
  }
  return out;
}

// how even one parameter is: the spread beyond each spectrum's own fit error (Cochran's Q, I²,
// the SD of the true values tau by Paule-Mandel with its 95 % interval) and Moran's I of the
// values over the map (neighbouring pixels) or the particles (6 nearest)
export function homogeneity(v, s, ok, where) {
  const y = [], e = [], at = [];
  for (let p = 0; p < v.length; p++) if (ok[p] && Number.isFinite(v[p]) && s[p] > 0) { y.push(v[p]); e.push(s[p]); at.push(p); }
  const H = heterogeneity(y, e);
  if (!H) return null;
  let moran = NaN;
  if (where.grid) {
    const img = new Float64Array(v.length), mask = new Uint8Array(v.length);
    for (const p of at) { img[p] = v[p]; mask[p] = 1; }
    moran = gridMoran(img, mask, where.grid.nx, where.grid.ny);
  } else if (where.x) {
    const M = moranI(Float64Array.from(at, p => v[p]), Float64Array.from(at, p => where.x[p]), Float64Array.from(at, p => where.y[p]), { perms: 0 });
    moran = M ? M.I : NaN;
  }
  return { ...H, cvTrue: H.tau / Math.abs(H.mu), cv: H.sd / Math.abs(H.mean), moran };
}

// I² read as Higgins et al. (2003) do: the share of the spread that is real variation
export const homogeneityLevel = I2 => (I2 < 0.25 ? "good" : I2 < 0.75 ? "warn" : "bad");
