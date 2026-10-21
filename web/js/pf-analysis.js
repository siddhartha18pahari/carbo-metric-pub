// Semi-quantitative statistics for ParticleFinder data: confidence intervals
// for shares, populations found by a Gaussian mixture, volume-weighted size
// quantiles and the crystallite size La. Pure functions (no DOM), so they run
// under Node for the tests.

import { pinvSym, erf } from "./engine/core.js";

// a seeded generator, so a bootstrap or a mixture fit gives the same answer twice
export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Wilson score interval for k of n (95 % by default): stays inside 0..1 and
// keeps its coverage for small n or shares near 0 or 1, where p ± 1.96 SE fails
export function wilson(k, n, z = 1.959964) {
  if (!n) return [NaN, NaN];
  const p = k / n, z2 = z * z, den = 1 + z2 / n;
  const c = (p + z2 / (2 * n)) / den, h = (z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / den;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

export function quantileSorted(s, q) {
  if (!s.length) return NaN;
  const p = (s.length - 1) * q, i = Math.floor(p), f = p - i;
  return i + 1 < s.length ? s[i] * (1 - f) + s[i + 1] * f : s[i];
}
export function median(a) { return quantileSorted(Float64Array.from(a).filter(Number.isFinite).sort(), 0.5); }

// the value below which a share q of the total weight lies (each item's weight
// centred on it, linear in between): D50 by volume is weightedQuantile(d, v, 0.5)
export function weightedQuantile(values, weights, q) {
  const idx = [];
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i]) && weights[i] > 0) idx.push(i);
  if (!idx.length) return NaN;
  idx.sort((a, b) => values[a] - values[b]);
  let tot = 0; for (const i of idx) tot += weights[i];
  let cum = 0, prevF = -Infinity, prevV = values[idx[0]];
  for (const i of idx) {
    const f = (cum + weights[i] / 2) / tot;
    if (f >= q) return prevF === -Infinity ? values[i] : prevV + (values[i] - prevV) * (q - prevF) / (f - prevF);
    cum += weights[i]; prevF = f; prevV = values[i];
  }
  return values[idx[idx.length - 1]];
}

// Cancado et al., Appl. Phys. Lett. 88, 163106 (2006): La (nm) = 2.4e-10 λ⁴ (A_D / A_G)⁻¹,
// λ the laser wavelength in nm and A the integrated band areas. Calibrated on
// nanographites with La of about 20 to 500 nm; below about 2 nm the relation inverts
// (Ferrari and Robertson, Phys. Rev. B 61, 14095, 2000).
export function laCancado(areaD, areaG, lambdaNm) {
  return areaD > 0 && areaG > 0 && lambdaNm > 0 ? 2.4e-10 * lambdaNm ** 4 * areaG / areaD : NaN;
}

// each column centred on its median and scaled by its MAD (as an SD), so one
// outlying particle does not set the scale of a feature
export function robustScale(cols) {
  return cols.map(c => {
    const s = Float64Array.from(c).sort(), m = quantileSorted(s, 0.5);
    const dev = Float64Array.from(c, v => Math.abs(v - m)).sort();
    let sd = 1.4826 * quantileSorted(dev, 0.5);
    if (!(sd > 0)) { let ss = 0; for (const v of c) ss += (v - m) ** 2; sd = Math.sqrt(ss / Math.max(1, c.length - 1)) || 1; }
    return Float64Array.from(c, v => (v - m) / sd);
  });
}

// Gaussian mixture with diagonal covariances, fitted by EM from a k-means++
// start. cols: d feature columns of n values. Returns the fit and its BIC.
export function gmm(cols, k, { seed = 1, iters = 300, tol = 1e-7 } = {}) {
  const d = cols.length, n = cols[0].length, R = rng(seed);
  const x = (i, j) => cols[j][i];
  // k-means++ seeding
  const means = [];
  means.push(Array.from({ length: d }, (_, j) => x(Math.floor(R() * n), j)));
  const d2 = new Float64Array(n).fill(Infinity);
  while (means.length < k) {
    const m = means[means.length - 1];
    let tot = 0;
    for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < d; j++) s += (x(i, j) - m[j]) ** 2; if (s < d2[i]) d2[i] = s; tot += d2[i]; }
    let r = R() * tot, pick = n - 1;
    for (let i = 0; i < n; i++) { r -= d2[i]; if (r <= 0) { pick = i; break; } }
    means.push(Array.from({ length: d }, (_, j) => x(pick, j)));
  }
  const gvar = Array.from({ length: d }, (_, j) => { let s = 0, q = 0; for (let i = 0; i < n; i++) { s += x(i, j); q += x(i, j) ** 2; } const m = s / n; return Math.max(q / n - m * m, 1e-12); });
  const floor = gvar.map(v => v * 1e-4);
  let vars = means.map(() => gvar.slice()), w = new Float64Array(k).fill(1 / k);
  const resp = new Float64Array(n * k), lp = new Float64Array(k);
  let ll = -Infinity;
  for (let it = 0; it < iters; it++) {
    // E step, in logs
    let llNew = 0;
    const lc = means.map((m, c) => Math.log(w[c]) - 0.5 * vars[c].reduce((s, v) => s + Math.log(2 * Math.PI * v), 0));
    for (let i = 0; i < n; i++) {
      let mx = -Infinity;
      for (let c = 0; c < k; c++) {
        let s = lc[c];
        for (let j = 0; j < d; j++) s -= (x(i, j) - means[c][j]) ** 2 / (2 * vars[c][j]);
        lp[c] = s; if (s > mx) mx = s;
      }
      let z = 0; for (let c = 0; c < k; c++) z += Math.exp(lp[c] - mx);
      const lz = mx + Math.log(z); llNew += lz;
      for (let c = 0; c < k; c++) resp[i * k + c] = Math.exp(lp[c] - lz);
    }
    // M step
    for (let c = 0; c < k; c++) {
      let nc = 0; const m = new Array(d).fill(0), v = new Array(d).fill(0);
      for (let i = 0; i < n; i++) { const r = resp[i * k + c]; nc += r; for (let j = 0; j < d; j++) m[j] += r * x(i, j); }
      if (nc < 1e-9) continue;                      // an emptied component keeps its last state
      for (let j = 0; j < d; j++) m[j] /= nc;
      for (let i = 0; i < n; i++) { const r = resp[i * k + c]; for (let j = 0; j < d; j++) v[j] += r * (x(i, j) - m[j]) ** 2; }
      for (let j = 0; j < d; j++) v[j] = Math.max(v[j] / nc, floor[j]);
      means[c] = m; vars[c] = v; w[c] = nc / n;
    }
    if (Math.abs(llNew - ll) < tol * Math.abs(llNew)) { ll = llNew; break; }
    ll = llNew;
  }
  const labels = new Int16Array(n);
  let entropy = 0;
  for (let i = 0; i < n; i++) {
    let b = 0;
    for (let c = 0; c < k; c++) { const r = resp[i * k + c]; if (r > 0) entropy -= r * Math.log(r); if (r > resp[i * k + b]) b = c; }
    labels[i] = b;
  }
  const params = k * 2 * d + (k - 1), bic = -2 * ll + params * Math.log(n);
  // ICL (Biernacki, Celeux and Govaert, IEEE TPAMI 22, 719, 2000): BIC plus twice the
  // entropy of the assignments, so overlapping components that only bend one skewed
  // cloud into shape cost more than they gain
  return { k, means, vars, weights: Array.from(w), logLik: ll, bic, icl: bic + 2 * entropy, entropy, labels };
}

// How far the density of two components dips between them: along the segment
// joining their means, the lowest density between the two highest peaks over
// the lower peak. 1 when there is a single peak (one population, however wide).
export function ridgeRatio(fit, a, b, steps = 101) {
  const ma = fit.means[a], mb = fit.means[b], wsum = fit.weights[a] + fit.weights[b];
  const dens = x => [a, b].reduce((s, c) => {
    let e = 0, norm = 1;
    for (let j = 0; j < x.length; j++) { e += (x[j] - fit.means[c][j]) ** 2 / (2 * fit.vars[c][j]); norm *= Math.sqrt(2 * Math.PI * fit.vars[c][j]); }
    return s + fit.weights[c] / wsum * Math.exp(-e) / norm;
  }, 0);
  const f = Array.from({ length: steps }, (_, i) => dens(ma.map((v, j) => v + (mb[j] - v) * i / (steps - 1))));
  const peaks = [];
  for (let i = 0; i < steps; i++) if ((i === 0 || f[i] >= f[i - 1]) && (i === steps - 1 || f[i] > f[i + 1])) peaks.push(i);
  if (peaks.length < 2) return 1;
  peaks.sort((u, v) => f[v] - f[u]);
  const [p, q] = [peaks[0], peaks[1]].sort((u, v) => u - v);
  let low = Infinity; for (let i = p; i <= q; i++) low = Math.min(low, f[i]);
  return low / Math.min(f[p], f[q]);
}

// The populations the data supports. The mixture with the lowest ICL among
// k = 1..kMax (each component at least 1 % of the particles; the best of a few
// seeded starts) describes the density; components whose joint density shows
// no real dip between them (ridge ratio above rStar) are then merged, since
// with thousands of particles BIC and even ICL slice one skewed cloud into
// pieces (Hennig, Adv. Data Anal. Classif. 4, 3, 2010; rStar = 0.2 as there).
export function bestGmm(cols, kMax = 4, seed = 1, { rStar = 0.2 } = {}) {
  const n = cols[0].length, top = Math.max(1, Math.min(kMax, Math.floor(n / 20)));
  let best = null;
  const tried = [];
  for (let k = 1; k <= top; k++) {
    let fit = null;
    for (let s = 0; s < (k === 1 ? 1 : 4); s++) {
      const f = gmm(cols, k, { seed: seed + 97 * s + k });
      if (!fit || f.logLik > fit.logLik) fit = f;
    }
    const counts = new Array(k).fill(0); for (const l of fit.labels) counts[l]++;
    const ok = Math.min(...counts) >= Math.max(5, 0.01 * n);
    tried.push({ k, bic: fit.bic, icl: fit.icl, smallest: Math.min(...counts), ok });
    if (ok && (!best || fit.icl < best.icl)) best = fit;
  }
  // merge components without a dip between them (union-find)
  const root = Array.from({ length: best.k }, (_, i) => i), find = i => (root[i] === i ? i : (root[i] = find(root[i])));
  const ratios = [];
  for (let a = 0; a < best.k; a++) for (let c = a + 1; c < best.k; c++) {
    const r = ridgeRatio(best, a, c); ratios.push({ a, b: c, ratio: r });
    if (r > rStar) root[find(a)] = find(c);
  }
  const groups = [...new Set(Array.from({ length: best.k }, (_, i) => find(i)))];
  const labels = Int16Array.from(best.labels, l => groups.indexOf(find(l)));
  return { ...best, components: best.k, k: groups.length, labels, ratios, tried };
}

// share of the total weight in each population, with a percentile bootstrap
// over particles (the assignment held fixed): [share, lo, hi] per population
export function bootstrapShares(labels, weights, k, { B = 500, seed = 7, conf = 0.95 } = {}) {
  const n = labels.length, R = rng(seed), draws = Array.from({ length: k }, () => new Float64Array(B));
  let tot = 0; const point = new Float64Array(k);
  for (let i = 0; i < n; i++) { tot += weights[i]; if (labels[i] >= 0) point[labels[i]] += weights[i]; }
  if (!(tot > 0)) return null;
  for (let b = 0; b < B; b++) {
    let t = 0; const s = new Float64Array(k);
    for (let i = 0; i < n; i++) { const j = Math.floor(R() * n), wj = weights[j]; t += wj; if (labels[j] >= 0) s[labels[j]] += wj; }
    for (let c = 0; c < k; c++) draws[c][b] = t > 0 ? s[c] / t : 0;
  }
  const a = (1 - conf) / 2;
  return draws.map((d, c) => { d.sort(); return [point[c] / tot, quantileSorted(d, a), quantileSorted(d, 1 - a)]; });
}

// --------------------------------------------------------------------------
// Uncertainty of each particle's fitted bands, and whether a coating is uniform
// --------------------------------------------------------------------------

// Covariance of the fitted band heights of every spectrum, from the model
// linearised at its solution: Cov = σ² (JᵀJ)⁺ over the parameters the fit
// moves (the baseline coefficients, every height, and the position and width
// of each band neither held nor on a bound); σ is the spectrum's residual
// (its std error). A broad band's height is then judged against everything
// under it, not against the noise of one channel. Returns P m x m blocks,
// row-major, NaN for spectra without a fit. only: the spectra to compute
// (all by default), written into out when it is given.
export function heightCovariances(fit, cfg, wn, only = null, out = null) {
  const [lo, hi] = fit.range, nu = Float64Array.from(wn).filter(v => v >= lo && v <= hi);
  const N = nu.length, m = fit.m, P = fit.std_error.length;
  const nb = cfg.fit_separately ? 0 : cfg.baseline_order + 1, mid = 0.5 * (lo + hi), half = 0.5 * (hi - lo);
  const held = new Set(fit.held || []);
  const shapeFree = cfg.peaks.map(p => p.fit !== false && !cfg.skip_fitting && !held.has(p.name));
  const lorentz = fit.shapes.map(s => s === "Lorentz"), G4 = 2.772588722239781;
  if (!out) out = new Float64Array(P * m * m).fill(NaN);
  const maxQ = nb + 3 * m, J = new Float64Array(N * maxQ), JtJ = new Float64Array(maxQ * maxQ);
  const onBound = (v, a, b) => Math.abs(v - a) <= 1e-6 * (b - a) || Math.abs(v - b) <= 1e-6 * (b - a);
  for (const p of only || Array.from({ length: P }, (_, i) => i)) {
    const sigma = fit.std_error[p];
    if (!(sigma > 0)) continue;
    // the columns of J, in order: baseline, then per band its height and any free shape
    const cols = [], hcol = new Int32Array(m);
    for (let k = 0; k < nb; k++) cols.push(n => ((nu[n] - mid) / half) ** k);
    for (let j = 0; j < m; j++) {
      const A = fit.intensity[p * m + j], c = fit.position[p * m + j], w = fit.width[p * m + j], pk = cfg.peaks[j];
      const L = lorentz[j];
      hcol[j] = cols.length;
      cols.push(n => { if (L) { const u = 2 * (nu[n] - c) / w; return 1 / (1 + u * u); } const v = (nu[n] - c) / w; return Math.exp(-G4 * v * v); });
      if (!shapeFree[j]) continue;
      if (!onBound(c, pk.pos_min, pk.pos_max)) cols.push(n => {
        if (L) { const u = 2 * (nu[n] - c) / w, d = 1 + u * u; return A * (-2 * u / (d * d)) * (-2 / w); }
        const v = (nu[n] - c) / w, e = Math.exp(-G4 * v * v); return A * (-2 * G4 * v * e) * (-1 / w);
      });
      if (!onBound(w, pk.width_min, pk.width_max)) cols.push(n => {
        if (L) { const u = 2 * (nu[n] - c) / w, d = 1 + u * u; return A * (-2 * u / (d * d)) * (-u / w); }
        const v = (nu[n] - c) / w, e = Math.exp(-G4 * v * v); return A * (-2 * G4 * v * e) * (-v / w);
      });
    }
    const q = cols.length;
    for (let a = 0; a < q; a++) for (let n = 0; n < N; n++) J[n * q + a] = cols[a](n);
    JtJ.fill(0, 0, q * q);
    for (let n = 0; n < N; n++) for (let a = 0; a < q; a++) {
      const ja = J[n * q + a]; if (ja === 0) continue;
      for (let b = a; b < q; b++) JtJ[a * q + b] += ja * J[n * q + b];
    }
    for (let a = 0; a < q; a++) for (let b = 0; b < a; b++) JtJ[a * q + b] = JtJ[b * q + a];
    const inv = pinvSym(JtJ.subarray(0, q * q), q), s2 = sigma * sigma, o = p * m * m;
    for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) out[o + a * m + b] = s2 * inv[hcol[a] * q + hcol[b]];
  }
  return out;
}

// variance of the ratio of two heights A_n / A_d by the delta method, valid at A_n = 0
export function ratioVariance(cov, o, m, jn, jd, An, Ad) {
  const R = An / Ad, vnn = cov[o + jn * m + jn], vdd = cov[o + jd * m + jd], vnd = cov[o + jn * m + jd];
  return (vnn - 2 * R * vnd + R * R * vdd) / (Ad * Ad);
}

const normCdf = z => 0.5 * (1 + erf(z / Math.SQRT2));
// chi-square: upper tail and quantile by the Wilson-Hilferty cube-root normal
// approximation (good to a few per cent of p for df >= 3, better as df grows)
export function chi2Upper(x, df) {
  if (!(df > 0)) return NaN;
  const h = 2 / (9 * df);
  return 1 - normCdf((Math.cbrt(x / df) - (1 - h)) / Math.sqrt(h));
}
export function chi2Quantile(z, df) { const h = 2 / (9 * df); return df * Math.max(0, 1 - h + z * Math.sqrt(h)) ** 3; }

// tau² at which a Q statistic, which falls as tau² grows, comes down to c;
// scale is a first guess at its size
function solveTau2(Qat, c, scale) {
  if (Qat(0) <= c) return 0;
  let a = 0, b = Math.max(1e-30, scale);
  while (Qat(b) > c) b *= 4;
  for (let it = 0; it < 100; it++) { const h = 0.5 * (a + b); if (Qat(h) > c) a = h; else b = h; }
  return 0.5 * (a + b);
}

// Are many measurements of one quantity, each with its own standard error,
// consistent with a single true value? Cochran's Q, I² (Higgins and Thompson,
// Stat. Med. 21, 1539, 2002), and the SD tau of the true values between items:
// the Paule-Mandel estimate (Q(tau²) = n - 1) with its Q-profile 95 % interval
// (Viechtbauer, Stat. Med. 26, 37, 2007), which agree by construction; the
// DerSimonian-Laird moment estimate can fall outside that interval when the
// errors differ widely between items.
export function heterogeneity(y, se) {
  const idx = []; for (let i = 0; i < y.length; i++) if (Number.isFinite(y[i]) && se[i] > 0) idx.push(i);
  const n = idx.length, df = n - 1;
  if (n < 3) return null;
  const Qat = t2 => {
    let sw = 0, swy = 0; for (const i of idx) { const w = 1 / (se[i] ** 2 + t2); sw += w; swy += w * y[i]; }
    const mu = swy / sw; let Q = 0; for (const i of idx) Q += (y[i] - mu) ** 2 / (se[i] ** 2 + t2);
    return { Q, mu, seMu: Math.sqrt(1 / sw) };
  };
  const { Q, mu: mu0 } = Qat(0);
  let vy = 0; for (const i of idx) vy += (y[i] - mu0) ** 2;
  const solve = c => solveTau2(t2 => Qat(t2).Q, c, 10 * vy / n);
  const tau2 = solve(df), lo = solve(chi2Quantile(1.959964, df)), hi = solve(chi2Quantile(-1.959964, df));
  const re = Qat(tau2);
  // the plain spread, and the part of it that the standard errors account for
  let s = 0, s2 = 0, ms = 0; for (const i of idx) { s += y[i]; s2 += y[i] ** 2; ms += se[i] ** 2; }
  const mean = s / n, sd = Math.sqrt(Math.max(0, (s2 - n * mean * mean) / (n - 1))), noise = Math.sqrt(ms / n);
  return { n, df, Q, p: chi2Upper(Q, df), I2: Q > df ? (Q - df) / Q : 0, tau: Math.sqrt(tau2), tauLo: Math.sqrt(lo), tauHi: Math.sqrt(hi),
           mu: re.mu, seMu: re.seMu, muFixed: mu0, mean, sd, noise };
}

// Coating coverage on a map: does the coating index vary inside a particle
// beyond each pixel's own error, and between particles beyond that? groups:
// per particle, the indices of its pixels into y and se.
//  - each particle: heterogeneity() over its pixels;
//  - within, pooled: one SD tau_W of the true values inside a particle,
//    Paule-Mandel on the summed Q (each particle about its own mean, sum of
//    n_i - 1 df) with its Q-profile 95 % interval;
//  - between: heterogeneity() of the particle means, each with the standard
//    error its pixels give (their noise and that particle's own spread).
// The true variance then splits into tau_W² within and tau_B² between. The
// plain pooled scatter sd bounds tau_W without trusting the errors: it is the
// limit to quote when the pixels scatter less than their errors predict.
export function coverage(y, se, groups) {
  const parts = groups.map(g => heterogeneity(g.map(i => y[i]), g.map(i => se[i])));
  const gs = [];
  let df = 0, n = 0, ms = 0, vy = 0;
  groups.forEach((g, k) => {
    if (!parts[k]) return;
    const u = g.filter(i => Number.isFinite(y[i]) && se[i] > 0);
    gs.push(u); df += u.length - 1; n += u.length;
    for (const i of u) { ms += se[i] ** 2; vy += (y[i] - parts[k].muFixed) ** 2; }
  });
  if (!gs.length) return null;
  const QW = t2 => {
    let Q = 0;
    for (const g of gs) {
      let sw = 0, swy = 0; for (const i of g) { const w = 1 / (se[i] ** 2 + t2); sw += w; swy += w * y[i]; }
      const mu = swy / sw; for (const i of g) Q += (y[i] - mu) ** 2 / (se[i] ** 2 + t2);
    }
    return Q;
  };
  const solve = c => solveTau2(QW, c, 10 * vy / n), Q = QW(0), tau2 = solve(df);
  const within = { n, df, Q, p: chi2Upper(Q, df), I2: Q > df ? (Q - df) / Q : 0, tau: Math.sqrt(tau2),
                   tauLo: Math.sqrt(solve(chi2Quantile(1.959964, df))), tauHi: Math.sqrt(solve(chi2Quantile(-1.959964, df))),
                   noise: Math.sqrt(ms / n), sd: Math.sqrt(vy / df) };
  const ok = parts.filter(Boolean);
  const between = heterogeneity(ok.map(h => h.mu), ok.map(h => h.seMu));
  const mu = between ? between.mu : ok.reduce((s, h) => s + h.mu, 0) / ok.length;
  const tB2 = between ? between.tau ** 2 : NaN;
  return { parts, within, between, mu, particles: ok.length, shareWithin: tau2 + tB2 > 0 ? tau2 / (tau2 + tB2) : NaN };
}

// Pixels of one particle lag pixels apart along a row or a column: their mean
// of (y_p - y_q)² / (se_p² + se_q²), about 1 when the errors are right and the
// true value does not change over that distance. Where the map's step is
// smaller than the laser spot, the nearest pairs check the errors; a rise with
// distance is real variation on that scale. labels: particle of each pixel, 0 for none.
export function pairRatio(y, se, labels, nx, ny, lag) {
  const ok = p => labels[p] > 0 && Number.isFinite(y[p]) && se[p] > 0;
  let s = 0, n = 0;
  for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
    const p = i * nx + j; if (!ok(p)) continue;
    for (const q of [j + lag < nx ? p + lag : -1, i + lag < ny ? p + lag * nx : -1]) {
      if (q < 0 || labels[q] !== labels[p] || !ok(q)) continue;
      s += (y[p] - y[q]) ** 2 / (se[p] ** 2 + se[q] ** 2); n++;
    }
  }
  return { ratio: n ? s / n : NaN, n };
}

// Moran's I of values at scattered positions, with each point's k nearest
// neighbours weighted equally (row-standardised), and a one-sided permutation
// p for clustering (like next to like). Near -1/(n-1) when values are placed at random.
export function moranI(v, x, y, { k = 6, perms = 199, seed = 3 } = {}) {
  const idx = []; for (let i = 0; i < v.length; i++) if (Number.isFinite(v[i]) && Number.isFinite(x[i]) && Number.isFinite(y[i])) idx.push(i);
  const n = idx.length; if (n < k + 2) return null;
  const nbr = new Int32Array(n * k), bd = new Float64Array(k);
  for (let a = 0; a < n; a++) {
    bd.fill(Infinity); const ia = idx[a];
    for (let b = 0; b < n; b++) {
      if (b === a) continue;
      const d = (x[idx[b]] - x[ia]) ** 2 + (y[idx[b]] - y[ia]) ** 2;
      if (d >= bd[k - 1]) continue;
      let s = k - 1; while (s > 0 && bd[s - 1] > d) { bd[s] = bd[s - 1]; nbr[a * k + s] = nbr[a * k + s - 1]; s--; }
      bd[s] = d; nbr[a * k + s] = b;
    }
  }
  const z = new Float64Array(n); let m = 0; for (let a = 0; a < n; a++) m += v[idx[a]]; m /= n;
  let ss = 0; for (let a = 0; a < n; a++) { z[a] = v[idx[a]] - m; ss += z[a] * z[a]; }
  const stat = zz => { let s = 0; for (let a = 0; a < n; a++) { let t = 0; for (let j = 0; j < k; j++) t += zz[nbr[a * k + j]]; s += zz[a] * t / k; } return s / ss; };
  const I = stat(z), R = rng(seed), perm = Float64Array.from(z);
  let ge = 0;
  for (let t = 0; t < perms; t++) {
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(R() * (i + 1)); const tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp; }
    if (stat(perm) >= I) ge++;
  }
  return { I, expected: -1 / (n - 1), p: (ge + 1) / (perms + 1), n, k };
}

// Spearman's rho with a 95 % interval from Fisher's z (SE 1.03 / sqrt(n - 3), Fieller et al. 1957)
export function spearmanCI(rho, n) {
  if (!Number.isFinite(rho) || n < 5) return [NaN, NaN];
  const z = Math.atanh(Math.max(-0.999999, Math.min(0.999999, rho))), h = 1.959964 * 1.03 / Math.sqrt(n - 3);
  return [Math.tanh(z - h), Math.tanh(z + h)];
}

// Points counted on a grid of pointy-top hexagons of size s (centre to corner), in
// the points' own units: [{ x, y, n }], each hexagon's centre and its count. A
// density for scatter plots too crowded for one dot per point.
export function hexBins(xs, ys, s) {
  const cells = new Map(), r3 = Math.sqrt(3);
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i], y = ys[i]; if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    // axial coordinates, rounded through cube coordinates to the nearest hexagon
    const qf = (r3 / 3 * x - y / 3) / s, rf = (2 / 3 * y) / s, sf = -qf - rf;
    let q = Math.round(qf), r = Math.round(rf); const t = Math.round(sf);
    const dq = Math.abs(q - qf), dr = Math.abs(r - rf), dt = Math.abs(t - sf);
    if (dq > dr && dq > dt) q = -r - t; else if (dr > dt) r = -q - t;
    const key = q * 100003 + r, c = cells.get(key);
    if (c) c.n++; else cells.set(key, { x: s * r3 * (q + r / 2), y: s * 1.5 * r, n: 1 });
  }
  return [...cells.values()];
}
