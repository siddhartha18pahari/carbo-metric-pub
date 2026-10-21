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

// --------------------------------------------------------------------------
// Coverage: a signal floor from detection power, and exact binomial counts
// --------------------------------------------------------------------------

// inverse of the normal CDF (Acklam's rational approximation, relative error ~1e-9)
export function normInv(p) {
  if (!(p > 0 && p < 1)) return p === 0 ? -Infinity : p === 1 ? Infinity : NaN;
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425, q = p < lo ? Math.sqrt(-2 * Math.log(p)) : p > 1 - lo ? Math.sqrt(-2 * Math.log(1 - p)) : 0;
  if (p < lo) return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  if (p > 1 - lo) return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  const r = p - 0.5, s = r * r;
  return (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1);
}

function lnGamma(z) {                                // Lanczos, g = 7
  const g = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
             12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lnGamma(1 - z);
  z -= 1;
  let x = g[0]; for (let i = 1; i < 9; i++) x += g[i] / (z + i);
  const t = z + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

// the regularised incomplete beta function I_x(a, b), by its continued fraction
export function betaInc(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  const cf = (a, b, x) => {
    let c = 1, d = 1 - (a + b) * x / (a + 1); d = 1 / (Math.abs(d) < 1e-300 ? 1e-300 : d);
    let h = d;
    for (let m = 1; m <= 500; m++) {
      for (const num of [m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)), -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))]) {
        d = 1 + num * d; d = 1 / (Math.abs(d) < 1e-300 ? 1e-300 : d);
        c = 1 + num / c; if (Math.abs(c) < 1e-300) c = 1e-300;
        h *= d * c;
      }
      if (Math.abs(d * c - 1) < 1e-15) break;
    }
    return h;
  };
  return x < (a + 1) / (a + b + 2) ? front * cf(a, b, x) / a : 1 - front * cf(b, a, 1 - x) / b;
}

// one-sided (1 - alpha) Clopper-Pearson lower bound on a success fraction, x failures in n
export function binomLower(n, x, alpha = 0.05) {
  if (!(n > 0) || x >= n) return 0;
  if (x <= 0) return alpha ** (1 / n);
  let lo = 0, hi = 1;
  for (let it = 0; it < 80; it++) { const m = 0.5 * (lo + hi); if (betaInc(m, n - x, x + 1) < alpha) lo = m; else hi = m; }
  return 0.5 * (lo + hi);
}

// How many particles certify that at least pSpec of a population is coated, with
// (1 - alpha) confidence. Each particle is detected as coated with probability
// `power` when it is coated and `fpr` when it is not, so the detected fraction must
// have a lower bound above pSpec x power + (1 - pSpec) x fpr; a fully coated population
// gives a detected fraction of power. Failures are rounded, so passing is not monotone
// in n just above the first passing count: this is the count from which every larger
// one passes. Null when no count can pass (power too close to pSpec).
export function certifyCount(pSpec, power = 1, fpr = 0, alpha = 0.05, nmax = 20000) {
  const need = pSpec * power + (1 - pSpec) * fpr, rate = 1 - power;
  if (!(power > need)) return null;
  const pass = n => binomLower(n, Math.round(n * rate), alpha) >= need;
  // failures come back just after each rounding step until the margin outgrows them: the
  // count is one past the last failure, taken once three rounding periods (or 50 counts
  // without failures to round) have passed in a row
  const span = rate > 0 ? Math.ceil(3 / rate) : 50;
  let last = 0;
  for (let n = 1; n <= nmax; n++) {
    if (!pass(n)) last = n;
    else if (n - last > span) return last + 1;
  }
  return null;
}

// The detection power for a coating of index rc against the threshold tau, at G height g:
// P = 1 - Phi[(tau - rc) / sigma_R(g)], sigma_R(g) = sqrt(sigmaD² + rc² sigmaG²) / g. The
// signal floor is the G height at which that power reaches `power`.
export function detectionPower(rc, tau, sigmaD, sigmaG, g) {
  return normCdf((rc - tau) * g / Math.sqrt(sigmaD * sigmaD + rc * rc * sigmaG * sigmaG));
}
export function signalFloor(rc, tau, sigmaD, sigmaG, power = 0.95) {
  return rc > tau ? normInv(power) * Math.sqrt(sigmaD * sigmaD + rc * rc * sigmaG * sigmaG) / (rc - tau) : Infinity;
}

// The floor and the threshold together, from an uncoated reference. Dim spectra scatter more,
// so the reference's 99th percentile depends on how bright the spectra are that it is taken
// over: tau(g) is taken over the reference spectra with G at least g. The floor is the lowest
// g at which g >= signalFloor(rc, tau(g)), with at least minRef reference spectra above it.
// Null when no brightness gets there. gRef, rRef: the reference's G heights and coating index.
export function floorWithReference(gRef, rRef, rc, sigmaD, sigmaG, power = 0.95, { q = 0.99, minRef = 100, steps = 80 } = {}) {
  const idx = Array.from(gRef, (_, i) => i).filter(i => Number.isFinite(gRef[i]) && Number.isFinite(rRef[i])).sort((a, b) => gRef[a] - gRef[b]);
  if (idx.length < minRef) return null;
  const at = k => {                                      // tau over the reference spectra from rank k up
    const r = Float64Array.from(idx.slice(k), i => rRef[i]).sort();
    return { g: gRef[idx[k]], tau: quantileSorted(r, q), n: r.length };
  };
  const top = idx.length - minRef;
  for (let s = 0; s <= steps; s++) {
    const t = at(Math.round(top * s / steps));
    if (t.g >= signalFloor(rc, t.tau, sigmaD, sigmaG, power)) return t;
  }
  return null;
}

// --------------------------------------------------------------------------
// Uniformity beyond the coefficient of variation, over a particle's pixels
// --------------------------------------------------------------------------

// Gini coefficient of a non-negative sample: 0 when every value is equal
export function gini(x) {
  const s = Float64Array.from(x).filter(Number.isFinite).map(v => Math.max(0, v)).sort(), n = s.length;
  let tot = 0, w = 0; for (let i = 0; i < n; i++) { tot += s[i]; w += (i + 1) * s[i]; }
  return n < 2 || !(tot > 0) ? NaN : 2 * w / (n * tot) - (n + 1) / n;
}
// what the Gini reads on noise alone: each value replaced by the level plus its own noise
export function noiseGini(level, se, reps = 8, seed = 11) {
  const R = rng(seed), g = () => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
  let s = 0; for (let k = 0; k < reps; k++) s += gini(Array.from(se, v => level + v * g()));
  return s / reps;
}
// the geometric (log-scale) CV of the positive values
export function logCV(x) {
  const l = Array.from(x).filter(v => v > 0).map(Math.log), n = l.length;
  if (n < 10) return NaN;
  const m = l.reduce((a, b) => a + b, 0) / n, v = l.reduce((a, b) => a + (b - m) ** 2, 0) / n;
  return Math.sqrt(Math.expm1(v));
}
// whether a lognormal fits the positive values better than a normal, by log-likelihood
export function lognormalPreferred(x) {
  const v = Array.from(x).filter(u => u > 0), n = v.length;
  if (n < 30) return null;
  const ll = a => { const m = a.reduce((s, u) => s + u, 0) / n, s2 = a.reduce((s, u) => s + (u - m) ** 2, 0) / n; return -0.5 * n * (Math.log(2 * Math.PI * s2) + 1); };
  const lx = v.map(Math.log);
  return ll(lx) - lx.reduce((s, u) => s + u, 0) > ll(v);
}

// Moran's I on a grid with rook neighbours over a mask (0 = no autocorrelation)
export function gridMoran(img, ok, nx, ny) {
  let s = 0, n = 0; for (let p = 0; p < nx * ny; p++) if (ok[p]) { s += img[p]; n++; }
  if (n < 3) return NaN;
  const m = s / n; let num = 0, den = 0, edges = 0;
  for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
    const p = i * nx + j; if (!ok[p]) continue;
    den += (img[p] - m) ** 2;
    if (j + 1 < nx && ok[p + 1]) { num += (img[p] - m) * (img[p + 1] - m); edges++; }
    if (i + 1 < ny && ok[p + nx]) { num += (img[p] - m) * (img[p + nx] - m); edges++; }
  }
  return edges && den ? n * num / (edges * den) : NaN;
}
// how many independent values n correlated pixels are worth, from the lag-one correlation
export const effectiveN = (n, rho) => { const r = Math.max(-0.99, Math.min(0.99, rho)); return n * (1 - r) / (1 + r); };

// Isotropic experimental variogram over a mask, lags binned by the rounded distance of
// the shift: the classical estimator and the robust one of Cressie and Hawkins.
export function variogram(img, ok, nx, ny, hmax = 15) {
  const sq = new Float64Array(hmax + 1), rb = new Float64Array(hmax + 1), cnt = new Float64Array(hmax + 1);
  for (let dy = 0; dy <= hmax; dy++) for (let dx = -hmax; dx <= hmax; dx++) {
    if (dy === 0 && dx <= 0) continue;
    const h = Math.round(Math.hypot(dx, dy)); if (h < 1 || h > hmax) continue;
    for (let i = 0; i + dy < ny; i++) for (let j = Math.max(0, -dx); j < nx && j + dx < nx; j++) {
      const p = i * nx + j, q = p + dy * nx + dx; if (!ok[p] || !ok[q]) continue;
      const d = img[p] - img[q]; sq[h] += d * d; rb[h] += Math.sqrt(Math.abs(d)); cnt[h]++;
    }
  }
  const out = { lags: [], classical: [], robust: [], pairs: [] };
  for (let h = 1; h <= hmax; h++) {
    if (cnt[h] < 30) continue;
    out.lags.push(h); out.pairs.push(cnt[h]);
    out.classical.push(0.5 * sq[h] / cnt[h]);
    out.robust.push(0.5 * (rb[h] / cnt[h]) ** 4 / (0.457 + 0.494 / cnt[h]));
  }
  return out;
}

const VG_MODELS = { exponential: [t => 1 - Math.exp(-t), 3], gaussian: [t => 1 - Math.exp(-t * t), 1.73], spherical: [t => (t < 1 ? 1.5 * t - 0.5 * t ** 3 : 1), 1] };
// gamma(h) = c0 + c1 f(h / a) for three families, a on a grid and c0, c1 >= 0 by least
// squares; the best family by its residual. The range is identified only if it lies
// within the lags measured (otherwise it is a lower bound).
export function fitVariogram(lags, gam, hmax = 15) {
  if (lags.length < 4) return null;
  const gmax = Math.max(...gam); if (!(gmax > 0)) return null;
  let best = null;
  for (const [model, [f, toRange]] of Object.entries(VG_MODELS)) {
    for (let k = 0; k <= 120; k++) {
      const a = 0.3 * (3 * hmax / 0.3) ** (k / 120), fh = lags.map(h => f(h / a));
      // two-parameter non-negative least squares
      const n = lags.length, sf = fh.reduce((s, v) => s + v, 0), sff = fh.reduce((s, v) => s + v * v, 0);
      const sg = gam.reduce((s, v) => s + v, 0), sfg = fh.reduce((s, v, i) => s + v * gam[i], 0), det = n * sff - sf * sf;
      let c0 = det ? (sg * sff - sf * sfg) / det : sg / n, c1 = det ? (n * sfg - sf * sg) / det : 0;
      if (c0 < 0) { c0 = 0; c1 = sfg / sff; }
      if (c1 < 0) { c1 = 0; c0 = sg / n; }
      const rms = Math.sqrt(gam.reduce((s, v, i) => s + (c0 + c1 * fh[i] - v) ** 2, 0) / n) / gmax;
      if (!best || rms < best.rms) best = { model, nugget: c0, sill: c0 + c1, range: toRange * a, rms };
    }
  }
  best.identified = best.range <= lags[lags.length - 1];
  return best;
}

// Local Getis-Ord Gi* z-scores over a mask, on a (2r + 1) square including the pixel
export function getisOrd(img, ok, nx, ny, r = 2) {
  let s = 0, s2 = 0, n = 0; for (let p = 0; p < nx * ny; p++) if (ok[p]) { s += img[p]; s2 += img[p] ** 2; n++; }
  const z = new Float64Array(nx * ny).fill(NaN); if (n < 3) return z;
  const xbar = s / n, sd = Math.sqrt(Math.max(0, s2 / n - xbar * xbar));
  // integral images of the values and of the mask
  const W = nx + 1, SV = new Float64Array(W * (ny + 1)), SM = new Float64Array(W * (ny + 1));
  for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
    const p = i * nx + j, v = ok[p] ? img[p] : 0, m = ok[p] ? 1 : 0, o = (i + 1) * W + j + 1;
    SV[o] = v + SV[o - 1] + SV[o - W] - SV[o - W - 1]; SM[o] = m + SM[o - 1] + SM[o - W] - SM[o - W - 1];
  }
  const box = (A, i0, j0, i1, j1) => A[(i1 + 1) * W + j1 + 1] - A[i0 * W + j1 + 1] - A[(i1 + 1) * W + j0] + A[i0 * W + j0];
  for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
    const p = i * nx + j; if (!ok[p]) continue;
    const i0 = Math.max(0, i - r), i1 = Math.min(ny - 1, i + r), j0 = Math.max(0, j - r), j1 = Math.min(nx - 1, j + r);
    const sl = box(SV, i0, j0, i1, j1), wl = box(SM, i0, j0, i1, j1);
    const den = sd * Math.sqrt(Math.max((n * wl - wl * wl) / (n - 1), 1e-12));
    z[p] = (sl - xbar * wl) / den;
  }
  return z;
}

// share of the masked pixels significantly low (cold) and high (hot) by Gi*, with the
// false discovery rate held at q across them (Benjamini-Hochberg)
export function hotSpots(img, ok, nx, ny, { r = 2, q = 0.05 } = {}) {
  const z = getisOrd(img, ok, nx, ny, r), idx = [];
  for (let p = 0; p < nx * ny; p++) if (ok[p] && Number.isFinite(z[p])) idx.push(p);
  const m = idx.length; if (!m) return { cold: NaN, hot: NaN, n: 0 };
  const pv = idx.map(p => 2 * (1 - normCdf(Math.abs(z[p])))), order = pv.map((_, i) => i).sort((a, b) => pv[a] - pv[b]);
  let thr = -1; order.forEach((i, k) => { if (pv[i] <= (k + 1) / m * q) thr = pv[i]; });
  let cold = 0, hot = 0; idx.forEach((p, i) => { if (pv[i] <= thr) { if (z[p] < 0) cold++; else hot++; } });
  return { cold: cold / m, hot: hot / m, n: m };
}

// the field with whole b x b blocks moved to other block places: structure inside a
// block survives, so hot spots found on it are what the autocorrelation alone gives
export function blockShuffle(img, ok, nx, ny, b = 8, seed = 5) {
  const bx = Math.floor(nx / b), by = Math.floor(ny / b);
  if (bx < 2 || by < 2) return null;
  const R = rng(seed), order = Array.from({ length: bx * by }, (_, i) => i);
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(R() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  const W = bx * b, Hh = by * b, v = new Float64Array(W * Hh), m = new Uint8Array(W * Hh);
  order.forEach((src, dst) => {
    const si = Math.floor(src / bx) * b, sj = (src % bx) * b, di = Math.floor(dst / bx) * b, dj = (dst % bx) * b;
    for (let a = 0; a < b; a++) for (let c = 0; c < b; c++) { const sp = (si + a) * nx + sj + c, dp = (di + a) * W + dj + c; v[dp] = img[sp]; m[dp] = ok[sp]; }
  });
  return { img: v, ok: m, nx: W, ny: Hh };
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

// --------------------------------------------------------------------------
// More uniformity measures, intervals, and checks on the measurement
// --------------------------------------------------------------------------

// 1.4826 x the median absolute deviation: the SD of a normal sample, robust to outliers
export function robustSD(x) {
  const s = Float64Array.from(x).filter(Number.isFinite).sort(); if (!s.length) return NaN;
  const m = quantileSorted(s, 0.5);
  return 1.4826 * quantileSorted(Float64Array.from(s, v => Math.abs(v - m)).sort(), 0.5);
}

// Coverage when some spectra cannot be judged (a band parameter on a fit bound, or the
// coating band fitted with zero height): x of n judged are coated and e are left out. The
// bounds count every left-out spectrum as uncoated (lo) or coated (hi); the exact lower
// bound is taken on the worst case.
export function boundedCoverage(x, n, e, alpha = 0.05) {
  const N = n + e;
  return { c: n ? x / n : NaN, lo: N ? x / N : NaN, hi: N ? (x + e) / N : NaN, lower: N ? binomLower(N, N - x, alpha) : NaN };
}

// share of the total held by the lowest q of the values (q itself if all are equal)
export function lowShare(x, q = 0.1) {
  const s = Float64Array.from(x).filter(Number.isFinite).map(v => Math.max(0, v)).sort(), n = s.length;
  const tot = s.reduce((a, b) => a + b, 0); if (n < 2 || !(tot > 0)) return NaN;
  const k = Math.max(Math.round(q * n), 1); let low = 0; for (let i = 0; i < k; i++) low += s[i];
  return low / tot;
}
// quartile coefficient of dispersion (q3 - q1) / (q3 + q1): a spread that needs no mean
export function qcd(x) {
  const s = Float64Array.from(x).filter(Number.isFinite).sort(), a = quantileSorted(s, 0.25), b = quantileSorted(s, 0.75);
  return a + b > 0 ? (b - a) / (b + a) : NaN;
}
export function qRatio(x, lo = 0.1, hi = 0.9) {
  const s = Float64Array.from(x).filter(Number.isFinite).sort(), a = quantileSorted(s, lo);
  return a > 0 ? quantileSorted(s, hi) / a : NaN;
}
// Shannon entropy of the histogram of the values over `bins` equal bins from 0 to their
// 99.5th percentile, over ln(bins): 1 when the values spread evenly over the range
export function evenness(x, bins = 32) {
  const s = Float64Array.from(x).filter(Number.isFinite).sort(), hi = quantileSorted(s, 0.995);
  if (!(hi > 0)) return NaN;
  const h = new Float64Array(bins);
  for (const v of s) h[Math.min(bins - 1, Math.floor(Math.min(Math.max(v, 0), hi) / hi * bins))]++;
  let e = 0; for (const c of h) if (c) { const p = c / s.length; e -= p * Math.log(p); }
  return e / Math.log(bins);
}
// skewness and excess kurtosis (the moment estimators, as scipy.stats gives them)
export function moments(x) {
  const v = Array.from(x).filter(Number.isFinite), n = v.length; if (n < 3) return { skew: NaN, kurt: NaN };
  const m = v.reduce((a, b) => a + b, 0) / n; let m2 = 0, m3 = 0, m4 = 0;
  for (const u of v) { const d = u - m; m2 += d * d; m3 += d ** 3; m4 += d ** 4; }
  m2 /= n; m3 /= n; m4 /= n;
  return m2 > 0 ? { skew: m3 / m2 ** 1.5, kurt: m4 / (m2 * m2) - 3 } : { skew: NaN, kurt: NaN };
}

// Geary's c on a grid with rook neighbours over a mask: 1 for no autocorrelation, below 1
// when neighbours resemble each other
export function gridGeary(img, ok, nx, ny) {
  let s = 0, n = 0; for (let p = 0; p < nx * ny; p++) if (ok[p]) { s += img[p]; n++; }
  if (n < 3) return NaN;
  const m = s / n; let num = 0, den = 0, edges = 0;
  for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
    const p = i * nx + j; if (!ok[p]) continue;
    den += (img[p] - m) ** 2;
    if (j + 1 < nx && ok[p + 1]) { num += (img[p] - img[p + 1]) ** 2; edges++; }
    if (i + 1 < ny && ok[p + nx]) { num += (img[p] - img[p + nx]) ** 2; edges++; }
  }
  return edges && den ? (n - 1) * num / (2 * edges * den) : NaN;
}
// p-values for Moran's I and Geary's c by shuffling the values over the same mask
// (clustering is a large I or a small c)
export function spatialPermutation(img, ok, nx, ny, nPerm = 199, seed = 3) {
  const I = gridMoran(img, ok, nx, ny), c = gridGeary(img, ok, nx, ny), idx = [];
  for (let p = 0; p < nx * ny; p++) if (ok[p]) idx.push(p);
  if (!Number.isFinite(I) || !Number.isFinite(c)) return { I, c, pI: NaN, pC: NaN };
  const R = rng(seed), vals = idx.map(p => img[p]), work = new Float64Array(nx * ny);
  let gi = 0, gc = 0;
  for (let k = 0; k < nPerm; k++) {
    for (let i = vals.length - 1; i > 0; i--) { const j = Math.floor(R() * (i + 1)); [vals[i], vals[j]] = [vals[j], vals[i]]; }
    idx.forEach((p, i) => { work[p] = vals[i]; });
    if (gridMoran(work, ok, nx, ny) >= I) gi++;
    if (gridGeary(work, ok, nx, ny) <= c) gc++;
  }
  return { I, c, pI: (gi + 1) / (nPerm + 1), pC: (gc + 1) / (nPerm + 1) };
}

// The noise of a band-height image from differences between neighbouring pixels (rows and
// columns, both in the mask): their robust SD over sqrt 2. Neighbours share the smooth
// field, so the differences leave the noise plus any real change at one pixel, and the
// estimate errs high (conservative for anything noise-corrected with it).
export function neighbourNoise(img, ok, nx, ny) {
  const d = [];
  for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
    const p = i * nx + j; if (!ok[p]) continue;
    if (j + 1 < nx && ok[p + 1]) d.push(img[p + 1] - img[p]);
    if (i + 1 < ny && ok[p + nx]) d.push(img[p + nx] - img[p]);
  }
  return { sd: d.length >= 10 ? robustSD(d) / Math.SQRT2 : NaN, n: d.length };
}

// the median of a, reordering it (Hoare's selection, linear time on average): as quantileSorted at 0.5
function medianSelect(a) {
  const n = a.length; if (!n) return NaN;
  const kth = k => {
    let lo = 0, hi = n - 1;
    while (lo < hi) {
      const x = a[(lo + hi) >> 1]; let i = lo, j = hi;
      while (i <= j) { while (a[i] < x) i++; while (a[j] > x) j--; if (i <= j) { const t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; } }
      if (k <= j) hi = j; else if (k >= i) lo = i; else return a[k];
    }
    return a[k];
  };
  const hi = kth(n >> 1);
  if (n % 2) return hi;
  let lo = -Infinity; for (let i = 0; i < n >> 1; i++) if (a[i] > lo) lo = a[i];   // the largest of the lower half
  return 0.5 * (lo + hi);
}

// Intervals for a particle's within-particle measures by resampling whole b x b blocks of
// pixels (8 x 8 by default), so that neighbours that share their noise or their patch move
// together: the observed CV (robust SD over the median), the excess CV beyond the pixels'
// own errors, Gini, QCD, the percentile uniformity U = (q95 - q5) / (q95 + q5) and, with a
// threshold tau, the covered share (alone with coverOnly). Percentile 95 % intervals; null with
// fewer than 4 blocks.
export function blockBootstrap(img, sig, ok, nx, ny, { b = 8, B = 500, seed = 13, tau = NaN, coverOnly = false } = {}) {
  const cells = [];
  for (let bi = 0; bi < ny; bi += b) for (let bj = 0; bj < nx; bj += b) {
    const v = [], s = [];
    for (let i = bi; i < Math.min(ny, bi + b); i++) for (let j = bj; j < Math.min(nx, bj + b); j++) { const p = i * nx + j; if (ok[p]) { v.push(img[p]); s.push(sig[p]); } }
    if (v.length) cells.push([Float64Array.from(v), Float64Array.from(s)]);
  }
  if (cells.length < 4) return null;
  // the measures of one sample: v is sorted in place once, and every measure read from it
  const measures = (v, s) => {
    const n = v.length; let above = 0;
    if (Number.isFinite(tau)) for (let i = 0; i < n; i++) if (v[i] > tau) above++;
    if (coverOnly) return { cover: above / n };
    let ss = 0; for (let i = 0; i < n; i++) ss += s[i] * s[i];
    v.sort();
    const med = quantileSorted(v, 0.5); if (!(med > 0)) return null;
    const dev = DEV.subarray(0, n); for (let i = 0; i < n; i++) dev[i] = Math.abs(v[i] - med);
    const cvObs = 1.4826 * medianSelect(dev) / med, cvNoise = Math.sqrt(ss / n) / med;
    let tot = 0, wsum = 0; for (let i = 0; i < n; i++) { const x = Math.max(0, v[i]); tot += x; wsum += (i + 1) * x; }   // Gini, as gini()
    const q = k => quantileSorted(v, k), [q5, q25, q75, q95] = [q(0.05), q(0.25), q(0.75), q(0.95)];
    return { cvObs, cvExcess: Math.sqrt(Math.max(cvObs * cvObs - cvNoise * cvNoise, 0)), gini: n < 2 || !(tot > 0) ? NaN : 2 * wsum / (n * tot) - (n + 1) / n,
             qcd: q25 + q75 > 0 ? (q75 - q25) / (q75 + q25) : NaN, U: q95 + q5 > 0 ? (q95 - q5) / (q95 + q5) : NaN, cover: Number.isFinite(tau) ? above / n : NaN };
  };
  // one set of buffers, as long as the largest resample can be, reused by every resample
  const size = cells.length * b * b, V = new Float64Array(size), SG = new Float64Array(size), DEV = new Float64Array(size);
  const fill = pick => { let o = 0; for (const c of pick) { V.set(c[0], o); SG.set(c[1], o); o += c[0].length; } return [V.subarray(0, o), SG.subarray(0, o)]; };
  const est = measures(...fill(cells)); if (!est) return null;
  const R = rng(seed), draws = Object.fromEntries(Object.keys(est).map(k => [k, []]));
  for (let r = 0; r < B; r++) {
    const m = measures(...fill(Array.from(cells, () => cells[Math.floor(R() * cells.length)])));
    if (m) for (const k in m) draws[k].push(m[k]);
  }
  const ci = {};
  for (const [k, d] of Object.entries(draws)) { const s = Float64Array.from(d).filter(Number.isFinite).sort(); ci[k] = s.length ? [quantileSorted(s, 0.025), quantileSorted(s, 0.975)] : [NaN, NaN]; }
  return { est, ci, blocks: cells.length, b };
}

// Two-sample Kolmogorov-Smirnov distance with its asymptotic p-value (Stephens' correction). The p-value
// can be taken at sizes other than the sample counts: the effective numbers of correlated pixels.
export function ksTest(a, b, nA = NaN, nB = NaN) {
  const x = Float64Array.from(a).filter(Number.isFinite).sort(), y = Float64Array.from(b).filter(Number.isFinite).sort();
  const n = x.length, m = y.length; if (!n || !m) return { d: NaN, p: NaN };
  let i = 0, j = 0, d = 0;
  while (i < n && j < m) { const v = Math.min(x[i], y[j]); while (i < n && x[i] <= v) i++; while (j < m && y[j] <= v) j++; d = Math.max(d, Math.abs(i / n - j / m)); }
  const ea = Number.isFinite(nA) ? nA : n, eb = Number.isFinite(nB) ? nB : m, en = Math.sqrt(ea * eb / (ea + eb)), lam = (en + 0.12 + 0.11 / en) * d;
  let p = 0; for (let k = 1; k <= 100; k++) { const t = 2 * (k % 2 ? 1 : -1) * Math.exp(-2 * k * k * lam * lam); p += t; if (Math.abs(t) < 1e-10) break; }
  return { d, p: lam < 0.2 ? 1 : Math.max(0, Math.min(1, p)) };
}

// Band heights without a band fit, for a check on the fitted ratio: a straight baseline
// through 1030-1090 and 1700-1740 cm-1; each height the mean of the residual over 7
// channels at the band's position. The error of a height combines channel noise (robust SD
// of successive channel differences in 1700-1790 cm-1, over sqrt 2) with baseline
// variation (the spread over the spectra of the residual at 1765 cm-1, where no band lies).
// Null when the spectra do not cover those windows. cube holds K channels per spectrum.
export function fitFreeHeights(cube, K, wn, list, positions) {
  const win = (lo, hi) => { const o = []; for (let k = 0; k < K; k++) if (wn[k] >= lo && wn[k] <= hi) o.push(k); return o; };
  const base = [...win(1030, 1090), ...win(1700, 1740)], nwin = win(1700, 1790), HALF = 3;
  const at = pos => { let best = 0; for (let k = 1; k < K; k++) if (Math.abs(wn[k] - pos) < Math.abs(wn[best] - pos)) best = k; return best; };
  const centres = [...positions, 1765].map(at);
  if (win(1030, 1090).length < 3 || win(1700, 1740).length < 3 || nwin.length < 8 || !(wn[0] <= 1030 && wn[K - 1] >= 1790)
      || centres.some(c => c - HALF < 0 || c + HALF >= K)) return null;
  const nb = base.length, mx = base.reduce((t, k) => t + wn[k], 0) / nb, sxx = base.reduce((t, k) => t + (wn[k] - mx) ** 2, 0);
  const H = centres.map(() => new Float64Array(list.length)), sH = new Float64Array(list.length), sig = new Float64Array(list.length);
  list.forEach((p, i) => {
    const o = p * K; let my = 0, sxy = 0;
    for (const k of base) my += cube[o + k] / nb;
    for (const k of base) sxy += (wn[k] - mx) * (cube[o + k] - my);
    const slope = sxy / sxx, res = k => cube[o + k] - (my + slope * (wn[k] - mx));
    centres.forEach((c, j) => { let s = 0; for (let k = c - HALF; k <= c + HALF; k++) s += res(k); H[j][i] = s / (2 * HALF + 1); });
    const d = []; for (let t = 1; t < nwin.length; t++) d.push(cube[o + nwin[t]] - cube[o + nwin[t - 1]]);
    sig[i] = robustSD(d) / Math.SQRT2; sH[i] = sig[i] * Math.sqrt(1 / (2 * HALF + 1) + 1 / nb);
  });
  const nul = H.pop(), sBase = robustSD(nul);
  return { heights: H, se: Float64Array.from(sH, s => Math.hypot(s, sBase)), sBase, sigma: median(sig) };
}

// How many particles, and maps, a mean needs to reach a relative standard error se, from
// SE² = CVb² / np + (CVw² + CVn²) / (np k): one spectrum per particle (k = 1) or maps of k
// independent pixels. No number of pixels removes the between-particle term, so the maps
// needed never fall below (CVb / se)².
export function nestedSize(se, cvB, cvW, cvN, k) {
  const v = cvB * cvB, w = cvW * cvW + cvN * cvN;
  return { population: Math.ceil((v + w) / (se * se)), maps: Math.ceil((v + w / Math.max(1, k)) / (se * se)), floor: Math.ceil(v / (se * se)) };
}

// --------------------------------------------------------------------------
// Dispersion, decomposition and Gaussian splats: cross-checks on the sample and the maps
// --------------------------------------------------------------------------

// How well the particles are spread on the slide (population mode). For each particle: the distance to
// its nearest neighbour, centre to centre, and the smallest gap to any neighbour between their equivalent
// circles. The Clark-Evans ratio, the mean nearest-neighbour distance over 0.5 / sqrt(density), is about 1
// for particles dropped at random, below 1 when they clump and up to 2.15 for a regular lattice (no edge
// correction, so it reads a little high). Also the share touching a neighbour (gap <= 0), and the share of
// objects so large (area above 3 x the median) that they are likely touching particles segmented as one,
// and the share that is either (crowded).
export function dispersion(xs, ys, ds, areas, fieldArea) {
  const n = xs.length; if (n < 3) return null;
  const nn = new Float64Array(n), gap = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let best = Infinity, bg = Infinity;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const d = Math.hypot(xs[i] - xs[j], ys[i] - ys[j]), g = d - 0.5 * (ds[i] + ds[j]);
      if (d < best) best = d;
      if (g < bg) bg = g;
    }
    nn[i] = best; gap[i] = bg;
  }
  const s = Float64Array.from(nn).sort(), mean = nn.reduce((t, v) => t + v, 0) / n, medA = median(areas);
  let touching = 0, merged = 0, crowded = 0;
  for (let i = 0; i < n; i++) { const t = gap[i] <= 0, m = areas[i] > 3 * medA; if (t) touching++; if (m) merged++; if (t || m) crowded++; }
  return { n, nnMedian: quantileSorted(s, 0.5), nnP10: quantileSorted(s, 0.1), clarkEvans: 2 * mean * Math.sqrt(n / fieldArea),
           touching: touching / n, merged: merged / n, crowded: crowded / n, nn, gap };
}

// Non-negative matrix factorisation V ~ W H (V: n spectra of K channels, row by row; H: k component spectra)
// by hierarchical alternating least squares (Cichocki and Phan 2009): each component's amounts, then its
// spectrum, solved in turn with the others held, clipped at zero. From the starting spectra H0, so the
// result is the same every time. Rows of H come back scaled to unit sum, W carrying the amounts.
export function nmf(V, n, K, H0, { iters = 200, eps = 1e-12 } = {}) {
  const k = H0.length, H = new Float64Array(k * K), W = new Float64Array(n * k);
  H0.forEach((r, a) => { for (let c = 0; c < K; c++) H[a * K + c] = Math.max(r[c], 0); });
  W.set(nmfProject(V, n, K, H, k, { iters: 20 }));
  const VHt = new Float64Array(n * k), HHt = new Float64Array(k * k), WtV = new Float64Array(k * K), WtW = new Float64Array(k * k);
  for (let it = 0; it < iters; it++) {
    // the amounts, one component at a time
    for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) { let t = 0; for (let c = 0; c < K; c++) t += H[a * K + c] * H[b * K + c]; HHt[a * k + b] = t; }
    for (let i = 0; i < n; i++) for (let a = 0; a < k; a++) { let t = 0; for (let c = 0; c < K; c++) t += V[i * K + c] * H[a * K + c]; VHt[i * k + a] = t; }
    for (let a = 0; a < k; a++) {
      const d = HHt[a * k + a]; if (!(d > eps)) continue;
      for (let i = 0; i < n; i++) { let t = VHt[i * k + a]; for (let b = 0; b < k; b++) t -= W[i * k + b] * HHt[b * k + a]; W[i * k + a] = Math.max(0, W[i * k + a] + t / d); }
    }
    // then the spectra
    WtV.fill(0); WtW.fill(0);
    for (let i = 0; i < n; i++) for (let a = 0; a < k; a++) {
      const w = W[i * k + a]; if (!w) continue;
      for (let c = 0; c < K; c++) WtV[a * K + c] += w * V[i * K + c];
      for (let b = 0; b < k; b++) WtW[a * k + b] += w * W[i * k + b];
    }
    for (let a = 0; a < k; a++) {
      const d = WtW[a * k + a]; if (!(d > eps)) continue;
      for (let c = 0; c < K; c++) { let t = WtV[a * K + c]; for (let b = 0; b < k; b++) t -= WtW[a * k + b] * H[b * K + c]; H[a * K + c] = Math.max(0, H[a * K + c] + t / d); }
    }
  }
  for (let a = 0; a < k; a++) {
    let t = 0; for (let c = 0; c < K; c++) t += H[a * K + c];
    if (t > 0) { for (let c = 0; c < K; c++) H[a * K + c] /= t; for (let i = 0; i < n; i++) W[i * k + a] *= t; }
  }
  return { W, H, k };
}
// the amounts of fixed component spectra H (k rows of K channels) in each of n spectra: non-negative
// least squares by the same component-wise updates with H held
export function nmfProject(V, n, K, H, k, { iters = 100, eps = 1e-12 } = {}) {
  const W = new Float64Array(n * k), HHt = new Float64Array(k * k), VHt = new Float64Array(n * k);
  for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) { let t = 0; for (let c = 0; c < K; c++) t += H[a * K + c] * H[b * K + c]; HHt[a * k + b] = t; }
  for (let i = 0; i < n; i++) for (let a = 0; a < k; a++) { let t = 0; for (let c = 0; c < K; c++) t += V[i * K + c] * H[a * K + c]; VHt[i * k + a] = t; }
  for (let it = 0; it < iters; it++) for (let i = 0; i < n; i++) for (let a = 0; a < k; a++) {
    const d = HHt[a * k + a]; if (!(d > eps)) continue;
    let t = VHt[i * k + a]; for (let b = 0; b < k; b++) t -= W[i * k + b] * HHt[b * k + a];
    W[i * k + a] = Math.max(0, W[i * k + a] + t / d);
  }
  return W;
}

// let the page draw and take input: a message on a fresh channel, which a hidden tab does not slow down
// as it does timers
const yieldToPage = () => (typeof MessageChannel === "undefined" ? new Promise(res => setTimeout(res, 0))
  : new Promise(res => { const ch = new MessageChannel(); ch.port1.onmessage = () => { ch.port1.close(); res(); }; ch.port2.postMessage(0); }));

// Two-dimensional anisotropic Gaussian splatting of a field sampled at points X (pairs in [0, 1]²):
// f(x) = b + sum_i a_i exp(-q_i / 2), q_i = (u1 / s1)² + (u2 / s2)², u = R(-theta_i)(x - mu_i). Centres,
// log widths and orientations follow Adam on the analytic gradient of the mean squared error; amplitudes
// and offset are re-solved by ridge least squares at intervals (the field is linear in them); primitives
// are added where the residual is largest and dropped where their amplitude vanishes. A port of the
// fitter used for the coating maps; it yields to the page every `yieldEvery` iterations.
export async function fitSplats(X0, y0, { nInit = 16, nMax = 64, iters = 600, seed = 0, lrMu = 3e-3, lrS = 1e-2, lrTheta = 1e-2,
  densifyEvery = 100, densifyUntil = 0.6, lstsqEvery = 25, sMin = 0.01, sMax = 0.6, ridge = 1e-8, yieldEvery = 25, onTick = null } = {}) {
  const N = y0.length, R = rng(seed + 1), lo = Math.log(sMin), hi = Math.log(sMax);
  const width = Math.min(Math.max(0.5 / Math.sqrt(nInit), sMin), sMax), Q_MAX = 40, REACH = Math.sqrt(Q_MAX);   // exp(-20): a primitive's reach
  // the points sorted by x, so that each primitive is evaluated only over the points within its reach
  const order = Array.from({ length: N }, (_, i) => i).sort((p, q) => X0[2 * p] - X0[2 * q]);
  const px = Float64Array.from(order, i => X0[2 * i]), py = Float64Array.from(order, i => X0[2 * i + 1]), y = Float64Array.from(order, i => y0[i]);
  const first = v => { let l = 0, h = N; while (l < h) { const m = (l + h) >> 1; if (px[m] < v) l = m + 1; else h = m; } return l; };
  // the field's parameters, in typed arrays of capacity nMax
  const mu = new Float64Array(2 * nMax), ls = new Float64Array(2 * nMax), th = new Float64Array(nMax), a = new Float64Array(nMax);
  let n = 0, b = y.reduce((t, v) => t + v, 0) / N;
  const add = (x, yy, w) => { mu[2 * n] = x; mu[2 * n + 1] = yy; ls[2 * n] = ls[2 * n + 1] = w; th[n] = R() * Math.PI; a[n] = 0; n++; };
  { const pick = Array.from({ length: N }, (_, i) => i); for (let i = 0; i < Math.min(nInit, N, nMax); i++) { const j = i + Math.floor(R() * (N - i)); [pick[i], pick[j]] = [pick[j], pick[i]]; add(px[pick[i]], py[pick[i]], Math.log(width)); } }
  // per primitive (row) and point: the Gaussian and the rotated offsets, over [from, to) of the sorted points
  const Gm = new Float64Array(nMax * N), U1 = new Float64Array(nMax * N), U2 = new Float64Array(nMax * N), from = new Int32Array(nMax), to = new Int32Array(nMax), r = new Float64Array(N);
  const basis = () => {
    for (let j = 0; j < n; j++) {
      const c = Math.cos(th[j]), s = Math.sin(th[j]), s1 = Math.exp(ls[2 * j]), s2 = Math.exp(ls[2 * j + 1]), i1 = 1 / (s1 * s1), i2 = 1 / (s2 * s2), mx = mu[2 * j], my = mu[2 * j + 1];
      const ex = REACH * Math.hypot(s1 * c, s2 * s), o = j * N, f0 = first(mx - ex), t0 = first(mx + ex);
      Gm.fill(0, o + from[j], o + to[j]); from[j] = f0; to[j] = t0;
      for (let i = f0; i < t0; i++) {
        const d0 = px[i] - mx, d1 = py[i] - my, u1 = c * d0 + s * d1, u2 = c * d1 - s * d0, q = u1 * u1 * i1 + u2 * u2 * i2;
        U1[o + i] = u1; U2[o + i] = u2; Gm[o + i] = q < Q_MAX ? Math.exp(-0.5 * q) : 0;
      }
    }
  };
  const solveAmplitudes = () => {
    const m = n + 1, M = new Float64Array(m * m), rhs = new Float64Array(m);
    for (let p = 0; p < n; p++) {
      const op = p * N;
      let t = 0, u = 0; for (let i = from[p]; i < to[p]; i++) { t += Gm[op + i] * y[i]; u += Gm[op + i]; }
      rhs[p] = t; M[p * m + n] = u;
      for (let q = p; q < n; q++) {
        const oq = q * N, i0 = Math.max(from[p], from[q]), i9 = Math.min(to[p], to[q]);
        let v = 0; for (let i = i0; i < i9; i++) v += Gm[op + i] * Gm[oq + i];
        M[p * m + q] = v;
      }
    }
    rhs[n] = y.reduce((t, v) => t + v, 0); M[n * m + n] = N;
    let tr = 0; for (let p = 0; p < m; p++) { for (let q = 0; q < p; q++) M[p * m + q] = M[q * m + p]; tr += M[p * m + p]; }
    for (let p = 0; p < m; p++) M[p * m + p] += ridge * Math.max(tr / m, 1e-12);
    const Mi = pinvSym(M, m);
    for (let p = 0; p < m; p++) { let t = 0; for (let q = 0; q < m; q++) t += Mi[p * m + q] * rhs[q]; if (p < n) a[p] = t; else b = t; }
  };
  const residual = () => {
    for (let i = 0; i < N; i++) r[i] = b - y[i];
    for (let j = 0; j < n; j++) { const o = j * N, aj = a[j]; for (let i = from[j]; i < to[j]; i++) r[i] += aj * Gm[o + i]; }
    let ss = 0; for (let i = 0; i < N; i++) ss += r[i] * r[i];
    return Math.sqrt(ss / N);
  };
  // Adam's moments, reset whenever primitives are added or dropped
  let M1 = new Float64Array(5 * nMax), M2 = new Float64Array(5 * nMax), step = 0;
  const grad = new Float64Array(5 * nMax), hist = [];
  for (let it = 1; it <= iters; it++) {
    basis();
    if (it === 1 || it % lstsqEvery === 0) solveAmplitudes();
    const rmse = residual();
    for (let j = 0; j < n; j++) {
      const c = Math.cos(th[j]), s = Math.sin(th[j]), i1 = Math.exp(-2 * ls[2 * j]), i2 = Math.exp(-2 * ls[2 * j + 1]), aj = a[j], o = j * N;
      let gx = 0, gy = 0, g1 = 0, g2 = 0, gt = 0;
      for (let i = from[j]; i < to[j]; i++) {
        const G = Gm[o + i]; if (!G) continue;
        const gq = -(r[i] * aj * G) / N, u1 = U1[o + i], u2 = U2[o + i], gu1 = 2 * gq * u1 * i1, gu2 = 2 * gq * u2 * i2;
        gx += -gu1 * c + gu2 * s; gy += -gu1 * s - gu2 * c;
        g1 -= gu1 * u1; g2 -= gu2 * u2; gt += 2 * gq * u1 * u2 * (i1 - i2);
      }
      grad[5 * j] = gx; grad[5 * j + 1] = gy; grad[5 * j + 2] = g1; grad[5 * j + 3] = g2; grad[5 * j + 4] = gt;
    }
    step++;
    const c1 = 1 - 0.9 ** step, c2 = 1 - 0.999 ** step;
    for (let j = 0; j < n; j++) for (let q = 0; q < 5; q++) {
      const z = 5 * j + q, gz = grad[z];
      M1[z] = 0.9 * M1[z] + 0.1 * gz; M2[z] = 0.999 * M2[z] + 0.001 * gz * gz;
      const d = (q < 2 ? lrMu : q < 4 ? lrS : lrTheta) * (M1[z] / c1) / (Math.sqrt(M2[z] / c2) + 1e-8);
      if (q < 2) mu[2 * j + q] = Math.min(1.05, Math.max(-0.05, mu[2 * j + q] - d));
      else if (q < 4) ls[2 * j + q - 2] = Math.min(hi, Math.max(lo, ls[2 * j + q - 2] - d));
      else th[j] -= d;
    }
    if (it % densifyEvery === 0 && it <= densifyUntil * iters) {
      // drop primitives whose amplitude has vanished, then add some where the residual is largest
      const medA = median(Array.from(a.subarray(0, n), Math.abs));
      let k = 0;
      for (let j = 0; j < n; j++) {
        Gm.fill(0, j * N + from[j], j * N + to[j]);
        if (Math.abs(a[j]) > 1e-3 * medA) { mu[2 * k] = mu[2 * j]; mu[2 * k + 1] = mu[2 * j + 1]; ls[2 * k] = ls[2 * j]; ls[2 * k + 1] = ls[2 * j + 1]; th[k] = th[j]; a[k] = a[j]; k++; }
      }
      from.fill(0); to.fill(0); n = k;
      const more = Math.min(Math.max(Math.floor(n / 2), 1), nMax - n);
      if (more > 0) {
        const worst = Array.from({ length: N }, (_, i) => i).sort((p, q) => Math.abs(r[q]) - Math.abs(r[p])), taken = [];
        for (const j of worst) {
          if (taken.every(t => Math.hypot(px[j] - px[t], py[j] - py[t]) > 0.03)) taken.push(j);
          if (taken.length === more) break;
        }
        const w = Math.log(Math.min(Math.max(width / 2, sMin), sMax));
        for (const j of taken) add(px[j], py[j], w);
      }
      basis(); solveAmplitudes();
      M1 = new Float64Array(5 * nMax); M2 = new Float64Array(5 * nMax); step = 0;
    }
    if (it === 1 || it % 50 === 0 || it === iters) hist.push({ it, rmse, n });
    if (it % yieldEvery === 0) { if (onTick) onTick(it / iters); await yieldToPage(); }
  }
  basis(); solveAmplitudes();
  const field = { mu: mu.slice(0, 2 * n), ls: ls.slice(0, 2 * n), th: th.slice(0, n), a: a.slice(0, n), b, n,
    predict(P) {
      const m = P.length / 2, out = new Float64Array(m).fill(this.b);
      for (let j = 0; j < this.n; j++) {
        const c = Math.cos(this.th[j]), s = Math.sin(this.th[j]), i1 = Math.exp(-2 * this.ls[2 * j]), i2 = Math.exp(-2 * this.ls[2 * j + 1]);
        for (let i = 0; i < m; i++) {
          const d0 = P[2 * i] - this.mu[2 * j], d1 = P[2 * i + 1] - this.mu[2 * j + 1], u1 = c * d0 + s * d1, u2 = c * d1 - s * d0, q = u1 * u1 * i1 + u2 * u2 * i2;
          if (q < Q_MAX) out[i] += this.a[j] * Math.exp(-0.5 * q);
        }
      }
      return out;
    } };
  return { field, hist };
}
// the spatial scale of a splat field in pixels (S pixels span 0 to 1): the median radius sqrt(s1 s2)
// weighted by each primitive's amplitude times its area, and the median elongation
export function splatScale(field, S) {
  if (!field.n) return { radius: NaN, elongation: NaN };
  const rows = Array.from({ length: field.n }, (_, j) => {
    const s1 = Math.exp(field.ls[2 * j]) * (S - 1), s2 = Math.exp(field.ls[2 * j + 1]) * (S - 1);
    return { r: Math.sqrt(s1 * s2), w: Math.abs(field.a[j]) * s1 * s2, e: Math.max(s1, s2) / Math.min(s1, s2) };
  }).sort((p, q) => p.r - q.r);
  const tot = rows.reduce((t, q) => t + q.w, 0); let cum = 0, radius = rows[rows.length - 1].r;
  for (const q of rows) { cum += q.w; if (cum / tot >= 0.5) { radius = q.r; break; } }
  return { radius, elongation: median(rows.map(q => q.e)) };
}
// normalised Gaussian smoothing of an image over a mask (the masked pixels only), evaluated everywhere:
// the local smoother the splats are compared with (scipy's gaussian_filter, truncated at 4 sigma)
export function gaussSmooth(img, ok, nx, ny, sigma = 1) {
  const R = Math.ceil(4 * sigma), w = Array.from({ length: 2 * R + 1 }, (_, i) => Math.exp(-0.5 * ((i - R) / sigma) ** 2));
  const pass = (src, horiz) => {
    const out = new Float64Array(nx * ny);
    for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
      let t = 0;
      for (let k = -R; k <= R; k++) { const ii = horiz ? i : i + k, jj = horiz ? j + k : j; if (ii >= 0 && ii < ny && jj >= 0 && jj < nx) t += w[k + R] * src[ii * nx + jj]; }
      out[i * nx + j] = t;
    }
    return out;
  };
  const num = pass(pass(Float64Array.from(img, (v, p) => (ok[p] ? v : 0)), true), false), den = pass(pass(Float64Array.from(ok, v => (v ? 1 : 0)), true), false);
  return Float64Array.from(num, (v, p) => (den[p] > 1e-6 ? v / den[p] : NaN));
}

// --------------------------------------------------------------------------
// Between maps: do samples differ, with the particle (or map) as the unit?
// --------------------------------------------------------------------------

// the regularised upper incomplete gamma Q(a, x), by its series below a + 1 and its continued fraction
// above (Numerical Recipes): the chi-square upper tail is Q(df / 2, x / 2), exact at any df
export function gammaQ(a, x) {
  if (!(x > 0)) return 1;
  const lg = lnGamma(a);
  if (x < a + 1) {
    let sum = 1 / a, del = sum, ap = a;
    for (let n = 0; n < 500; n++) { ap++; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-15) break; }
    return Math.max(0, 1 - sum * Math.exp(-x + a * Math.log(x) - lg));
  }
  let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (let i = 1; i < 500; i++) {
    const an = -i * (i - a); b += 2;
    d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
    c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
    d = 1 / d; const del = d * c; h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return Math.min(1, Math.exp(-x + a * Math.log(x) - lg) * h);
}

// Kruskal-Wallis H over groups of values (ranks over all of them, ties averaged and corrected), with its
// chi-square p on groups - 1 degrees of freedom
export function kruskalWallis(groups) {
  const all = []; groups.forEach((g, k) => { for (const v of g) if (Number.isFinite(v)) all.push([v, k]); });
  const N = all.length, k = groups.filter(g => g.some(Number.isFinite)).length;
  if (k < 2 || N <= k) return null;
  all.sort((p, q) => p[0] - q[0]);
  const R = new Float64Array(groups.length), cnt = new Float64Array(groups.length); let ties = 0;
  for (let i = 0; i < N;) {
    let j = i; while (j + 1 < N && all[j + 1][0] === all[i][0]) j++;
    const r = (i + j) / 2 + 1, t = j - i + 1; ties += t * t * t - t;
    for (let q = i; q <= j; q++) { R[all[q][1]] += r; cnt[all[q][1]]++; }
    i = j + 1;
  }
  let H = 0; for (let g = 0; g < groups.length; g++) if (cnt[g]) H += R[g] * R[g] / cnt[g];
  H = 12 / (N * (N + 1)) * H - 3 * (N + 1);
  const C = 1 - ties / (N * N * N - N); if (C > 0) H /= C;
  return { H, df: k - 1, p: gammaQ((k - 1) / 2, H / 2), n: N };
}

// Brown-Forsythe test of equal spread (Levene's, about each group's median), with its F p-value
export function brownForsythe(groups) {
  const z = groups.map(g => { const v = Array.from(g).filter(Number.isFinite), m = median(v); return v.map(x => Math.abs(x - m)); }).filter(g => g.length);
  const k = z.length, N = z.reduce((t, g) => t + g.length, 0); if (k < 2 || N <= k) return null;
  const means = z.map(g => g.reduce((t, v) => t + v, 0) / g.length), all = z.reduce((t, g, i) => t + means[i] * g.length, 0) / N;
  let between = 0, within = 0;
  z.forEach((g, i) => { between += g.length * (means[i] - all) ** 2; for (const v of g) within += (v - means[i]) ** 2; });
  const d1 = k - 1, d2 = N - k, F = (between / d1) / (within / d2);
  return { F, df1: d1, df2: d2, p: within > 0 ? betaInc(d2 / (d2 + d1 * F), d2 / 2, d1 / 2) : NaN, n: N };
}

// Distances between maps against their sample labels: the mean distance between maps of different samples
// minus the mean between maps of the same sample, and the share of relabellings (the labels shuffled over
// the maps) that give a gap at least as large. With few maps there are few distinct relabellings, and p can
// fall no lower than one over their number (also returned).
export function groupGap(d, labels, { nPerm = 4999, seed = 7 } = {}) {
  const m = labels.length;
  const gap = lab => {
    let s = 0, ns = 0, t = 0, nt = 0;
    for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) { if (lab[i] === lab[j]) { s += d[i][j]; ns++; } else { t += d[i][j]; nt++; } }
    return ns && nt ? { gap: t / nt - s / ns, within: s / ns, between: t / nt } : null;
  };
  const obs = gap(labels); if (!obs) return null;
  const R = rng(seed), lab = [...labels]; let ge = 0;
  for (let k = 0; k < nPerm; k++) {
    for (let i = m - 1; i > 0; i--) { const j = Math.floor(R() * (i + 1)); [lab[i], lab[j]] = [lab[j], lab[i]]; }
    const g = gap(lab); if (g && g.gap >= obs.gap - 1e-12) ge++;
  }
  // distinct groupings of the maps into groups of these sizes: m! / (prod size! x prod over sizes of count!)
  const sizes = Object.values(labels.reduce((t, l) => ({ ...t, [l]: (t[l] || 0) + 1 }), {}));
  const lf = n => { let s = 0; for (let i = 2; i <= n; i++) s += Math.log(i); return s; };
  const bySize = sizes.reduce((t, s) => ({ ...t, [s]: (t[s] || 0) + 1 }), {});
  const groupings = Math.round(Math.exp(lf(m) - sizes.reduce((t, s) => t + lf(s), 0) - Object.values(bySize).reduce((t, c) => t + lf(c), 0)));
  return { ...obs, p: (ge + 1) / (nPerm + 1), groupings, pMin: 1 / groupings };
}
