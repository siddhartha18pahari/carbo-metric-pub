// Semi-quantitative statistics for ParticleFinder data: confidence intervals
// for shares, populations found by a Gaussian mixture, volume-weighted size
// quantiles and the crystallite size La. Pure functions (no DOM), so they run
// under Node for the tests.

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
