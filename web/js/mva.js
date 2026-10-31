// MVA: multivariate analysis of a map's spectra, as HORIBA's MVA Plus does it: loadings by PCA,
// MCR, K-means or HCA (or the user's reference spectra), images by classical least squares (CLS)
// against them, the residual over the noise, a comparison between methods, and each loading named
// after the reference it matches best. The vendor publishes no method; these are the textbook ones.
//
// The spectra are prepared as on the Layers tab (layers.js: cut to the range, cosmic rays removed
// against the neighbouring spectra, binned, polynomial background off), then scaled to unit area or
// unit length if asked, their noise with them.
//  * PCA: eigenvectors of the covariance of the mean-centred spectra (from up to 3 000 of them,
//    evenly spread), each loading signed so that its largest value is positive. The number
//    suggested is the smallest that leaves no more than a handful of spectra (2, or 0.2 %) with
//    over twice their noise unexplained.
//  * MCR: non-negative unmixing (layers.js decompose: successive projections, refined by NMF).
//  * K-means: on the first PCA scores, k-means++ starts, the best of three. HCA: Ward's linkage by
//    the nearest-neighbour chain on up to 1 000 spectra (evenly spread), the others going to the
//    nearest class. Unless set, k is the one of 2..8 with the highest mean silhouette. The loadings
//    are the classes' mean spectra.
//  * CLS: every spectrum fitted by non-negative least squares to the loadings (PCA's images are its
//    scores), with what is left over its noise.

import { nmfProject, rng } from "./pf-analysis.js";
import { decompose, prepRef } from "./layers.js";

export const MVA_DEFAULTS = {
  method: "pca",          // "pca", "mcr", "kmeans", "hca" or "refs" (CLS against the reference spectra)
  k: 0,                   // components or classes: 0 chooses, else that many
  kMax: 12,
  range: [150, 3600],     // cm⁻¹, cut to what was measured
  baseline: 3,            // degree of the polynomial background taken off each spectrum (0: none)
  norm: "vector",         // "none", "area" (unit sum) or "vector" (unit length), at the median scale
  stop: 2,                // a spectrum counts as explained with no more than this × its noise left
  minHQI: 80,             // a loading takes a reference's name at this match or above
};
export const METHODS = { pca: "PCA", mcr: "MCR", kmeans: "K-means", hca: "HCA", refs: "References (CLS)" };
const SAMPLE = 3000, HCA_MAX = 1000, SIL_MAX = 1000;

const need = n => Math.max(2, Math.ceil(0.002 * n));

// --------------------------------------------------------------------------
// the spectra, scaled
// --------------------------------------------------------------------------
// Each spectrum scaled to unit area (sum) or unit length, times the median of those, so the numbers
// stay in counts. A spectrum whose area or length is under 4 × that of its noise (σ √bins) is
// scaled as if it had that much, so empty points stay at a quarter to a third of full size instead of
// being blown up to it.
export function normalise(prep, norm) {
  const { n, nb, V, noise } = prep, X = new Float32Array(n * nb), size = new Float64Array(n), out = new Float64Array(n);
  if (norm !== "area" && norm !== "vector") { X.set(V); return { X, noise: Float64Array.from(noise), scale: new Float64Array(n).fill(1) }; }
  for (let i = 0; i < n; i++) {
    let t = 0;
    for (let j = 0; j < nb; j++) { const v = V[i * nb + j]; t += norm === "area" ? v : v * v; }
    size[i] = Math.max(norm === "area" ? t : Math.sqrt(t), 4 * noise[i] * Math.sqrt(nb));
  }
  const med = Float64Array.from(size).sort()[n >> 1] || 1, scale = Float64Array.from(size, s => med / s);
  for (let i = 0; i < n; i++) { for (let j = 0; j < nb; j++) X[i * nb + j] = V[i * nb + j] * scale[i]; out[i] = noise[i] * scale[i]; }
  return { X, noise: out, scale };
}

// --------------------------------------------------------------------------
// PCA
// --------------------------------------------------------------------------
// Eigenvalues w and eigenvectors (columns of V, V[k][j] = vector j's k-th entry) of a symmetric
// n x n matrix: Householder reduction to tridiagonal form, then the implicit QL algorithm (tred2
// and tql2 of EISPACK, as in JAMA). About n³ operations, where cyclic Jacobi takes ten times more.
export function symEig(A, n) {
  const V = Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, j) => A[i * n + j])), d = new Float64Array(n), e = new Float64Array(n);
  for (let j = 0; j < n; j++) d[j] = V[n - 1][j];
  for (let i = n - 1; i > 0; i--) {
    let scale = 0, h = 0;
    for (let k = 0; k < i; k++) scale += Math.abs(d[k]);
    if (scale === 0) {
      e[i] = d[i - 1];
      for (let j = 0; j < i; j++) { d[j] = V[i - 1][j]; V[i][j] = 0; V[j][i] = 0; }
    } else {
      for (let k = 0; k < i; k++) { d[k] /= scale; h += d[k] * d[k]; }
      let f = d[i - 1], g = Math.sqrt(h);
      if (f > 0) g = -g;
      e[i] = scale * g; h -= f * g; d[i - 1] = f - g;
      for (let j = 0; j < i; j++) e[j] = 0;
      for (let j = 0; j < i; j++) {
        f = d[j]; V[j][i] = f; g = e[j] + V[j][j] * f;
        for (let k = j + 1; k <= i - 1; k++) { g += V[k][j] * d[k]; e[k] += V[k][j] * f; }
        e[j] = g;
      }
      f = 0;
      for (let j = 0; j < i; j++) { e[j] /= h; f += e[j] * d[j]; }
      const hh = f / (h + h);
      for (let j = 0; j < i; j++) e[j] -= hh * d[j];
      for (let j = 0; j < i; j++) {
        f = d[j]; g = e[j];
        for (let k = j; k <= i - 1; k++) V[k][j] -= f * e[k] + g * d[k];
        d[j] = V[i - 1][j]; V[i][j] = 0;
      }
    }
    d[i] = h;
  }
  for (let i = 0; i < n - 1; i++) {
    V[n - 1][i] = V[i][i]; V[i][i] = 1;
    const h = d[i + 1];
    if (h !== 0) {
      for (let k = 0; k <= i; k++) d[k] = V[k][i + 1] / h;
      for (let j = 0; j <= i; j++) {
        let g = 0;
        for (let k = 0; k <= i; k++) g += V[k][i + 1] * V[k][j];
        for (let k = 0; k <= i; k++) V[k][j] -= g * d[k];
      }
    }
    for (let k = 0; k <= i; k++) V[k][i + 1] = 0;
  }
  for (let j = 0; j < n; j++) { d[j] = V[n - 1][j]; V[n - 1][j] = 0; }
  V[n - 1][n - 1] = 1; e[0] = 0;
  for (let i = 1; i < n; i++) e[i - 1] = e[i];
  e[n - 1] = 0;
  let f = 0, tst1 = 0;
  const eps = 2 ** -52;
  for (let l = 0; l < n; l++) {
    tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
    let m = l;
    while (m < n && Math.abs(e[m]) > eps * tst1) m++;
    if (m > l) {
      do {
        let g = d[l], p = (d[l + 1] - g) / (2 * e[l]), r = Math.hypot(p, 1);
        if (p < 0) r = -r;
        d[l] = e[l] / (p + r); d[l + 1] = e[l] * (p + r);
        const dl1 = d[l + 1];
        let h = g - d[l];
        for (let i = l + 2; i < n; i++) d[i] -= h;
        f += h;
        p = d[m];
        let c = 1, c2 = 1, c3 = 1, s = 0, s2 = 0;
        const el1 = e[l + 1];
        for (let i = m - 1; i >= l; i--) {
          c3 = c2; c2 = c; s2 = s;
          g = c * e[i]; h = c * p; r = Math.hypot(p, e[i]);
          e[i + 1] = s * r; s = e[i] / r; c = p / r; p = c * d[i] - s * g;
          d[i + 1] = h + s * (c * g + s * d[i]);
          for (let k = 0; k < n; k++) { h = V[k][i + 1]; V[k][i + 1] = s * V[k][i] + c * h; V[k][i] = c * V[k][i] - s * h; }
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p; d[l] = c * p;
      } while (Math.abs(e[l]) > eps * tst1);
    }
    d[l] += f; e[l] = 0;
  }
  return { w: d, V };
}

export function pca(X, n, nb, noise, { q = 20, stop = 2 } = {}) {
  const mean = new Float64Array(nb);
  for (let i = 0; i < n; i++) for (let j = 0; j < nb; j++) mean[j] += X[i * nb + j] / n;
  const step = Math.max(1, Math.floor(n / SAMPLE)), d = new Float64Array(nb), C = new Float64Array(nb * nb);
  let m = 0;
  for (let i = 0; i < n; i += step) {
    m++;
    for (let j = 0; j < nb; j++) d[j] = X[i * nb + j] - mean[j];
    for (let a = 0; a < nb; a++) { const da = d[a]; if (!da) continue; for (let b = a; b < nb; b++) C[a * nb + b] += da * d[b]; }
  }
  for (let a = 0; a < nb; a++) for (let b = a; b < nb; b++) { C[a * nb + b] /= Math.max(1, m - 1); C[b * nb + a] = C[a * nb + b]; }
  const { w, V } = symEig(C, nb), order = Array.from(w.keys()).sort((a, b) => w[b] - w[a]);
  let total = 0; for (const v of w) total += Math.max(0, v);
  q = Math.min(q, nb, Math.max(1, n - 1));
  const L = new Float64Array(q * nb), eig = new Float64Array(q);
  for (let a = 0; a < q; a++) {
    const c = order[a];
    eig[a] = Math.max(0, w[c]);
    let big = 0; for (let j = 0; j < nb; j++) if (Math.abs(V[j][c]) > Math.abs(big)) big = V[j][c];
    const sgn = big < 0 ? -1 : 1;
    for (let j = 0; j < nb; j++) L[a * nb + j] = sgn * V[j][c];
  }
  // scores, and what each spectrum has left with 0, 1, ... q components
  const S = new Float64Array(n * q), left = new Float64Array(n), over = new Int32Array(q + 1);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let j = 0; j < nb; j++) { d[j] = X[i * nb + j] - mean[j]; r += d[j] * d[j]; }
    const lim = stop * stop * nb * noise[i] * noise[i];
    if (r > lim) over[0]++;
    for (let a = 0; a < q; a++) {
      let s = 0; for (let j = 0; j < nb; j++) s += d[j] * L[a * nb + j];
      S[i * q + a] = s; r -= s * s;
      if (r > lim) over[a + 1]++;
    }
    left[i] = Math.sqrt(Math.max(0, r) / nb) / noise[i];
  }
  let suggested = q;
  for (let a = 0; a <= q; a++) if (over[a] < need(n)) { suggested = Math.max(1, a); break; }
  return { mean, L, q, eig, total, S, over: Array.from(over), suggested, left };
}

// --------------------------------------------------------------------------
// clustering
// --------------------------------------------------------------------------
const dist2 = (A, i, B, j, d) => { let s = 0; for (let c = 0; c < d; c++) { const t = A[i * d + c] - B[j * d + c]; s += t * t; } return s; };

// k-means on the rows of Z (n x d): k-means++ starts, Lloyd's iterations, the best of `inits`
export function kmeans(Z, n, d, k, { seed = 1, inits = 5, iters = 100 } = {}) {
  const R = rng(seed);
  let best = null;
  for (let t = 0; t < inits; t++) {
    const C = new Float64Array(k * d), D = new Float64Array(n).fill(Infinity);
    let first = Math.floor(R() * n);
    C.set(Z.subarray(first * d, first * d + d), 0);
    for (let c = 1; c < k; c++) {
      let tot = 0;
      for (let i = 0; i < n; i++) { D[i] = Math.min(D[i], dist2(Z, i, C, c - 1, d)); tot += D[i]; }
      let r = R() * tot, pick = n - 1;
      for (let i = 0; i < n; i++) { r -= D[i]; if (r <= 0) { pick = i; break; } }
      C.set(Z.subarray(pick * d, pick * d + d), c * d);
    }
    const lab = new Int32Array(n).fill(-1), cnt = new Float64Array(k);
    let inertia = 0;
    for (let it = 0; it < iters; it++) {
      let moved = 0; inertia = 0;
      for (let i = 0; i < n; i++) {
        let bc = 0, bd = Infinity;
        for (let c = 0; c < k; c++) { const v = dist2(Z, i, C, c, d); if (v < bd) { bd = v; bc = c; } }
        if (lab[i] !== bc) { lab[i] = bc; moved++; }
        inertia += bd;
      }
      if (!moved && it) break;
      C.fill(0); cnt.fill(0);
      for (let i = 0; i < n; i++) { cnt[lab[i]]++; for (let c = 0; c < d; c++) C[lab[i] * d + c] += Z[i * d + c]; }
      for (let c = 0; c < k; c++) {
        if (cnt[c]) { for (let e = 0; e < d; e++) C[c * d + e] /= cnt[c]; continue; }
        // an empty class restarts at the point farthest from its own centre
        let far = 0, fd = -1;
        for (let i = 0; i < n; i++) { const v = dist2(Z, i, C, lab[i], d); if (v > fd) { fd = v; far = i; } }
        C.set(Z.subarray(far * d, far * d + d), c * d);
      }
    }
    if (!best || inertia < best.inertia) best = { labels: Int32Array.from(lab), centres: Float64Array.from(C), inertia };
  }
  return best;
}

// the mean silhouette of the labelled rows `idx` of Z (each point's mean distance to its own class
// against the nearest other class)
export function silhouette(Z, d, labels, k, idx) {
  const m = idx.length;
  if (k < 2 || m < 3) return NaN;
  let tot = 0, cnt = 0;
  const sum = new Float64Array(k), num = new Float64Array(k);
  for (const i of idx) num[labels[i]]++;
  for (const i of idx) {
    sum.fill(0);
    for (const j of idx) if (j !== i) sum[labels[j]] += Math.sqrt(dist2(Z, i, Z, j, d));
    const own = labels[i];
    if (num[own] < 2) continue;
    const a = sum[own] / (num[own] - 1);
    let b = Infinity; for (let c = 0; c < k; c++) if (c !== own && num[c]) b = Math.min(b, sum[c] / num[c]);
    if (!Number.isFinite(b)) continue;
    tot += (b - a) / Math.max(a, b); cnt++;
  }
  return cnt ? tot / cnt : NaN;
}

// Ward's linkage of the rows of P (m x d) by the nearest-neighbour chain, on squared distances
// with the Lance-Williams update. Returns the m - 1 merges sorted by height, as [a, b, height,
// size] with leaves 0..m-1 and merge i making node m + i (heights in distance units, as scipy's).
export function ward(P, m, d) {
  const D = new Float64Array(m * (m - 1) / 2), at = (i, j) => (i < j ? i * (2 * m - i - 1) / 2 + j - i - 1 : j * (2 * m - j - 1) / 2 + i - j - 1);
  for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) D[at(i, j)] = dist2(P, i, P, j, d);
  const size = new Float64Array(m).fill(1), alive = new Uint8Array(m).fill(1), raw = [], chain = [];
  let left = m;
  while (left > 1) {
    if (!chain.length) { let a = 0; while (!alive[a]) a++; chain.push(a); }
    const a = chain[chain.length - 1], prev = chain.length > 1 ? chain[chain.length - 2] : -1;
    let b = -1, bd = Infinity;
    if (prev >= 0) { b = prev; bd = D[at(a, prev)]; }
    for (let c = 0; c < m; c++) if (alive[c] && c !== a && D[at(a, c)] < bd) { bd = D[at(a, c)]; b = c; }
    if (b !== prev) { chain.push(b); continue; }
    chain.pop(); chain.pop();
    const na = size[a], nbb = size[b];
    for (let c = 0; c < m; c++) {
      if (!alive[c] || c === a || c === b) continue;
      const nc = size[c];
      D[at(b, c)] = ((na + nc) * D[at(a, c)] + (nbb + nc) * D[at(b, c)] - nc * bd) / (na + nbb + nc);
    }
    raw.push([a, b, Math.sqrt(Math.max(0, bd)), na + nbb]);
    alive[a] = 0; size[b] = na + nbb; left--;           // the merged class lives on in b's place
  }
  // in order of height, with node numbers
  raw.sort((p, q) => p[2] - q[2]);
  const node = Int32Array.from({ length: m }, (_, i) => i), parent = Int32Array.from({ length: m }, (_, i) => i);
  const find = i => { while (parent[i] !== i) i = parent[i] = parent[parent[i]]; return i; };
  return raw.map(([a, b, h, s], i) => {
    const ra = find(a), rb = find(b), out = [node[ra], node[rb], h, s];
    parent[ra] = rb; node[rb] = m + i;
    return out;
  });
}

// the k classes left when the tree is cut below its k - 1 highest merges, numbered by first leaf
export function cutTree(merges, m, k) {
  const parent = Int32Array.from({ length: 2 * m - 1 }, (_, i) => i);
  const find = i => { while (parent[i] !== i) i = parent[i] = parent[parent[i]]; return i; };
  for (let i = 0; i < m - k; i++) { const [a, b] = merges[i]; parent[find(a)] = m + i; parent[find(b)] = m + i; }
  const ids = new Map(), lab = new Int32Array(m);
  for (let i = 0; i < m; i++) { const r = find(i); if (!ids.has(r)) ids.set(r, ids.size); lab[i] = ids.get(r); }
  return lab;
}

// the top of the tree for drawing: its last `top` merges, the subtrees below them as leaves
export function treeTop(merges, m, top = 24) {
  const T = Math.min(top, merges.length), first = merges.length - T, size = n => (n < m ? 1 : merges[n - m][3]);
  return { m, first: m + first, nodes: merges.slice(first).map(([a, b, h, s], i) => ({ id: m + first + i, a, b, h, s, sa: size(a), sb: size(b) })) };
}

// --------------------------------------------------------------------------
// CLS and comparison
// --------------------------------------------------------------------------
// every spectrum fitted (non-negative) to the k loadings L (k x nb): amounts W (n x k), each
// component's share of the fitted spectrum, and the residual RMS over the noise
export function cls(X, n, nb, L, k, noise) {
  const W = nmfProject(X, n, nb, L, k, { iters: 150 }), area = new Float64Array(k), share = new Float64Array(n * k);
  for (let a = 0; a < k; a++) for (let j = 0; j < nb; j++) area[a] += L[a * nb + j];
  for (let i = 0; i < n; i++) {
    let tot = 0; for (let a = 0; a < k; a++) tot += Math.max(0, W[i * k + a] * area[a]);
    for (let a = 0; a < k; a++) share[i * k + a] = tot > 0 ? Math.max(0, W[i * k + a] * area[a]) / tot : NaN;
  }
  return { W, share, ...residuals(X, n, nb, noise, (i, f) => { f.fill(0); for (let a = 0; a < k; a++) { const w = W[i * k + a]; if (w) for (let j = 0; j < nb; j++) f[j] += w * L[a * nb + j]; } }) };
}

// what the fit leaves: each spectrum's residual RMS over its noise, and per channel the mean
// residual and its RMS over all spectra (a band the components miss shows in both)
function residuals(X, n, nb, noise, fitRow) {
  const resid = new Float64Array(n), mean = new Float64Array(nb), rms = new Float64Array(nb), f = new Float64Array(nb);
  for (let i = 0; i < n; i++) {
    fitRow(i, f);
    let r2 = 0;
    for (let j = 0; j < nb; j++) { const r = X[i * nb + j] - f[j]; r2 += r * r; mean[j] += r / n; rms[j] += r * r / n; }
    resid[i] = Math.sqrt(r2 / nb) / noise[i];
  }
  for (let j = 0; j < nb; j++) rms[j] = Math.sqrt(rms[j]);
  return { resid, residSpec: { mean, rms } };
}

// two sets of loadings matched component by component, the most alike pair first (|r|, as a PCA
// loading's sign is arbitrary); the mean |r| of the pairs
export function compareLoadings(A, ka, B, kb, nb) {
  const r = (P, i, Q, j) => {
    let mx = 0, my = 0; for (let c = 0; c < nb; c++) { mx += P[i * nb + c]; my += Q[j * nb + c]; }
    mx /= nb; my /= nb;
    let sxy = 0, sxx = 0, syy = 0;
    for (let c = 0; c < nb; c++) { const x = P[i * nb + c] - mx, y = Q[j * nb + c] - my; sxy += x * y; sxx += x * x; syy += y * y; }
    return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : 0;
  };
  const all = [];
  for (let i = 0; i < ka; i++) for (let j = 0; j < kb; j++) all.push({ a: i, b: j, r: r(A, i, B, j) });
  all.sort((p, q) => Math.abs(q.r) - Math.abs(p.r));
  const ua = new Set(), ub = new Set(), pairs = [];
  for (const p of all) if (!ua.has(p.a) && !ub.has(p.b)) { pairs.push(p); ua.add(p.a); ub.add(p.b); }
  return { pairs, mean: pairs.length ? pairs.reduce((t, p) => t + Math.abs(p.r), 0) / pairs.length : NaN };
}

// --------------------------------------------------------------------------
// one run
// --------------------------------------------------------------------------
// The loadings of the chosen method and the images from them. `prep` is layers.js's prepare().
export function runMva(prep, settings, refs = [], onStage = null) {
  const { n, nb } = prep, s = settings, stage = t => onStage && onStage(t);
  if (n < 3) throw new Error("multivariate analysis needs at least three measured spectra");
  const { X, noise, scale } = normalise(prep, s.norm), out = { method: s.method, nb, n, scale };
  const want = s.k ? Math.min(s.k, s.kMax, n) : 0;
  if (s.method === "pca") {
    stage("Principal components");
    const P = pca(X, n, nb, noise, { q: Math.min(20, Math.max(want, 12)), stop: s.stop });
    const k = Math.min(P.q, want || P.suggested);
    Object.assign(out, { k, kind: "signed", L: P.L.slice(0, k * nb), mean: P.mean, q: P.q,
      W: Float64Array.from({ length: n * k }, (_, t) => P.S[Math.floor(t / k) * P.q + (t % k)]), share: null,
      ...residuals(X, n, nb, noise, (i, f) => { f.set(P.mean); for (let a = 0; a < k; a++) { const s0 = P.S[i * P.q + a]; for (let j = 0; j < nb; j++) f[j] += s0 * P.L[a * nb + j]; } }),
      pca: { eig: P.eig, total: P.total, over: P.over, suggested: P.suggested },
      scatter: { x: Float64Array.from({ length: n }, (_, i) => P.S[i * P.q]), y: Float64Array.from({ length: n }, (_, i) => P.S[i * P.q + 1] || 0), xl: "PC 1 score", yl: "PC 2 score" } });
    return out;
  }
  let L, k;
  if (s.method === "mcr" || s.method === "refs") {
    stage(s.method === "mcr" ? "Finding the components" : "Unmixing against the references");
    const c = decompose({ ...prep, V: X, noise }, { ...s, k: want, useRefs: s.method === "refs" }, refs, stage);
    L = c.H; k = c.k;
    out.steps = c.steps;
  } else {
    // clustering on the first principal components
    stage("Principal components");
    const P = pca(X, n, nb, noise, { q: 10, stop: s.stop }), d = Math.min(P.q, Math.max(2, Math.min(10, P.suggested))), Z = new Float64Array(n * d);
    for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) Z[i * d + a] = P.S[i * P.q + a];
    const stepS = Math.max(1, Math.floor(n / SIL_MAX)), silIdx = []; for (let i = 0; i < n; i += stepS) silIdx.push(i);
    const ks = want ? [want] : [2, 3, 4, 5, 6, 7, 8].filter(v => v < n), sil = [];
    let labels = null;
    if (s.method === "kmeans") {
      stage("K-means");
      let best = null;
      for (const kk of ks) {
        const r = kmeans(Z, n, d, kk, { seed: 7, inits: 3 }), si = silhouette(Z, d, r.labels, kk, silIdx);
        sil.push({ k: kk, s: si });
        if (!best || (!want && si > best.s)) best = { k: kk, s: si, labels: r.labels };
      }
      k = best.k; labels = best.labels;
    } else {
      stage("Hierarchical clustering");
      const stepH = Math.max(1, Math.ceil(n / HCA_MAX)), hIdx = []; for (let i = 0; i < n; i += stepH) hIdx.push(i);
      const m = hIdx.length, Pm = new Float64Array(m * d);
      hIdx.forEach((i, r) => Pm.set(Z.subarray(i * d, i * d + d), r * d));
      const merges = ward(Pm, m, d), loc = Int32Array.from({ length: m }, (_, i) => i);
      let best = null;
      for (const kk of ks) {
        const lab = cutTree(merges, m, kk), si = silhouette(Pm, d, lab, kk, Array.from(loc));
        sil.push({ k: kk, s: si });
        if (!best || (!want && si > best.s)) best = { k: kk, s: si, lab };
      }
      k = best.k;
      // the rest go to the nearest class centre
      const C = new Float64Array(k * d), cnt = new Float64Array(k);
      for (let r = 0; r < m; r++) { cnt[best.lab[r]]++; for (let a = 0; a < d; a++) C[best.lab[r] * d + a] += Pm[r * d + a]; }
      for (let c = 0; c < k; c++) for (let a = 0; a < d; a++) C[c * d + a] /= cnt[c] || 1;
      labels = new Int32Array(n);
      for (let i = 0; i < n; i++) { let bc = 0, bd = Infinity; for (let c = 0; c < k; c++) { const v = dist2(Z, i, C, c, d); if (v < bd) { bd = v; bc = c; } } labels[i] = bc; }
      hIdx.forEach((i, r) => { labels[i] = best.lab[r]; });
      out.tree = { ...treeTop(merges, m, 24), cut: k > 1 ? (merges[m - k][2] + merges[m - k - 1][2]) / 2 : NaN, sampled: m < n ? m : 0 };
    }
    // the loadings: each class's mean spectrum, classes numbered by size
    const size = new Float64Array(k); for (const v of labels) size[v]++;
    const order = Array.from(size.keys()).sort((a, b) => size[b] - size[a]), rank = new Int32Array(k); order.forEach((c, r) => { rank[c] = r; });
    for (let i = 0; i < n; i++) labels[i] = rank[labels[i]];
    L = new Float64Array(k * nb);
    for (let i = 0; i < n; i++) for (let j = 0; j < nb; j++) L[labels[i] * nb + j] += X[i * nb + j] / size[order[labels[i]]];
    Object.assign(out, { labels, sizes: order.map(c => size[c]), sil, dims: d,
      scatter: { x: Float64Array.from({ length: n }, (_, i) => Z[i * d]), y: Float64Array.from({ length: n }, (_, i) => Z[i * d + 1] || 0), xl: "PC 1 score", yl: "PC 2 score" } });
  }
  stage("CLS images");
  const C = cls(X, n, nb, L, k, noise);
  Object.assign(out, { k, kind: "nonneg", L: Float64Array.from(L), W: C.W, share: C.share, resid: C.resid, residSpec: C.residSpec });
  if (!out.scatter) out.scatter = { x: Float64Array.from({ length: n }, (_, i) => C.share[i * k]), y: Float64Array.from({ length: n }, (_, i) => (k > 1 ? C.share[i * k + 1] : 0)), xl: "Component 1 share", yl: "Component 2 share" };
  return out;
}

// each non-negative loading's best reference (HQI = 100 r² over the bins it covers)
export function identify(R, prep, refs, minHQI) {
  if (R.kind !== "nonneg" || !refs.length) return null;
  const P = refs.map(r => prepRef(r, prep)), nb = R.nb;
  return Array.from({ length: R.k }, (_, a) => {
    const hqi = P.map(r => {
      if (r.cover < 0.5) return NaN;
      let n = 0, mx = 0, my = 0;
      for (let j = 0; j < nb; j++) if (Number.isFinite(r.y[j])) { n++; mx += R.L[a * nb + j]; my += r.y[j]; }
      mx /= n; my /= n;
      let sxy = 0, sxx = 0, syy = 0;
      for (let j = 0; j < nb; j++) if (Number.isFinite(r.y[j])) { const x = R.L[a * nb + j] - mx, y = r.y[j] - my; sxy += x * y; sxx += x * x; syy += y * y; }
      const c = sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : 0;
      return c > 0 ? 100 * c * c : 0;
    });
    let best = -1; hqi.forEach((h, r) => { if (h >= minHQI && (best < 0 || h > hqi[best])) best = r; });
    return { hqi, best };
  });
}
