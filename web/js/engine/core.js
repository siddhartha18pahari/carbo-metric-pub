// The fitting engine for the browser: a port of qcarbon/fitter.py.
//
// Same model, bounds, starting point, Levenberg-Marquardt steps, mean-spectrum
// shapes and conventions (area over the fit window, Std. error with N - 3 x
// bands degrees of freedom), so a map fitted here and by the Python server
// gives the same numbers to fit tolerance. Pure functions on typed arrays: the
// workers run lmBatch, the page runs the rest.

const G4 = 4 * Math.log(2);
export const LORENTZ = 0, GAUSS = 1;

// --------------------------------------------------------------------------
// line shapes and areas
// --------------------------------------------------------------------------
export function shapeEval(nu, c, w, kind) {
  if (kind === "Lorentz") { const u = 2 * (nu - c) / w; return 1 / (1 + u * u); }
  const v = (nu - c) / w;
  return Math.exp(-G4 * v * v);
}

// erf to double precision: Taylor series near 0, continued fraction for erfc beyond
export function erf(x) {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? NaN : Math.sign(x);
  const ax = Math.abs(x);
  if (ax < 2.5) {
    let term = ax, sum = ax;
    for (let n = 1; n < 200; n++) {
      term *= -ax * ax / n;
      const t = term / (2 * n + 1);
      sum += t;
      if (Math.abs(t) < 1e-17 * Math.abs(sum)) break;
    }
    return Math.sign(x) * sum * 2 / Math.sqrt(Math.PI);
  }
  if (ax > 27) return Math.sign(x);
  // erfc(x) = exp(-x^2)/sqrt(pi) * 1 / (x + 1/2 / (x + 1 / (x + 3/2 / (x + ...)))), modified Lentz
  const tiny = 1e-300;
  let f = ax, C = ax, D = 0;
  for (let n = 1; n < 500; n++) {
    const a = n / 2;
    D = ax + a * D; D = Math.abs(D) < tiny ? tiny : D; D = 1 / D;
    C = ax + a / C; C = Math.abs(C) < tiny ? tiny : C;
    const delta = C * D;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  const erfc = Math.exp(-ax * ax) / Math.sqrt(Math.PI) / f;
  return Math.sign(x) * (1 - erfc);
}

export function windowedArea(A, c, w, kind, lo, hi) {
  if (kind === "Lorentz") return A * (w / 2) * (Math.atan(2 * (hi - c) / w) - Math.atan(2 * (lo - c) / w));
  const s = w / (2 * Math.sqrt(2 * Math.log(2)));
  return A * s * Math.sqrt(Math.PI / 2) * (erf((hi - c) / (s * Math.SQRT2)) - erf((lo - c) / (s * Math.SQRT2)));
}

// (N, order+1) row-major, t = (nu - mid) / half
export function baselineBasis(nu, lo, hi, order) {
  const N = nu.length, nb = order + 1, B = new Float64Array(N * Math.max(nb, 0));
  for (let n = 0; n < N; n++) {
    const t = (nu[n] - 0.5 * (lo + hi)) / (0.5 * (hi - lo));
    let v = 1;
    for (let k = 0; k < nb; k++) { B[n * nb + k] = v; v *= t; }
  }
  return B;
}

// --------------------------------------------------------------------------
// small dense linear algebra (row-major Float64Array)
// --------------------------------------------------------------------------
// eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations
function jacobiEig(Ain, n) {
  const A = Float64Array.from(Ain), V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0, diag = 0;
    for (let i = 0; i < n; i++) {
      diag += A[i * n + i] ** 2;
      for (let j = i + 1; j < n; j++) off += A[i * n + j] ** 2;
    }
    if (off <= 1e-30 * diag || off === 0) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      const apq = A[p * n + q];
      if (apq === 0) continue;
      const app = A[p * n + p], aqq = A[q * n + q];
      const theta = (aqq - app) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = A[k * n + p], akq = A[k * n + q];
        A[k * n + p] = c * akp - s * akq; A[k * n + q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = A[p * n + k], aqk = A[q * n + k];
        A[p * n + k] = c * apk - s * aqk; A[q * n + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k * n + p], vkq = V[k * n + q];
        V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq;
      }
    }
  }
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = A[i * n + i];
  return { w, V };
}

// Moore-Penrose inverse of a symmetric matrix (numpy's rcond = 1e-15 of the largest)
export function pinvSym(S, n, rcond = 1e-15) {
  const { w, V } = jacobiEig(S, n);
  let wmax = 0;
  for (const v of w) wmax = Math.max(wmax, Math.abs(v));
  const out = new Float64Array(n * n);
  for (let k = 0; k < n; k++) {
    if (Math.abs(w[k]) <= rcond * wmax || w[k] === 0) continue;
    const inv = 1 / w[k];
    for (let i = 0; i < n; i++) {
      const vik = V[i * n + k] * inv;
      for (let j = 0; j < n; j++) out[i * n + j] += vik * V[j * n + k];
    }
  }
  return out;
}

// pinv of an (N, q) matrix as (q, N), via (A^T A)^+ A^T (exact for any rank)
export function pinv(A, N, q) {
  if (q === 0) return new Float64Array(0);
  const AtA = new Float64Array(q * q);
  for (let n = 0; n < N; n++) for (let i = 0; i < q; i++) {
    const ai = A[n * q + i];
    if (ai === 0) continue;
    for (let j = 0; j < q; j++) AtA[i * q + j] += ai * A[n * q + j];
  }
  const inv = pinvSym(AtA, q);
  const out = new Float64Array(q * N);
  for (let i = 0; i < q; i++) for (let n = 0; n < N; n++) {
    let s = 0;
    for (let j = 0; j < q; j++) s += inv[i * q + j] * A[n * q + j];
    out[i * N + n] = s;
  }
  return out;
}

// solve H x = g for symmetric positive definite H (Cholesky), LU fallback
function solveSPD(H, g, n) {
  const L = new Float64Array(n * n);
  let ok = true;
  for (let i = 0; i < n && ok; i++) for (let j = 0; j <= i; j++) {
    let s = H[i * n + j];
    for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
    if (i === j) { if (!(s > 0)) { ok = false; break; } L[i * n + i] = Math.sqrt(s); }
    else L[i * n + j] = s / L[j * n + j];
  }
  if (ok) {
    const y = new Float64Array(n), x = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = g[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k]; y[i] = s / L[i * n + i]; }
    for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
    return x;
  }
  const A = Float64Array.from(H), b = Float64Array.from(g);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r * n + c]) > Math.abs(A[piv * n + c])) piv = r;
    if (piv !== c) {
      for (let k = 0; k < n; k++) { const t = A[c * n + k]; A[c * n + k] = A[piv * n + k]; A[piv * n + k] = t; }
      const t = b[c]; b[c] = b[piv]; b[piv] = t;
    }
    const d = A[c * n + c] || 1e-300;
    for (let r = c + 1; r < n; r++) {
      const f = A[r * n + c] / d;
      if (f === 0) continue;
      for (let k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k];
      b[r] -= f * b[c];
    }
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < n; k++) s -= A[i * n + k] * x[k]; x[i] = s / (A[i * n + i] || 1e-300); }
  return x;
}

// --------------------------------------------------------------------------
// model and one Levenberg-Marquardt fit (as _eval / _lm_one)
// --------------------------------------------------------------------------
function evalModel(nu, B, th, nbase, kinds, J, f) {
  const N = nu.length, M = th.length;
  f.fill(0);
  for (let k = 0; k < nbase; k++) {
    const t = th[k];
    for (let n = 0; n < N; n++) { f[n] += t * B[n * nbase + k]; if (J) J[n * M + k] = B[n * nbase + k]; }
  }
  for (let j = 0; j < kinds.length; j++) {
    const i = nbase + 3 * j, A = th[i], c = th[i + 1], w = th[i + 2];
    for (let n = 0; n < N; n++) {
      let phi;
      if (kinds[j] === LORENTZ) {
        const u = 2 * (nu[n] - c) / w, d = 1 + u * u;
        phi = 1 / d;
        if (J) {
          const dphi = -2 * u / (d * d);
          J[n * M + i] = phi; J[n * M + i + 1] = A * dphi * (-2 / w); J[n * M + i + 2] = A * dphi * (-u / w);
        }
      } else {
        const v = (nu[n] - c) / w;
        phi = Math.exp(-2.772588722239781 * v * v);
        if (J) {
          const dphi = -2 * 2.772588722239781 * v * phi;
          J[n * M + i] = phi; J[n * M + i + 1] = A * dphi * (-1 / w); J[n * M + i + 2] = A * dphi * (-v / w);
        }
      }
      f[n] += A * phi;
    }
  }
  return f;
}

function sse(y, f) { let s = 0; for (let n = 0; n < y.length; n++) { const d = y[n] - f[n]; s += d * d; } return s; }

export function lmOne(nu, y, B, th, lo, hi, free, nbase, kinds, maxIter, tol) {
  const N = nu.length, M = th.length;
  // only free parameters enter the normal equations (held ones never move)
  const fi = [];
  for (let m = 0; m < M; m++) if (free[m]) fi.push(m);
  const nf = fi.length;
  const J = new Float64Array(N * M), f = new Float64Array(N), fn = new Float64Array(N);
  const g = new Float64Array(nf), H = new Float64Array(nf * nf), tn = new Float64Array(M);
  const Ha = new Float64Array(nf * nf), Hd = new Float64Array(nf * nf), ga = new Float64Array(nf);
  const idx = new Int32Array(nf), pos = new Int32Array(nf);
  let cost = sse(y, evalModel(nu, B, th, nbase, kinds, null, f));
  let lam = 1e-3, rel = 1, it = 0;
  while (it < maxIter) {
    it++;
    evalModel(nu, B, th, nbase, kinds, J, f);
    g.fill(0); H.fill(0);
    for (let n = 0; n < N; n++) {
      const r = y[n] - f[n], row = n * M;
      for (let a = 0; a < nf; a++) {
        const ja = J[row + fi[a]];
        if (ja === 0) continue;
        g[a] += ja * r;
        for (let b = a; b < nf; b++) H[a * nf + b] += ja * J[row + fi[b]];
      }
    }
    // hold parameters that sit on a bound with the gradient pointing outward
    let na = 0;
    for (let a = 0; a < nf; a++) {
      const m = fi[a];
      if (th[m] <= lo[m] && g[a] <= 0) continue;
      if (th[m] >= hi[m] && g[a] >= 0) continue;
      idx[na] = m; pos[na++] = a;
    }
    if (na === 0) break;
    let dmax = 0;
    for (let a = 0; a < na; a++) {
      ga[a] = g[pos[a]];
      for (let b = 0; b < na; b++) {
        const i = pos[a], j = pos[b];
        Ha[a * na + b] = i <= j ? H[i * nf + j] : H[j * nf + i];
      }
      dmax = Math.max(dmax, Ha[a * na + a]);
    }
    let accepted = false;
    for (let t = 0; t < 12; t++) {
      Hd.set(Ha.subarray(0, na * na));
      for (let a = 0; a < na; a++) Hd[a * na + a] += lam * Ha[a * na + a] + 1e-10 * dmax + 1e-12;
      const step = solveSPD(Hd.subarray(0, na * na), ga.subarray(0, na), na);
      tn.set(th);
      for (let a = 0; a < na; a++) {
        const m = idx[a];
        let v = th[m] + step[a];
        if (v < lo[m]) v = lo[m]; else if (v > hi[m]) v = hi[m];
        tn[m] = v;
      }
      const cn = sse(y, evalModel(nu, B, tn, nbase, kinds, null, fn));
      if (cn < cost) {
        rel = (cost - cn) / Math.max(cost, 1e-300);
        th.set(tn); cost = cn;
        lam = Math.max(lam / 3, 1e-9);
        accepted = true;
        break;
      }
      lam *= 4;
    }
    if (!accepted || rel < tol) break;
  }
  return it;
}

// fit rows [p0, p1) of Y (P, N) in place in TH (P, M); what a worker runs
export function lmBatch(job, Y, TH, iters, p0 = 0, p1 = null) {
  const { nu, B, lo, hi, free, nbase, kinds, maxIter, tol } = job;
  const N = nu.length, M = lo.length;
  p1 = p1 ?? Y.length / N;
  for (let p = p0; p < p1; p++) {
    const th = TH.subarray(p * M, (p + 1) * M);
    iters[p] = lmOne(nu, Y.subarray(p * N, (p + 1) * N), B, th, lo, hi, free, nbase, kinds, maxIter, tol);
  }
}

// --------------------------------------------------------------------------
// optional smoothing before the fit (as fitter.smooth_spectra)
// --------------------------------------------------------------------------
// (w, w) projection onto polynomials of `order` on offsets -h..h: row h is the
// Savitzky-Golay filter, the other rows evaluate an end-window fit
function sgHat(w, order) {
  const h = (w - 1) / 2, q = order + 1, A = new Float64Array(w * q);
  for (let i = 0; i < w; i++) for (let k = 0; k < q; k++) A[i * q + k] = (i - h) ** k;
  const pA = pinv(A, w, q), H = new Float64Array(w * w);
  for (let i = 0; i < w; i++) for (let j = 0; j < w; j++) {
    let s = 0;
    for (let k = 0; k < q; k++) s += A[i * q + k] * pA[k * w + j];
    H[i * w + j] = s;
  }
  return H;
}

// cube (ny * nx * K, row-major) -> the spectra the fit sees, Float64Array
export function smoothSpectra(cube, ny, nx, K, cfg) {
  const P = ny * nx;
  let Y = Float64Array.from(cube);
  const w = cfg.smooth_window || 0;
  if (w && K >= w) {
    const H = sgHat(w, cfg.smooth_order ?? 2), h = (w - 1) / 2, out = new Float64Array(Y.length);
    for (let p = 0; p < P; p++) {
      const o = p * K;
      for (let n = h; n < K - h; n++) {
        let s = 0;
        for (let i = 0; i < w; i++) s += H[h * w + i] * Y[o + n - h + i];
        out[o + n] = s;
      }
      for (let n = 0; n < h; n++) {
        let a = 0, b = 0;
        for (let i = 0; i < w; i++) { a += H[n * w + i] * Y[o + i]; b += H[(h + 1 + n) * w + i] * Y[o + K - w + i]; }
        out[o + n] = a; out[o + K - h + n] = b;
      }
    }
    Y = out;
  }
  const k = cfg.spatial_kernel || 0;
  if (k) {
    const r = (k - 1) / 2, g = Array.from({ length: k }, (_, i) => Math.exp(-0.5 * ((i - r) / (k / 4)) ** 2));
    // normalised 1-D Gaussian along Y (down the rows), then along X: the
    // product kernel, renormalised wherever it hangs over the map edge
    let out = new Float64Array(Y.length);
    for (let i = 0; i < ny; i++) {
      let den = 0;
      for (let d = -r; d <= r; d++) if (i + d >= 0 && i + d < ny) den += g[d + r];
      for (let j = 0; j < nx; j++) for (let c = 0; c < K; c++) {
        let s = 0;
        for (let d = -r; d <= r; d++) if (i + d >= 0 && i + d < ny) s += g[d + r] * Y[((i + d) * nx + j) * K + c];
        out[(i * nx + j) * K + c] = s / den;
      }
    }
    Y = out; out = new Float64Array(Y.length);
    for (let j = 0; j < nx; j++) {
      let den = 0;
      for (let d = -r; d <= r; d++) if (j + d >= 0 && j + d < nx) den += g[d + r];
      for (let i = 0; i < ny; i++) for (let c = 0; c < K; c++) {
        let s = 0;
        for (let d = -r; d <= r; d++) if (j + d >= 0 && j + d < nx) s += g[d + r] * Y[(i * nx + j + d) * K + c];
        out[(i * nx + j) * K + c] = s / den;
      }
    }
    Y = out;
  }
  return Y;
}

// --------------------------------------------------------------------------
// the fit, split into prepare (starting point) and finish (areas, errors)
// --------------------------------------------------------------------------
export function otsu(v) {
  let lo = Infinity, hi = -Infinity;
  for (const x of v) { if (x < lo) lo = x; if (x > hi) hi = x; }
  const nb = 128, h = new Float64Array(nb);
  for (const x of v) { let b = Math.floor((x - lo) / (hi - lo) * nb); if (b >= nb) b = nb - 1; if (b < 0) b = 0; h[b]++; }
  const tot = v.length;
  let w = 0, mu = 0, muT = 0;
  for (let b = 0; b < nb; b++) muT += (h[b] / tot) * (lo + (b + 0.5) * (hi - lo) / nb);
  let best = -1, thr = lo;
  for (let b = 0; b < nb; b++) {
    const p = h[b] / tot, c = lo + (b + 0.5) * (hi - lo) / nb;
    w += p; mu += p * c;
    const between = (muT * w - mu) ** 2 / Math.max(w * (1 - w), 1e-12);
    if (between > best) { best = between; thr = c; }
  }
  return thr;
}

function median(a) {
  const s = Float64Array.from(a).sort();
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

export function checkConfig(cfg) {
  if (!(cfg.range_hi > cfg.range_lo)) throw new Error("spectral range: high must exceed low");
  if (!(cfg.baseline_order >= -1 && cfg.baseline_order <= 6)) throw new Error("baseline order must be between -1 (none) and 6");
  const names = cfg.peaks.map(p => p.name);
  if (new Set(names).size !== names.length) throw new Error("peak names must be unique");
  for (const p of cfg.peaks) {
    if (!["Lorentz", "Gauss"].includes(p.shape)) throw new Error(`peak ${p.name}: shape must be Lorentz or Gauss`);
    if (!(p.pos_min <= p.pos && p.pos <= p.pos_max)) throw new Error(`peak ${p.name}: position ${p.pos} outside [${p.pos_min}, ${p.pos_max}]`);
    if (!(0 < p.width_min && p.width_min <= p.width && p.width <= p.width_max)) throw new Error(`peak ${p.name}: width ${p.width} outside [${p.width_min}, ${p.width_max}]`);
  }
  for (const r of [cfg.ratio_num, cfg.ratio_den]) if (!names.includes(r)) throw new Error(`ratio peak '${r}' is not in the peak table`);
  const w = cfg.smooth_window || 0, o = cfg.smooth_order ?? 2;
  if (w && (w < 3 || w % 2 === 0 || !(o >= 0 && o < w))) throw new Error("smoothing window must be odd, at least 3, and larger than the order");
  if (![0, 3, 5, 7].includes(cfg.spatial_kernel || 0)) throw new Error("spatial averaging must be off, 3, 5 or 7 pixels");
  return cfg;
}

// Everything up to the per-pixel LM: returns the job for lmBatch plus the
// state finishFit needs. cube: Float32Array (P * K), wn: (K,)
export function prepareFit(cube, P, wn, cfg) {
  checkConfig(cfg);
  const K = wn.length, ks = [];
  for (let c = 0; c < K; c++) if (wn[c] >= cfg.range_lo && wn[c] <= cfg.range_hi) ks.push(c);
  if (ks.length < 10) throw new Error("fewer than 10 channels inside the spectral range");
  const N = ks.length, nu = Float64Array.from(ks, c => wn[c]);
  const Y = new Float64Array(P * N);
  for (let p = 0; p < P; p++) for (let n = 0; n < N; n++) Y[p * N + n] = cube[p * K + ks[n]];
  const shapes = cfg.peaks.map(p => p.shape);
  const kinds = Int32Array.from(shapes, s => (s === "Lorentz" ? LORENTZ : GAUSS));
  const m = shapes.length, nbl = cfg.baseline_order + 1;
  const B = baselineBasis(nu, cfg.range_lo, cfg.range_hi, cfg.baseline_order);

  // fit separately: modified-polynomial baseline first (Lieber & Mahadevan-Jansen)
  let baseFixed = null;
  if (cfg.fit_separately && nbl) {
    const pB = pinv(B, N, nbl);
    baseFixed = new Float64Array(P * nbl);
    const z = new Float64Array(N), coef = new Float64Array(nbl);
    for (let p = 0; p < P; p++) {
      const y = Y.subarray(p * N, (p + 1) * N);
      z.set(y);
      const fitc = () => { for (let k = 0; k < nbl; k++) { let s = 0; for (let n = 0; n < N; n++) s += pB[k * N + n] * z[n]; coef[k] = s; } };
      fitc();
      for (let it = 0; it < 60; it++) {
        for (let n = 0; n < N; n++) { let b = 0; for (let k = 0; k < nbl; k++) b += coef[k] * B[n * nbl + k]; if (b < z[n]) z[n] = b; }
        fitc();
      }
      for (let k = 0; k < nbl; k++) baseFixed[p * nbl + k] = coef[k];
      for (let n = 0; n < N; n++) { let b = 0; for (let k = 0; k < nbl; k++) b += coef[k] * B[n * nbl + k]; y[n] -= b; }
    }
  }
  const jnb = baseFixed ? 0 : nbl;
  const Bj = jnb ? B : new Float64Array(0);
  const M = jnb + 3 * m;
  const lo = new Float64Array(M).fill(-Infinity), hi = new Float64Array(M).fill(Infinity);
  let free = new Uint8Array(M).fill(1);
  let theta0 = new Float64Array(M);
  cfg.peaks.forEach((p, j) => {
    const i = jnb + 3 * j;
    lo[i] = 0;
    lo[i + 1] = p.pos_min; hi[i + 1] = p.pos_max;
    lo[i + 2] = p.width_min; hi[i + 2] = p.width_max;
    theta0[i + 1] = p.pos; theta0[i + 2] = p.width;
    if (!p.fit || cfg.skip_fitting) free[i + 1] = free[i + 2] = 0;
  });

  const linearStart = th0 => {
    const q = jnb + m, A = new Float64Array(N * q), Phi = new Float64Array(N * m);
    for (let n = 0; n < N; n++) {
      for (let k = 0; k < jnb; k++) A[n * q + k] = Bj[n * jnb + k];
      for (let j = 0; j < m; j++) {
        const v = shapeEval(nu[n], th0[jnb + 3 * j + 1], th0[jnb + 3 * j + 2], shapes[j]);
        Phi[n * m + j] = v; A[n * q + jnb + j] = v;
      }
    }
    const pA = pinv(A, N, q), pB = jnb ? pinv(Bj, N, jnb) : null;
    const TH = new Float64Array(P * M), amps = new Float64Array(m), res = new Float64Array(N);
    for (let p = 0; p < P; p++) {
      const y = Y.subarray(p * N, (p + 1) * N), row = TH.subarray(p * M, (p + 1) * M);
      row.set(th0);
      for (let j = 0; j < m; j++) {
        let s = 0; const r = (jnb + j) * N;
        for (let n = 0; n < N; n++) s += pA[r + n] * y[n];
        amps[j] = Math.max(s, 0);
        row[jnb + 3 * j] = amps[j];
      }
      if (jnb) {
        for (let n = 0; n < N; n++) { let s = y[n]; for (let j = 0; j < m; j++) s -= amps[j] * Phi[n * m + j]; res[n] = s; }
        for (let k = 0; k < jnb; k++) { let s = 0; for (let n = 0; n < N; n++) s += pB[k * N + n] * res[n]; row[k] = s; }
      }
    }
    return TH;
  };

  const maxIter = cfg.skip_fitting ? Math.min(cfg.max_iter, 20) : cfg.max_iter;
  let TH = linearStart(theta0);
  let held = [];
  if (cfg.auto_shapes !== false && !cfg.skip_fitting && P > 1) {
    ({ theta0, free, held } = meanShapes(nu, Y, Bj, TH, lo, hi, free, jnb, kinds, cfg.peaks.map(p => p.name), maxIter));
    TH = linearStart(theta0);
  }
  const job = { nu, B: Bj, lo, hi, free, nbase: jnb, kinds, maxIter, tol: 1e-7 };
  return { job, Y, TH, P, N, M, m, nbl, jnb, B, baseFixed, shapes, held, cfg };
}

// shapes from the mean spectrum of the carbon pixels (as fitter._mean_shapes)
function meanShapes(nu, Y, B, TH, lo, hi, free, nbase, kinds, names, maxIter) {
  const N = nu.length, M = lo.length, P = Y.length / N;
  const signal = new Float64Array(P);
  for (let p = 0; p < P; p++) { let s = 0; for (let j = 0; j < kinds.length; j++) s += TH[p * M + nbase + 3 * j]; signal[p] = Math.log1p(s); }
  let smin = Infinity, smax = -Infinity;
  for (const s of signal) { smin = Math.min(smin, s); smax = Math.max(smax, s); }
  let part = new Uint8Array(P).fill(1);
  if (smax > smin) { const t = otsu(signal); for (let p = 0; p < P; p++) part[p] = signal[p] > t ? 1 : 0; }
  let np = part.reduce((a, b) => a + b, 0);
  if (np < 2) { part.fill(1); np = P; }
  const ym = new Float64Array(N), th = new Float64Array(M);
  const d2 = new Float64Array(np * (N - 2));
  let q = 0;
  for (let p = 0; p < P; p++) {
    if (!part[p]) continue;
    for (let n = 0; n < N; n++) ym[n] += Y[p * N + n] / np;
    for (let k = 0; k < M; k++) th[k] += TH[p * M + k] / np;
    for (let n = 0; n < N - 2; n++) d2[q++] = Math.abs(Y[p * N + n + 2] - 2 * Y[p * N + n + 1] + Y[p * N + n]);
  }
  lmOne(nu, ym, B, th, lo, hi, free, nbase, kinds, maxIter, 1e-9);

  // standard errors a single pixel would have, at the map's own noise level
  const sigma = median(d2) / (0.6745 * Math.sqrt(6));
  const J = new Float64Array(N * M);
  evalModel(nu, B, th, nbase, kinds, J, new Float64Array(N));
  const fi = [];
  for (let k = 0; k < M; k++) if (free[k]) fi.push(k);
  const nf = fi.length, JtJ = new Float64Array(nf * nf);
  for (let n = 0; n < N; n++) for (let a = 0; a < nf; a++) {
    const ja = J[n * M + fi[a]];
    if (ja === 0) continue;
    for (let b = 0; b < nf; b++) JtJ[a * nf + b] += ja * J[n * M + fi[b]];
  }
  const cov = pinvSym(JtJ, nf), se = new Float64Array(M);
  fi.forEach((k, a) => { se[k] = sigma * Math.sqrt(Math.max(cov[a * nf + a], 0)); });

  const fr = Uint8Array.from(free), held = [];
  names.forEach((name, j) => {
    const i = nbase + 3 * j;
    if (!fr[i + 1]) return;                    // held by the user: keeps the table's shape
    const A = th[i], w = th[i + 2];
    if (A <= 1e-9 || se[i + 2] > 0.25 * w || se[i + 1] > 0.5 * w) { fr[i + 1] = fr[i + 2] = 0; held.push(name); }
  });
  return { theta0: th, free: fr, held };
}

// after the LM: residual statistics, areas, the result in the server's layout
export function finishFit(st, iters, seconds) {
  const { job, Y, TH, P, N, M, m, nbl, jnb, B, baseFixed, shapes, held, cfg } = st;
  const { nu, kinds } = job;
  const f = new Float64Array(N), ssr = new Float64Array(P), chi = new Float64Array(P);
  for (let p = 0; p < P; p++) {
    evalModel(nu, job.B, TH.subarray(p * M, (p + 1) * M), jnb, kinds, null, f);
    let s = 0, c = 0;
    for (let n = 0; n < N; n++) {
      let fv = f[n], yv = Y[p * N + n];
      if (baseFixed) {
        let b = 0;
        for (let k = 0; k < nbl; k++) b += baseFixed[p * nbl + k] * B[n * nbl + k];
        fv += b; yv += b;
      }
      const d = yv - fv;
      s += d * d; c += d * d / Math.max(fv, 1);
    }
    ssr[p] = s; chi[p] = c;
  }
  let nFree = job.free.reduce((a, b) => a + b, 0) + (baseFixed ? nbl : 0);
  const dof = Math.max(N - 3 * m, 1);
  const out = {
    names: cfg.peaks.map(p => p.name), shapes, m,
    intensity: new Float64Array(P * m), position: new Float64Array(P * m), width: new Float64Array(P * m),
    area: new Float64Array(P * m), baseline: new Float64Array(P * nbl),
    std_error: new Float64Array(P), red_chi2: new Float64Array(P), n_iter: new Float64Array(P),
    nb: baseFixed ? nbl : jnb, range: [cfg.range_lo, cfg.range_hi], n_channels: N, n_free: nFree,
    seconds, held,
  };
  for (let p = 0; p < P; p++) {
    for (let j = 0; j < m; j++) {
      const i = p * M + jnb + 3 * j, A = TH[i], c = TH[i + 1], w = TH[i + 2];
      out.intensity[p * m + j] = A; out.position[p * m + j] = c; out.width[p * m + j] = w;
      out.area[p * m + j] = windowedArea(A, c, w, shapes[j], cfg.range_lo, cfg.range_hi);
    }
    const nbOut = out.nb;
    for (let k = 0; k < nbOut; k++) out.baseline[p * nbOut + k] = baseFixed ? baseFixed[p * nbl + k] : TH[p * M + k];
    out.std_error[p] = Math.sqrt(ssr[p] / dof); out.red_chi2[p] = chi[p] / dof; out.n_iter[p] = iters[p];
  }
  return out;
}

// single-threaded convenience (tests, small maps)
export function fitSpectra(cube, P, wn, cfg) {
  const t0 = Date.now();
  const st = prepareFit(cube, P, wn, cfg);
  const iters = new Int32Array(P);
  lmBatch(st.job, st.Y, st.TH, iters);
  return finishFit(st, iters, (Date.now() - t0) / 1000);
}
