// node --test tests/js/mva.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { MVA_DEFAULTS, normalise, symEig, pca, kmeans, silhouette, ward, cutTree, compareLoadings, runMva, identify } from "../../web/js/mva.js";
import { prepare } from "../../web/js/layers.js";
import { rng } from "../../web/js/pf-analysis.js";

const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ""} ${a} vs ${b}`);
const gauss = R => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());

// three materials (polymer-like band sets) of different brightness
const WN = Float64Array.from({ length: 1401 }, (_, i) => 400 + 2 * i), K = WN.length;
const lor = (x, c, w) => 1 / (1 + (2 * (x - c) / w) ** 2);
const BANDS = {
  A: [[1063, 12, 0.6], [1130, 12, 0.55], [1296, 10, 0.7], [1440, 16, 1], [2850, 18, 1.4], [2883, 16, 1.6]],
  B: [[1003, 8, 0.5], [1240, 30, 0.6], [1640, 26, 1], [2930, 40, 1.2]],
  C: [[850, 14, 0.5], [1090, 20, 0.6], [1350, 30, 0.4], [1455, 20, 0.8], [2910, 40, 1.0]],
};
const NAMES = ["A", "B", "C"], SPEC = NAMES.map(m => Float64Array.from(WN, x => BANDS[m].reduce((t, [c, w, h]) => t + h * lor(x, c, w), 0)));
const BRIGHT = [1000, 400, 250], REFS = NAMES.map((name, a) => ({ name, wn: WN, y: SPEC[a] }));

// a map whose point (i, j) holds the materials in amounts f(i, j) (fractions of each one's brightness)
function map(nx, ny, f, { noise = 8, seed = 1 } = {}) {
  const R = rng(seed), cube = new Float32Array(nx * ny * K);
  for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
    const a = f(i, j), o = (i * nx + j) * K;
    for (let c = 0; c < K; c++) {
      let v = 200 + 0.05 * (WN[c] - 400) + noise * gauss(R);
      for (let m = 0; m < 3; m++) v += a[m] * BRIGHT[m] * SPEC[m][c];
      cube[o + c] = v;
    }
  }
  return { cube, geom: { nx, ny }, f };
}
const run = (data, over = {}, refs = []) => {
  const S = { ...MVA_DEFAULTS, ...over }, prep = prepare(data.cube, K, WN, data.geom, S);
  return { prep, R: runMva(prep, S, refs), S };
};
// three regions, one material each
const regions = (nx = 30, ny = 18) => map(nx, ny, (i, j) => [j < nx / 3 ? 1 : 0, j >= nx / 3 && j < 2 * nx / 3 ? 1 : 0, j >= 2 * nx / 3 ? 1 : 0]);
const truthRegion = (p, nx) => { const j = p % nx; return j < nx / 3 ? 0 : j < 2 * nx / 3 ? 1 : 2; };
// how well classes match the truth: the share of points in their class's majority truth
function purity(labels, truth, k) {
  const tab = Array.from({ length: k }, () => [0, 0, 0]);
  labels.forEach((c, i) => tab[c][truth[i]]++);
  return tab.reduce((t, r) => t + Math.max(...r), 0) / labels.length;
}

test("Ward by the nearest-neighbour chain matches a brute-force Ward, heights and classes", () => {
  const R = rng(3), m = 40, d = 3, P = new Float64Array(m * d);
  for (let i = 0; i < m; i++) for (let c = 0; c < d; c++) P[i * d + c] = (i % 3) * 4 + gauss(R);
  const merges = ward(P, m, d);
  // brute force: merge the pair whose union raises the sum of squares least; height sqrt(2 dSSE)
  let cl = Array.from({ length: m }, (_, i) => ({ n: 1, c: Array.from(P.slice(i * d, i * d + d)) }));
  const hs = [];
  while (cl.length > 1) {
    let best = null;
    for (let a = 0; a < cl.length; a++) for (let b = a + 1; b < cl.length; b++) {
      const A = cl[a], B = cl[b], e = A.c.reduce((t, v, q) => t + (v - B.c[q]) ** 2, 0) * A.n * B.n / (A.n + B.n);
      if (!best || e < best.e) best = { a, b, e };
    }
    const A = cl[best.a], B = cl[best.b], n = A.n + B.n;
    hs.push(Math.sqrt(2 * best.e));
    cl = cl.filter((_, i) => i !== best.a && i !== best.b).concat([{ n, c: A.c.map((v, q) => (A.n * v + B.n * B.c[q]) / n) }]);
  }
  hs.sort((a, b) => a - b);
  merges.forEach(([, , h], i) => near(h, hs[i], 1e-9, `merge ${i}`));
  const lab = cutTree(merges, m, 3), truth = Array.from({ length: m }, (_, i) => i % 3);
  assert.equal(purity(Array.from(lab), truth, 3), 1);
});

test("the symmetric eigen-solver: A v = w v, orthonormal vectors", () => {
  const R = rng(9), n = 60, A = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) A[i * n + j] = A[j * n + i] = gauss(R);
  const { w, V } = symEig(A, n);
  let worst = 0, orth = 0;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    let Av = 0; for (let k = 0; k < n; k++) Av += A[i * n + k] * V[k][j];
    worst = Math.max(worst, Math.abs(Av - w[j] * V[i][j]));
  }
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) { let t = 0; for (let k = 0; k < n; k++) t += V[k][a] * V[k][b]; orth = Math.max(orth, Math.abs(t - (a === b ? 1 : 0))); }
  assert.ok(worst < 1e-10 && orth < 1e-10, `residual ${worst}, orthogonality ${orth}`);
});

test("k-means finds three blobs and the silhouette prefers three", () => {
  const R = rng(5), n = 300, d = 2, Z = new Float64Array(n * d), truth = [];
  for (let i = 0; i < n; i++) { const c = i % 3; truth.push(c); Z[i * d] = 6 * c + gauss(R); Z[i * d + 1] = 3 * (c === 1) + gauss(R); }
  const idx = Array.from({ length: n }, (_, i) => i), s = {};
  for (const k of [2, 3, 4, 5]) s[k] = silhouette(Z, d, kmeans(Z, n, d, k).labels, k, idx);
  assert.equal(Object.entries(s).sort((a, b) => b[1] - a[1])[0][0], "3", JSON.stringify(s));
  assert.ok(purity(Array.from(kmeans(Z, n, d, 3).labels), truth, 3) > 0.97);
});

test("PCA: orthonormal loadings, scores that rebuild the spectra, and the number of components the data hold", () => {
  // three materials in independent amounts: three components about the mean
  const nx = 24, ny = 16, data = map(nx, ny, (i, j) => [j / nx, i / ny, 0.5 + 0.5 * Math.sin(i * 0.7 + j * 0.4)]);
  const { prep, R } = run(data, { method: "pca", norm: "none" });
  assert.equal(R.k, 3, `suggested ${R.pca.suggested}: over ${R.pca.over.slice(0, 6)}`);
  const nb = prep.nb, L = R.L;
  for (let a = 0; a < R.k; a++) for (let b = 0; b < R.k; b++) {
    let t = 0; for (let j = 0; j < nb; j++) t += L[a * nb + j] * L[b * nb + j];
    near(t, a === b ? 1 : 0, 1e-9, `loadings ${a}, ${b}`);
  }
  // what is left is noise (the noise from second differences reads a little high next to sharp bands)
  const med = Float64Array.from(R.resid).sort()[R.resid.length >> 1];
  assert.ok(med > 0.6 && med < 1.1, `median residual ${med} × the noise`);
  // the same map in pure regions: two components about the mean (three spectra in a plane)
  assert.equal(run(regions(), { method: "pca", norm: "none" }).R.k, 2);
});

test("MCR recovers the three materials of a mixed map; CLS against references gives their shares", () => {
  const nx = 24, ny = 16, f = (i, j) => [j / nx, i / ny, 0.5 + 0.5 * Math.sin(i * 0.7 + j * 0.4)], data = map(nx, ny, f);
  const { prep, R } = run(data, { method: "mcr" });
  assert.equal(R.k, 3, JSON.stringify(R.steps));
  const id = identify(R, prep, REFS, 80);
  assert.deepEqual(id.map(x => NAMES[x.best]).sort(), ["A", "B", "C"]);
  id.forEach((x, a) => assert.ok(x.hqi[x.best] > 95, `component ${a} HQI ${x.hqi[x.best]}`));
  // shares by CLS against the true spectra: each material's part of the spectrum's band area
  const C = run(data, { method: "refs" }, REFS).R, area = SPEC.map(s => s.reduce((t, v) => t + v, 0));
  let worst = 0;
  for (let p = 0; p < nx * ny; p += 7) {
    const i = Math.floor(p / nx), j = p % nx, a = f(i, j).map((v, m) => v * BRIGHT[m] * area[m]), tot = a.reduce((t, v) => t + v, 0);
    const row = prep.rowOf[p];
    for (let m = 0; m < 3; m++) if (tot > 0) worst = Math.max(worst, Math.abs(C.share[row * 3 + m] - a[m] / tot));
  }
  assert.ok(worst < 0.04, `largest share error ${worst}`);
});

test("K-means and HCA find the three regions, and their class spectra are the materials", () => {
  const nx = 30, data = regions(nx), truth = Array.from({ length: nx * 18 }, (_, p) => truthRegion(p, nx));
  for (const method of ["kmeans", "hca"]) {
    const { prep, R } = run(data, { method });
    assert.equal(R.k, 3, `${method}: ${JSON.stringify(R.sil)}`);
    const lab = Array.from(prep.list, (p, i) => R.labels[i]), tr = Array.from(prep.list, p => truth[p]);
    assert.ok(purity(lab, tr, 3) > 0.99, method);
    const id = identify(R, prep, REFS, 80);
    assert.deepEqual(id.map(x => NAMES[x.best]).sort(), ["A", "B", "C"], method);
    if (method === "hca") assert.ok(R.tree.nodes.length > 0 && R.tree.cut > 0);
  }
});

test("the comparator pairs the same components across methods", () => {
  const data = regions();
  const a = run(data, { method: "mcr", k: 3 }).R, b = run(data, { method: "kmeans", k: 3 }).R, c = run(data, { method: "pca", k: 2 }).R;
  const ab = compareLoadings(a.L, a.k, b.L, b.k, a.nb);
  assert.equal(ab.pairs.length, 3);
  assert.ok(ab.mean > 0.97, `MCR against K-means: mean |r| ${ab.mean}`);
  // PCA loadings are differences of spectra, not spectra: they agree less, and the table says so
  assert.ok(compareLoadings(a.L, a.k, c.L, c.k, a.nb).mean < ab.mean);
});

test("scaling to unit length keeps empty points small and the noise follows the scale", () => {
  const nx = 12, data = map(nx, 6, (i, j) => (j < 3 ? [0, 0, 0] : [0.2 + 0.8 * j / nx, 0, 0]));
  const prep = prepare(data.cube, K, WN, data.geom, MVA_DEFAULTS), N = normalise(prep, "vector"), nb = prep.nb;
  const len = i => Math.sqrt(N.X.subarray(i * nb, (i + 1) * nb).reduce((t, v) => t + v * v, 0));
  const full = Array.from(prep.list, (p, i) => [p % nx, len(i), N.noise[i] / prep.noise[i] - N.scale[i]]);
  const signal = full.filter(([j]) => j >= 3).map(([, l]) => l), empty = full.filter(([j]) => j < 3).map(([, l]) => l);
  near(Math.max(...signal) / Math.min(...signal), 1, 1e-6, "every spectrum with signal at the same length");
  assert.ok(Math.max(...empty) < 0.4 * Math.min(...signal), `empty points stay small: ${Math.max(...empty)} against ${Math.min(...signal)}`);
  full.forEach(([, , d]) => near(d, 0, 1e-9, "noise scaled with the spectrum"));
});
