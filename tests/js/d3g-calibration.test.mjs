// node --test tests/js/d3g-calibration.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { BANDS, QC_SETTINGS, PRESET_21OCT, noiseSigma, despike, savgol, prepare, boxSolver, unmix, freeFits, calibrate, standards,
         blocksOf, blockMedian, compareMedians, mapBlock, stageBlock, toJSON, fromJSON, shapeArea,
         medianSpectrumBand, refineOnMedian, standardRatio, scaleIndex } from "../../web/js/d3g-calibration.js";
import { rng } from "../../web/js/pf-analysis.js";

const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ""} ${a} vs ${b}`);
const gauss = R => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
const lor = (x, c, w) => 1 / (1 + ((x - c) / (w / 2)) ** 2), gau = (x, c, w) => Math.exp(-4 * Math.LN2 * ((x - c) / w) ** 2);
const WN = Float64Array.from({ length: 294 }, (_, i) => 1000.66 + i * (798.35 / 293));

// a carbon spectrum on the 21 Oct shapes: heights by band, a sloping background, white noise
function spectrum(h, { shapes = PRESET_21OCT.steps[2].shapes, bg = [60, -8, 4], sd = 0, R = null } = {}) {
  return Float64Array.from(WN, x => {
    const t = (x - 1400) / 400;
    let v = bg[0] + bg[1] * t + bg[2] * t * t;
    for (const b of BANDS) v += (h[b] || 0) * (QC_SETTINGS.bands[b].shape === "Lorentz" ? lor(x, shapes[b].pos, shapes[b].fwhm) : gau(x, shapes[b].pos, shapes[b].fwhm));
    return v + (sd && R ? sd * gauss(R) : 0);
  });
}

test("noise from the second difference recovers white noise", () => {
  const R = rng(5), y = Float64Array.from({ length: 4000 }, () => 100 + 3 * gauss(R));
  near(noiseSigma(y), 3, 0.15);
});

test("despiking removes a cosmic ray and keeps a band", () => {
  const R = rng(2), y = spectrum({ G: 200, D1: 60, D3: 10 }, { sd: 2, R });
  const spiked = Float64Array.from(y); spiked[40] += 900; spiked[41] += 500;
  const d = despike(spiked);
  assert.equal(d.spikes, 1);
  assert.ok(Math.abs(d.y[40] - y[40]) < 15 && Math.abs(d.y[41] - y[41]) < 15, "the spike's points are put back near the spectrum");
  // the G band is several points wide: left alone
  const g = WN.findIndex(x => x > 1571.5);
  assert.equal(despike(y).spikes, 0);
  near(despike(y).y[g], y[g], 1e-12);
});

test("Savitzky-Golay 11/2 has the textbook centre weights and keeps a quadratic, ends included", () => {
  const imp = j => savgol(Float64Array.from({ length: 31 }, (_, i) => (i === 15 + j ? 1 : 0)))[15];
  const want = [-36, 9, 44, 69, 84, 89, 84, 69, 44, 9, -36].map(v => v / 429);
  for (let j = -5; j <= 5; j++) near(imp(j), want[j + 5], 1e-12, `weight ${j}`);
  const q = Float64Array.from({ length: 40 }, (_, i) => 3 - 0.2 * i + 0.05 * i * i), s = savgol(q);
  for (let i = 0; i < 40; i++) near(s[i], q[i], 1e-9, `point ${i}`);
});

test("the carbon test passes a strong spectrum and fails a flat one; NaN spectra are left out", () => {
  const R = rng(3), K = WN.length, cube = new Float32Array(3 * K);
  cube.set(spectrum({ G: 150, D1: 50 }, { sd: 2, R }), 0);
  cube.set(spectrum({}, { sd: 2, R }), K);
  cube.fill(NaN, 2 * K);
  const p = prepare(cube, K, WN, 3);
  assert.deepEqual(p.carbon, [0]);
  assert.ok(p.snr[0] > 30 && p.snr[1] < 10 && Number.isNaN(p.snr[2]));
});

test("the bounded solver: on random problems no feasible point nearby does better", () => {
  const R = rng(11);
  for (let t = 0; t < 40; t++) {
    const N = 30, q = 5, nb = 3, hmax = 2, X = Float64Array.from({ length: N * q }, () => gauss(R)), y = Float64Array.from({ length: N }, () => 2 * gauss(R));
    const s = boxSolver(X, N, q, nb, hmax), bv = new Float64Array(q);
    for (let n = 0; n < N; n++) for (let k = 0; k < q; k++) bv[k] += X[n * q + k] * y[n];
    const x = s.solve(bv), cost = v => { let c = 0; for (let n = 0; n < N; n++) { let f = 0; for (let k = 0; k < q; k++) f += X[n * q + k] * v[k]; c += (y[n] - f) ** 2; } return c; };
    for (let j = q - nb; j < q; j++) assert.ok(x[j] >= 0 && x[j] <= hmax);
    // 400 random feasible points around the answer, none lower
    const c0 = cost(x);
    for (let k = 0; k < 400; k++) {
      const v = Float64Array.from(x, (xi, i) => xi + 0.05 * gauss(R));
      for (let j = q - nb; j < q; j++) v[j] = Math.min(hmax, Math.max(0, v[j]));
      assert.ok(cost(v) >= c0 - 1e-9, `problem ${t}`);
    }
  }
});

test("held shapes recover the heights exactly without noise, a negative band clamps to zero, ratios follow", () => {
  const K = WN.length, h = { D4: 5, D1: 70, D3: 9, G: 160, D2: 25 }, cube = new Float32Array(2 * K);
  cube.set(spectrum(h, { bg: [60, -8, 4] }), 0);
  // the second spectrum with D4 taken away below zero: the best fit holds it at 0
  cube.set(Float64Array.from(spectrum({ ...h, D4: 0 }), (v, i) => v - 6 * lor(WN[i], 1210, 120)), K);
  const p = prepare(cube, K, WN, 2), U = unmix(p, PRESET_21OCT.steps[2].shapes);
  BANDS.forEach((b, j) => near(U.h[j], h[b], 1e-3, b));
  near(U.ratio["ID3/IG"][0], 9 / 160, 1e-6);
  near(U.ratio["AD3/AG"][0], shapeArea("Gauss", 9, PRESET_21OCT.steps[2].shapes.D3.fwhm) / shapeArea("Lorentz", 160, PRESET_21OCT.steps[2].shapes.G.fwhm), 1e-6);
  assert.equal(U.h[5 + BANDS.indexOf("D4")], 0);
  assert.ok(U.r2[0] > 0.999999);
});

test("the height errors from the linear fit match the scatter of repeated noisy spectra", () => {
  const R = rng(17), K = WN.length, n = 300, cube = new Float32Array(n * K);
  for (let p = 0; p < n; p++) cube.set(spectrum({ D4: 4, D1: 50, D3: 6, G: 120, D2: 20 }, { sd: 3, R }), p * K);
  const pr = prepare(cube, K, WN, n), U = unmix(pr, PRESET_21OCT.steps[2].shapes, Array.from({ length: n }, (_, i) => i)), nb = BANDS.length;
  for (const b of ["D3", "G"]) {
    const j = BANDS.indexOf(b), v = Array.from({ length: n }, (_, i) => U.h[i * nb + j]), m = v.reduce((t, x) => t + x, 0) / n;
    const sd = Math.sqrt(v.reduce((t, x) => t + (x - m) ** 2, 0) / (n - 1)), se = U.se.filter((_, i) => i % nb === j).reduce((t, x) => t + x, 0) / n;
    near(se / sd, 1, 0.12, b);
  }
});

test("free fits and one calibration step recover the shapes of noisy spectra", () => {
  const R = rng(23), K = WN.length, n = 120, cube = new Float32Array(n * K), truth = { ...PRESET_21OCT.steps[2].shapes, G: { pos: 1575, fwhm: 26 }, D1: { pos: 1345, fwhm: 80 } };
  for (let p = 0; p < n; p++) cube.set(spectrum({ D4: 1, D1: 90, D3: 4, G: 200, D2: 30 }, { shapes: truth, sd: 2, R }), p * K);
  const pr = prepare(cube, K, WN, n), prev = PRESET_21OCT.steps[2].shapes;
  const F = freeFits(pr, pr.carbon, prev), cal = calibrate(F, pr.sig, prev);
  near(cal.G.pos, 1575, 0.2); near(cal.G.fwhm, 26, 0.5); near(cal.D1.pos, 1345, 0.5); near(cal.D1.fwhm, 80, 2);
  assert.ok(cal.G.support === n && cal.D1.support === n);
  // D4 at 1 count is under 3 x noise everywhere: it keeps the step before's shape
  assert.equal(cal.D4.note, "kept"); assert.ok(cal.D4.support <= 0); assert.equal(cal.D4.pos, prev.D4.pos);
});

test("the standards step takes G from graphite and the D bands from glassy carbon", () => {
  const R = rng(29), noisy = f => Float64Array.from(WN, x => f(x) + 2 * gauss(R));
  const graphite = { wn: WN, y: noisy(x => 80 + 2000 * lor(x, 1578, 15) + 40 * lor(x, 1350, 40)) };
  const glassy = { wn: WN, y: noisy(x => 200 + 800 * lor(x, 1590, 30) + 2400 * lor(x, 1348, 85) + 260 * gau(x, 1505, 160) + 2100 * lor(x, 1612, 45) + 90 * lor(x, 1205, 110)) };
  const s = standards(graphite, glassy).shapes;
  near(s.G.pos, 1578, 0.3); near(s.G.fwhm, 15, 0.5); assert.equal(s.G.note, "graphite");
  near(s.D1.pos, 1348, 1); assert.equal(s.D1.note, "glassy carbon");
  assert.ok(s.D3.fwhm > 120 && s.D3.pos >= 1490 && s.D3.pos <= 1510);
});

test("block medians: intervals cover, comparisons are exact in their centre and repeatable", () => {
  const R = rng(31), nx = 40, ny = 40, a = [], b = [], ka = [], kb = [];
  for (let p = 0; p < nx * ny; p++) { a.push(0.05 + 0.01 * gauss(R)); ka.push(mapBlock(p, nx)); b.push(0.025 + 0.008 * gauss(R)); kb.push(mapBlock(p, nx)); }
  const A = blocksOf(a, ka), B = blocksOf(b, kb);
  assert.equal(A.length, 25);
  const m = blockMedian(A);
  assert.ok(m.lo < 0.05 && m.hi > 0.05 && m.hi - m.lo < 0.004);
  const c = compareMedians(A, B), c2 = compareMedians(A, B);
  near(c.diff, m.median - blockMedian(B).median, 1e-15);
  assert.deepEqual([c.lo, c.hi], [c2.lo, c2.hi]);
  assert.ok(c.lo < 0.025 && c.hi > 0.025);
  assert.equal(stageBlock(-1, 151), "-1,1");
});

// an uncoated-like map: D3 far below the noise in every spectrum
function weakD3Map(d3, n = 400, seed = 37) {
  const R = rng(seed), K = WN.length, cube = new Float32Array(n * K), truth = { ...PRESET_21OCT.steps[2].shapes, D3: { pos: 1500, fwhm: 160 } };
  for (let p = 0; p < n; p++) cube.set(spectrum({ D4: 2, D1: 40, D3: d3, G: 150, D2: 15 }, { shapes: truth, sd: 3, R }), p * K);
  return prepare(cube, K, WN, n);
}

// D3/G 0.02 at 3 counts of noise: about 1 x noise in each spectrum, 15 x the residual in the
// median of 400 (its position then scatters about 3 cm-1 from map to map, 7 at D3/G 0.01)
test("D3 that no single spectrum carries is measured on the median spectrum", () => {
  const p = weakD3Map(3), prev = PRESET_21OCT.steps[0].shapes;
  const per = calibrate(freeFits(p, p.carbon, prev), p.sig, prev);
  assert.equal(per.D3.note, "kept");                       // under 3 x noise in almost every spectrum
  const m = medianSpectrumBand(p, p.carbon, prev, "D3");
  assert.ok(m.present, `z ${m.z}`);
  near(m.pos, 1500, 5); near(m.fwhm, 160, 15);
  const r = refineOnMedian(per, p, p.carbon, prev, "D3");
  assert.match(r.shapes.D3.note, /^median spectrum/);
  near(r.shapes.D3.pos, m.pos, 1e-12);
  assert.equal(r.shapes.G, per.G);                         // the other bands are left as they were
});

test("without D3 the median spectrum finds none, and glassy carbon's shape stays", () => {
  const p = weakD3Map(0), prev = PRESET_21OCT.steps[0].shapes;
  const per = calibrate(freeFits(p, p.carbon, prev), p.sig, prev), r = refineOnMedian(per, p, p.carbon, prev, "D3");
  assert.ok(!r.median.present, `z ${r.median.z}`);
  assert.equal(r.shapes.D3, per.D3);
  assert.equal(r.shapes.D3.pos, prev.D3.pos);
});

test("glassy carbon's ratio with held shapes, and the scale from the uncoated reference to it", () => {
  const sh = PRESET_21OCT.steps[2].shapes, y = spectrum({ D4: 40, D1: 2300, D3: 280, G: 720, D2: 2400 }, { bg: [300, 10, 5] });
  near(standardRatio({ wn: WN, y }, sh).ratio, 280 / 720, 1e-9);
  near(scaleIndex(0.06, 0.025, 0.39), 0.035 / 0.365, 1e-15);
  assert.ok(Number.isNaN(scaleIndex(0.06, 0.025, 0)));     // glassy carbon not above the reference: no scale
  const R = rng(41), a = [], ka = [];
  for (let p = 0; p < 1600; p++) { a.push(0.05 + 0.01 * gauss(R)); ka.push(mapBlock(p, 40)); }
  const A = blocksOf(a, ka), c = compareMedians(A, A, { top: 0.39 });
  near(c.index, 0, 1e-15);
  assert.ok(c.indexLo < 0 && c.indexHi > 0);
  assert.equal(compareMedians(A, A).index, undefined);     // no top, no scale
});

test("calibration files: this app's round trip and the calibration scripts' layout", () => {
  const back = fromJSON(toJSON(PRESET_21OCT));
  assert.equal(back.steps.length, 3);
  assert.equal(back.steps[2].shapes.G.pos, PRESET_21OCT.steps[2].shapes.G.pos);
  assert.equal(back.steps[2].glassy, PRESET_21OCT.steps[2].glassy);
  const pkg = { description: "x", qcarbon_settings: { "range_cm-1": [1000, 1800], baseline_poly_order: 2, height_range: [0, 5000],
    bands: Object.fromEntries(BANDS.map(b => { const w = QC_SETTINGS.bands[b]; return [b, { position: w.pos, position_min: w.pos_min, position_max: w.pos_max, width: w.width, width_min: w.width_min, width_max: w.width_max, shape: w.shape }]; })) },
    "baseline_2 - USE FOR ALL SAMPLES": Object.fromEntries(BANDS.map(b => [b, { "position_cm-1": PRESET_21OCT.steps[2].shapes[b].pos, "FWHM_cm-1": PRESET_21OCT.steps[2].shapes[b].fwhm, note: "calibrated" }])) };
  const cal = fromJSON(JSON.stringify(pkg), "pkg");
  assert.equal(cal.steps.length, 1); assert.equal(cal.settings.hmax, 5000); assert.equal(cal.settings.bands.D3.shape, "Gauss");
  assert.throws(() => fromJSON(JSON.stringify({ a: 1 }), "bad"), /not a D3\/G calibration/);
});
