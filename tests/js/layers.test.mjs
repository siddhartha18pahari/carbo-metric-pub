// node --test tests/js/layers.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { LAYER_DEFAULTS, baselineFitter, prepare, purest, decompose, matchRefs, findLayers, thicknessTest, amountMap, layerSpectra } from "../../web/js/layers.js";
import { rng } from "../../web/js/pf-analysis.js";

const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ""} ${a} vs ${b}`);
const gauss = R => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
// erf to 1.5e-7 (Abramowitz and Stegun 7.1.26)
const erf = x => { const s = Math.sign(x), t = 1 / (1 + 0.3275911 * Math.abs(x));
  return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)); };
const Phi = z => 0.5 * (1 + erf(z / Math.SQRT2));

// three materials (polymer-like band sets), each with its own brightness
const WN = Float64Array.from({ length: 1401 }, (_, i) => 400 + 2 * i);
const lor = (x, c, w) => 1 / (1 + (2 * (x - c) / w) ** 2);
const BANDS = {
  A: [[1063, 12, 0.6], [1130, 12, 0.55], [1296, 10, 0.7], [1440, 16, 1], [2850, 18, 1.4], [2883, 16, 1.6]],
  B: [[1003, 8, 0.5], [1240, 30, 0.6], [1640, 26, 1], [2930, 40, 1.2]],
  C: [[850, 14, 0.5], [1090, 20, 0.6], [1350, 30, 0.4], [1455, 20, 0.8], [2910, 40, 1.0]],
};
const SPEC = Object.fromEntries(Object.entries(BANDS).map(([m, b]) => [m, Float64Array.from(WN, x => b.reduce((t, [c, w, h]) => t + h * lor(x, c, w), 0))]));
const BRIGHT = { A: 1000, B: 400, C: 250 };
const REFS = Object.keys(SPEC).map(m => ({ name: m, wn: WN, y: SPEC[m] }));

// a spectrum at position s through a stack of [material, from, to], blurred by a Gaussian spot
function spectrum(s, stack, R, { sigma = 0.4, noise = 8 } = {}) {
  const y = Float64Array.from(WN, x => 200 + 0.05 * (x - 400) + noise * gauss(R));
  for (const [m, a, b] of stack) {
    const amt = Phi((s - a) / sigma) - Phi((s - b) / sigma);
    if (amt > 1e-7) for (let c = 0; c < WN.length; c++) y[c] += BRIGHT[m] * amt * SPEC[m][c];
  }
  return y;
}
// a line scan across the stack, as a 1-row map
function lineScan(stack, s0, s1, ds, opts = {}, seed = 1) {
  const R = rng(seed), xs = []; for (let s = s0; s <= s1 + 1e-9; s += ds) xs.push(s);
  const cube = new Float32Array(xs.length * WN.length);
  xs.forEach((s, p) => cube.set(spectrum(s, stack, R, opts), p * WN.length));
  return { cube, geom: { nx: xs.length, ny: 1, x: Float64Array.from(xs), y: Float64Array.of(0) } };
}
const run = (data, settings = {}, refs = []) => {
  const S = { ...structuredClone(LAYER_DEFAULTS), ...settings };
  const prep = prepare(data.cube, WN.length, WN, data.geom, S), comps = decompose(prep, S, refs);
  return { S, prep, comps, res: findLayers(prep, comps, data.geom, S) };
};
// each material found named by its best-matching true spectrum
const namesOf = (r) => matchRefs(r.comps, r.prep, REFS, 0).map(m => REFS[m.best].name);

const STACK = [["A", 0, 6], ["C", 6, 8.4], ["B", 8.4, 13], ["A", 13, 20]];

test("modified polyfit settles under the bands and recovers the background", () => {
  const nu = Float64Array.from({ length: 280 }, (_, i) => 400 + 10 * i), bg = x => 300 + 0.08 * (x - 400) - 1.5e-5 * (x - 1500) ** 2;
  const y = Float64Array.from(nu, x => bg(x) + 800 * lor(x, 1000, 20) + 500 * lor(x, 1600, 40) + 900 * lor(x, 2900, 30));
  const base = baselineFitter(nu, 3)(y);
  let worst = 0; nu.forEach((x, i) => { if (Math.min(...[1000, 1600, 2900].map(c => Math.abs(x - c))) > 150) worst = Math.max(worst, Math.abs(base[i] - bg(x))); });
  assert.ok(worst < 25, `background off by ${worst} counts away from the bands (bands 500-900)`);
  assert.equal(baselineFitter(nu, 0)(y).every(v => v === 0), true);
});

test("a line scan: three materials found, four layers, boundaries and thicknesses to a fraction of the step", () => {
  const r = run(lineScan(STACK, -4, 24, 0.25));
  assert.equal(r.comps.k, 3, `materials: steps ${JSON.stringify(r.comps.steps)}`);
  const names = namesOf(r), M = matchRefs(r.comps, r.prep, REFS, 0);
  M.forEach((m, a) => assert.ok(m.hqi[m.best] > 95, `material ${a} matches ${names[a]} at HQI ${m.hqi[m.best]}`));
  assert.deepEqual(r.res.stack.map(a => names[a]), ["A", "C", "B", "A"]);
  assert.equal(r.res.nCons, 1);
  r.res.layers.forEach((L, i) => {
    const [, a, b] = STACK[i];
    near(L.from, a, 0.12, `layer ${i + 1} from`); near(L.to, b, 0.12, `layer ${i + 1} to`); near(L.t.mean, b - a, 0.15, `layer ${i + 1} thickness`);
    assert.equal(L.open, false); assert.equal(L.thin, false);
  });
  near(r.res.total.mean, 20, 0.2, "total");
  // the spot's blur, recovered from the steps: 16-84 % width = 2 sigma = 0.8 µm
  for (const L of r.res.layers) { near(L.wFrom, 0.8, 0.25, "width at from"); near(L.wTo, 0.8, 0.25, "width at to"); }
  // the brightest material does not push the boundaries towards the dimmer ones: the plateaus differ 4x
  const pl = Array.from(r.res.plateau), byName = Object.fromEntries(names.map((n, a) => [n, pl[a]]));
  assert.ok(byName.A / byName.C > 3, "plateaus scale each material to its own pure level");
});

test("a material the decomposition cannot tell apart is not invented: the stop rule keeps to what is there", () => {
  const r = run(lineScan([["A", 0, 10], ["B", 10, 20]], -4, 24, 0.25));
  assert.equal(r.comps.k, 2, JSON.stringify(r.comps.steps));
  assert.equal(r.res.layers.length, 2);
  const forced = run(lineScan([["A", 0, 10], ["B", 10, 20]], -4, 24, 0.25), { k: 2 });
  assert.equal(forced.comps.k, 2);
});

test("unmixing against reference spectra names the materials and finds the same layers", () => {
  const r = run(lineScan(STACK, -4, 24, 0.25), { useRefs: true }, REFS);
  assert.equal(r.comps.mode, "refs"); assert.equal(r.comps.k, 3);
  assert.deepEqual(r.res.stack.map(a => REFS[a].name), ["A", "C", "B", "A"]);
  r.res.layers.forEach((L, i) => near(L.t.mean, STACK[i][2] - STACK[i][1], 0.15, `layer ${i + 1}`));
  assert.throws(() => decompose(r.prep, { ...r.S, useRefs: true }, REFS.slice(0, 1)), /at least two/);
  const short = { name: "short", wn: WN.slice(0, 700), y: SPEC.A.slice(0, 700) };
  assert.throws(() => decompose(r.prep, { ...r.S, useRefs: true }, [REFS[0], short]), /covers only/);
});

test("a scan that starts or ends inside a layer reports it open; a gap between layers is a layer of no material", () => {
  const r = run(lineScan(STACK, 2, 17, 0.25));
  const L = r.res.layers;
  assert.equal(L.length, 4); assert.equal(L[0].open, true); assert.equal(L[3].open, true); assert.equal(L[1].open, false);
  near(L[0].t.mean, 4, 0.15, "first layer runs from the scan's start"); near(L[1].t.mean, 2.4, 0.15);
  const gap = run(lineScan([["A", 0, 6], ["B", 7.5, 13]], -4, 17, 0.25));
  assert.deepEqual(gap.res.stack.map(a => (a < 0 ? "none" : namesOf(gap)[a])), ["A", "none", "B"]);
  near(gap.res.layers[1].t.mean, 1.5, 0.15, "gap");
});

test("a map with tilted layers: the axis is found, the tilt measured and taken out of the thickness", () => {
  const R = rng(7), nx = 64, ny = 24, dx = 0.5, slope = 0.1, xs = Float64Array.from({ length: nx }, (_, i) => -6 + i * dx), ys = Float64Array.from({ length: ny }, (_, i) => i);
  const cube = new Float32Array(nx * ny * WN.length);
  for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) {
    const off = slope * ys[r], st = STACK.map(([m, a, b]) => [m, a + off, b + off]);
    cube.set(spectrum(xs[c], st, R, { noise: 10 }), (r * nx + c) * WN.length);
  }
  const data = { cube, geom: { nx, ny, x: xs, y: ys } }, r = run(data);
  assert.equal(r.res.axis, "x");
  assert.ok(r.res.share.x > 0.6 && r.res.share.y < 0.1, JSON.stringify(r.res.share));
  near(r.res.tilt.deg, Math.atan(slope) * 180 / Math.PI, 0.6, "tilt");
  assert.equal(r.res.nCons, ny);
  const cos = 1 / Math.sqrt(1 + slope * slope);
  r.res.layers.forEach((L, i) => {
    const app = STACK[i][2] - STACK[i][1];                 // along x; the true thickness is app × cos
    near(L.t.mean, app * cos, 0.12, `layer ${i + 1} corrected`);
    assert.ok(L.t.sd < 0.2, `layer ${i + 1} SD ${L.t.sd}`);
  });
  // the mean profile lines the lines up by the tilt: the thin layer still reaches its pure level in it
  const thinPeak = Math.max(...Array.from(r.res.profile.u[r.res.stack[1]]).filter(Number.isFinite));
  assert.ok(thinPeak > 0.95, `thin layer's peak in the mean profile ${thinPeak}`);
  const raw = findLayers(r.prep, r.comps, data.geom, { ...r.S, tilt: false });
  raw.layers.forEach((L, i) => near(L.t.mean, STACK[i][2] - STACK[i][1], 0.12, `layer ${i + 1} along x`));
  const lab = r.res.label, layerOf = r.res.layerOf;
  assert.equal(lab.length, nx * ny); assert.ok(layerOf.filter(v => v >= 0).length > 0.6 * nx * ny);
  const sp = layerSpectra(cube, WN.length, nx * ny, layerOf, r.res.layers.length);
  assert.ok(sp.every(q => q.n > 0 && q.y.length === WN.length));
  const amt = amountMap(r.prep, r.comps, r.res.plateau, r.res.stack[1]);
  assert.ok(Math.max(...amt.filter(Number.isFinite)) > 0.8);
});

test("a confocal depth profile: the depth scale multiplies every position and thickness", () => {
  const data = lineScan(STACK, -4, 24, 0.25), a = run(data), b = run(data, { kind: "depth", depthScale: 1.5 });
  b.res.layers.forEach((L, i) => { near(L.t.mean, 1.5 * a.res.layers[i].t.mean, 1e-9); near(L.from, 1.5 * a.res.layers[i].from, 1e-9); });
  const c = run(data, { kind: "section", depthScale: 1.5 });
  c.res.layers.forEach((L, i) => near(L.t.mean, a.res.layers[i].t.mean, 1e-9, "a cross-section ignores the depth scale"));
});

test("thickness limits: pass, fail, and a layer cut by the scan's edge that can pass a minimum only", () => {
  const r = run(lineScan(STACK, -4, 24, 0.25));
  let Q = thicknessTest(r.res, [null, { lo: 2, use_lo: true }]);
  assert.equal(Q.rows[1].level, "good"); assert.equal(Q.level, "good"); assert.equal(Q.set, 1);
  Q = thicknessTest(r.res, [null, { lo: 3, use_lo: true }]);
  assert.equal(Q.rows[1].level, "bad"); assert.equal(Q.rows[1].fail, 1);
  const open = run(lineScan(STACK, 2, 17, 0.25)).res;
  Q = thicknessTest(open, [{ lo: 3, use_lo: true }]);
  assert.equal(Q.rows[0].pass, 1, "4 µm seen, at least 3 required: passes");
  Q = thicknessTest(open, [{ lo: 3, use_lo: true, hi: 10, use_hi: true }]);
  assert.equal(Q.rows[0].undet, 1); assert.equal(Q.rows[0].n, 0); assert.equal(Q.level, null);
});

test("purest stops at the noise and the range must hold enough channels", () => {
  const data = lineScan([["A", 0, 30]], 1, 25, 0.5), S = structuredClone(LAYER_DEFAULTS);
  const prep = prepare(data.cube, WN.length, WN, data.geom, S), sp = purest(prep, S);
  assert.equal(sp.picks.length, 1, JSON.stringify(sp.steps));
  assert.throws(() => prepare(data.cube, WN.length, WN, data.geom, { ...S, range: [1000, 1020] }), /widen the range/);
});

test("cosmic rays are taken out against the neighbouring spectra, and a thin layer's sharp bands are not", () => {
  const data = lineScan(STACK, -4, 24, 0.25), K = WN.length, R = rng(3), hit = [];
  for (let s = 0; s < 6; s++) {
    const p = 5 + Math.floor(R() * 100), c = 300 + Math.floor(R() * 900);
    data.cube[p * K + c] += 3000; data.cube[p * K + c + 1] += 1500; hit.push(p);
  }
  const r = run(data);
  assert.equal(r.prep.spikes, 6, "every cosmic ray, and nothing else");
  assert.equal(r.comps.k, 3, JSON.stringify(r.comps.steps));
  r.res.layers.forEach((L, i) => near(L.t.mean, STACK[i][2] - STACK[i][1], 0.15, `layer ${i + 1}`));
  // a layer as thin as the spot is wide stands out from its neighbours over its whole band width: left alone
  const thin = run(lineScan([["A", 0, 6], ["C", 6, 6.6], ["A", 6.6, 12]], -2, 14, 0.25));
  assert.equal(thin.prep.spikes, 0);
});

test("a layer thinner than the spot is flagged against the resolution of the stack's other steps", () => {
  const r = run(lineScan([["A", 0, 6], ["C", 6, 6.5], ["B", 6.5, 12]], -2, 14, 0.25), { k: 3 });
  const names = namesOf(r), i = r.res.stack.findIndex(a => names[a] === "C");
  assert.ok(i === 1, `C found as the middle layer: ${r.res.stack.map(a => names[a])}`);
  near(r.res.resolution, 0.8, 0.25, "resolution: 2 sigma of the spot");
  assert.equal(r.res.layers[i].thin, true);
  assert.ok(r.res.layers[i].t.mean > 0.5, `a sub-resolution layer reads too thick: ${r.res.layers[i].t.mean}`);
  assert.equal(r.res.layers[0].thin, false); assert.equal(r.res.layers[2].thin, false);
});
