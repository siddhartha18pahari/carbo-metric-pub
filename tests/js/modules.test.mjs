// Graphene, Nanotubes, Silicon, axis calibration and Identify:  node --test tests/js/modules.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { compile, fitOne, cut } from "../../web/js/peakfit.js";
import { GRAPHENE_DEFAULTS, models as grModels, readOne as grRead, defectType } from "../../web/js/graphene.js";
import { CNT_DEFAULTS, models as cntModels, rbmPeaks, readOne as cntRead1, gMax, preferOne } from "../../web/js/cnt.js";
import { STANDARDS, fitBands, fitCorrection, correctAxis, referenceBands } from "../../web/js/calibrate.js";
import { applyOps } from "../../web/js/spectra.js";
import { SI_DEFAULTS, model as siModel, readOne as siRead, calCheck, controlLimits } from "../../web/js/silicon.js";
import { refVectors, hits, mixture, searchAll } from "../../web/js/identify.js";
import { prepare } from "../../web/js/layers.js";

const close = (a, b, tol, what = "") => assert.ok(Math.abs(a - b) <= tol, `${what} ${a} vs ${b}`);
function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function gauss(r) { return () => Math.sqrt(-2 * Math.log(r() + 1e-300)) * Math.cos(2 * Math.PI * r()); }
const lor = (v, c, w) => 1 / (1 + (2 * (v - c) / w) ** 2);
// bands as the model reads them: height, error, position, error, FWHM, area
const bandsOf = r => r.D.vals.map((v, j) => ({ h: v[0], sh: r.D.ses[j][0], c: v[1], sc: r.D.ses[j][1], w: v[2], a: v[3] }));

test("graphene: a strained, doped, defective monolayer read back", () => {
  const s = { ...GRAPHENE_DEFAULTS, laser: 514.5 }, eps = 0.1, dop = 2;
  // Lee et al.'s decomposition run forwards: strain moves G by −69.1 per % and 2D by 2.2 × that
  const dgS = s.sensG * eps, wG = s.g0 + dgS + dop, w2D = s.d0 + s.slopeT * dgS + s.slopeH * dop;
  const g = gauss(rng(4)), wn = Float64Array.from({ length: 1900 }, (_, k) => 1100 + k);
  const y = Float64Array.from(wn, v => 100 + 0.02 * (v - 1100) + 400 * lor(v, 1345, 22) + 800 * lor(v, wG, 14) + 400 / 7 * lor(v, 1620, 9) + 2000 * lor(v, w2D, 28) + 4 * g());
  const M = grModels(s), r1 = fitOne(compile(M.m1, wn), cut(compile(M.m1, wn), y)), r2 = fitOne(compile(M.m2, wn), cut(compile(M.m2, wn), y));
  const out = grRead(bandsOf(r1), bandsOf(r2)[0], Math.max(r1.D.rn, r2.D.rn), s);
  assert.equal(out.layer, 0);
  close(out.idg, 0.5, 0.02, "I(D)/I(G)"); close(out.i2dg, 2.5, 0.05, "I(2D)/I(G)"); close(out.w2d, 28, 0.5, "FWHM(2D)");
  close(out.ld, Math.sqrt(1.8e-9 * 514.5 ** 4 / 0.5), 0.3, "L_D"); close(out.iddp, 7, 0.6, "I(D)/I(D′)");
  assert.equal(defectType(out.iddp).name, "vacancies");
  close(out.eps, eps, 0.005, "strain"); close(out.dop, dop, 0.3, "doping shift");
});

test("graphene: a wide 2D is not a monolayer; no G, no graphene", () => {
  const s = GRAPHENE_DEFAULTS, f = (h, c, w) => ({ h, sh: 1, c, w, a: h * w });
  assert.equal(grRead([f(0, 1350, 20), f(500, 1582, 15), f(0, 1620, 9)], f(600, 2700, 55), 1, s).layer, 1);
  assert.equal(grRead([f(0, 1350, 20), f(500, 1582, 15), f(0, 1620, 9)], f(600, 2700, 75), 1, s).layer, 2);
  assert.equal(grRead([f(0, 1350, 20), f(0, 1582, 15), f(0, 1620, 9)], f(600, 2700, 30), 1, s).layer, 3);
  // a heavily defective sample: L_D below 10 nm is outside the calibration and not given
  assert.ok(Number.isNaN(grRead([f(2000, 1350, 20), f(500, 1582, 15), f(0, 1620, 9)], f(600, 2700, 30), 1, { ...s, laser: 514.5 }).ld));
});

test("silicon: stress from the shift, the crystalline fraction of a film, the calibration check", () => {
  const g = gauss(rng(8)), wn = Float64Array.from({ length: 600 }, (_, k) => 400 + 0.4 * k);
  const s = { ...SI_DEFAULTS }, y = Float64Array.from(wn, v => 50 + 3000 * lor(v, 521.7, 4) + 5 * g());
  const C = compile(siModel(s), wn), r = fitOne(C, cut(C, y)), out = siRead(bandsOf(r), r.D.rn, s);
  close(out.shift, 1, 0.01, "shift"); close(out.stress, -434, 5, "stress"); close(out.fwhm, 4, 0.05, "FWHM");
  const film = { ...s, model: "film" }, A = (h, w) => h * w * Math.sqrt(Math.PI / (4 * Math.LN2));
  const yf = Float64Array.from(wn, v => 30 + 1000 * lor(v, 520.5, 5) + 150 * Math.exp(-4 * Math.LN2 * ((v - 508) / 14) ** 2) + 200 * Math.exp(-4 * Math.LN2 * ((v - 480) / 60) ** 2) + 3 * g());
  const Cf = compile(siModel(film), wn), rf = fitOne(Cf, cut(Cf, yf)), of = siRead(bandsOf(rf), rf.D.rn, film);
  const ac = 1000 * 5 * Math.PI / 2, agb = A(150, 14), aa = A(200, 60);
  close(of.xc, 100 * (ac + agb) / (ac + agb + 0.8 * aa), 2, "crystalline fraction");
  assert.ok(calCheck(520.9, 3.5, s).pass); assert.ok(!calCheck(521.4, 3.5, s).pass); assert.ok(!calCheck(520.7, 8, s).pass);
  const L = controlLimits([520.7, 520.6, 520.8, 520.7, 520.65, 520.75, 522]  , 6);
  assert.deepEqual(L.out, [6]);
});

test("identify: the right reference, its HQI, and a mixture of two", () => {
  const wn = Float64Array.from({ length: 900 }, (_, k) => 200 + 2 * k), g = gauss(rng(2));
  const bands = [[[465, 8], [1085, 10]], [[1001, 5], [1602, 8], [3054, 12]], [[1350, 60], [1590, 40]], [[520, 5], [950, 30]]];
  const spec = (b, h = 1000) => Float64Array.from(wn, v => b.reduce((t, [c, w]) => t + h * lor(v, c, w), 0));
  const refs = bands.map((b, i) => ({ name: `ref ${i}`, wn, y: spec(b) }));
  // five spectra: ref 1, ref 2, a 70/30 mix of refs 0 and 3, and two more of ref 1, over a background
  const rows = [spec(bands[1]), spec(bands[2]), Float64Array.from(spec(bands[0]), (v, k) => 0.7 * v + 0.3 * spec(bands[3])[k]), spec(bands[1], 700), spec(bands[1], 500)];
  const cube = new Float32Array(5 * wn.length);
  rows.forEach((y, p) => { for (let k = 0; k < wn.length; k++) cube[p * wn.length + k] = 80 + 0.05 * wn[k] + y[k] + 3 * g(); });
  const prep = prepare(cube, wn.length, wn, { nx: 5, ny: 1 }, { range: [200, 1990], baseline: 2 });
  const R = refVectors(refs, prep), res = searchAll(prep, R);
  assert.deepEqual(Array.from(res.best), [1, 2, 0, 1, 1]);
  assert.ok(res.hqi[0] > 95 && res.hqi[1] > 95, `HQI ${res.hqi[0]} ${res.hqi[1]}`);
  const x = prep.V.subarray(2 * prep.nb, 3 * prep.nb), top = hits(x, R).map(h => h.r);
  const mix = mixture(x, R, top.slice(0, 4));
  assert.deepEqual([mix.a, mix.b].sort(), [0, 3]);
  assert.ok(mix.hqi > res.hqi[2] + 3, `mixture ${mix.hqi} vs single ${res.hqi[2]}`);
  // equal band heights: the area shares follow the bands' widths
  const area = b => b.reduce((t, [, w]) => t + w, 0), want = 0.7 * area(bands[0]) / (0.7 * area(bands[0]) + 0.3 * area(bands[3]));
  close(mix.a === 0 ? mix.share : 1 - mix.share, want, 0.06, "share of ref 0");
  // the derivative search agrees
  assert.deepEqual(Array.from(searchAll(prep, refVectors(refs, prep, true), true).best), [1, 2, 0, 1, 1]);
});

// a BWF band as peakfit.js draws it (height h at c, width parameter w, coupling q)
const bwf = (v, c, w, q) => { const x = 2 * (v - c) / w + q, a = 1 + q * x; return a * a / ((1 + x * x) * (1 + q * q)); };
const withQ = r => bandsOf(r).map((b, j) => ({ ...b, q: r.D.vals[j][4], sq: r.D.ses[j][4] }));
function cntRead(wn, y, s) {
  const M = cntModels(s, rbmPeaks(wn, y, s)), fit = m => { const C = compile(m, wn); return { r: fitOne(C, cut(C, y)), N: C.N }; };
  const G = fit(M.mG), G1 = fit(M.mG1), rR = M.mR ? fit(M.mR).r : null, r2 = fit(M.m2).r;
  const one = preferOne(G.r.D.rn, G1.r.D.rn, G.N), rnG = one ? G1.r.D.rn : G.r.D.rn;
  return { M, one, out: cntRead1(rR ? withQ(rR) : null, withQ(G.r), withQ(r2)[0], Math.max(rnG, r2.D.rn, rR ? rR.D.rn : 0), s, one ? withQ(G1.r) : null) };
}

test("nanotubes: RBM diameters, a metallic G⁻ and the diameter from the splitting read back", () => {
  const s = { ...CNT_DEFAULTS, laser: 532 }, g = gauss(rng(11)), wn = Float64Array.from({ length: 2921 }, (_, k) => 80 + k);
  const gm = { h: 400, c: 1550, w: 40, q: -0.25 }, gp = { h: 1500, c: 1591, w: 12 };
  const y = Float64Array.from(wn, v => 80 + 0.01 * v + 300 * lor(v, 160, 6) + 500 * lor(v, 260, 5) + 150 * lor(v, 1340, 30)
    + gm.h * bwf(v, gm.c, gm.w, gm.q) + gp.h * lor(v, gp.c, gp.w) + 600 * lor(v, 2680, 50) + 4 * g());
  const { M, one, out } = cntRead(wn, y, s);
  assert.equal(M.mR.peaks.length, 2, "two RBM bands found");
  assert.equal(one, false, "the split G model is kept");
  assert.equal(out.nrbm, 2);
  close(out.prbm, 260, 0.1, "strongest RBM"); close(out.drbm, 248 / 260, 0.002, "its diameter");
  close(out.dmean, (500 * 248 / 260 + 300 * 248 / 160) / 800, 0.01, "mean diameter");
  assert.equal(out.type, 0, "metallic");
  close(out.q, -0.25, 0.03, "1/Q"); close(out.pgm, 1550, 0.5, "Pos(G⁻)"); close(out.pgp, 1591, 0.1, "Pos(G⁺)");
  close(out.dg, Math.sqrt(79.5 / 41), 0.02, "diameter from the splitting");
  const fw = gm.w * (1 + gm.q ** 2) / (1 - gm.q ** 2), iG = gMax({ ...gm, w: fw }, gp).h;
  close(out.igd, iG / 150, 0.05 * iG / 150, "I(G)/I(D)"); close(out.p2d, 2680, 0.3, "Pos(2D)");
  // the laser filter's steep edge under the RBMs: both found, both read
  const yt = Float64Array.from(wn, (v, k) => y[k] + 5000 * Math.exp(-(v - 80) / 60)), t = cntRead(wn, yt, s).out;
  assert.equal(t.nrbm, 2); close(t.prbm, 260, 0.3, "strongest RBM on a tail");
  // the RBM relation is applied at reading: bundles give larger diameters for the same bands
  const b = { ...s, relation: "bundles", A: 234, B: 10 }, ob = cntRead1([{ h: 500, sh: 1, c: 260, w: 5 }], [], null, 1, b);
  close(ob.drbm, 234 / 250, 1e-12, "bundle relation");
});

test("nanotubes: a Lorentzian G⁻ is semiconducting; one broad G band without RBMs is neither", () => {
  const s = CNT_DEFAULTS, g = gauss(rng(12)), wn = Float64Array.from({ length: 1700 }, (_, k) => 1000 + k);
  const y = Float64Array.from(wn, v => 60 + 300 * lor(v, 1570, 15) + 1500 * lor(v, 1591, 12) + 100 * lor(v, 1340, 30) + 400 * lor(v, 2680, 50) + 4 * g());
  const { M, one, out } = cntRead(wn, y, s);
  assert.equal(M.mR, null, "no RBM window: no RBM model");
  assert.equal(one, false, "the split G model is kept");
  assert.equal(out.type, 1, "semiconducting");
  close(out.dg, Math.sqrt(47.7 / 21), 0.03, "diameter from the splitting");
  // multi-wall tubes: one broad G band, a strong D
  const y2 = Float64Array.from(wn, v => 60 + 800 * lor(v, 1582, 35) + 900 * lor(v, 1345, 45) + 300 * lor(v, 1615, 20) + 300 * lor(v, 2690, 70) + 4 * g());
  const r2 = cntRead(wn, y2, s), o2 = r2.out;
  assert.equal(r2.one, true, "the one-G model is kept");
  assert.equal(o2.type, 2, "one G band");
  close(o2.pgp, 1582, 0.3, "Pos(G)"); close(o2.wgp, 35, 0.5, "FWHM(G)");
  assert.ok(Number.isNaN(o2.dg));
  assert.ok(o2.idg > 0.9 && o2.idg < 1.3, `I(D)/I(G) ${o2.idg}`);
  // where the RBM window shows no tube the caller reads every G as one band, whatever its components
  const f = (h, c, w, q = 0) => ({ h, sh: 1, c, w, q, sq: 0.01 }), G = [f(100, 1340, 30), f(450, 1550, 45, -0.25), f(1500, 1591, 12), f(0, 1620, 10)];
  assert.equal(cntRead1(null, G, null, 1, s).type, 0); assert.equal(cntRead1(null, G, null, 1, s, null, false).type, 2);
});

test("axis calibration: polystyrene on a shifted, stretched axis corrected back to ASTM E1840", () => {
  const g = gauss(rng(21)), refs = STANDARDS.ps.bands, wn = Float64Array.from({ length: 3200 }, (_, k) => 200 + k);
  // the instrument reads each band at m, where true = m + 1.5 + 0.9 (m − 1500) / 1000
  const meas = refs.map(r => (r - 1.5 + 0.9 * 1.5) / 1.0009), heights = [300, 120, 2000, 700, 500, 600, 300, 800, 900, 1500, 1800];
  const y = Float64Array.from(wn, v => 100 + 0.05 * v + meas.reduce((t, m, i) => t + heights[i] * lor(v, m, 7), 0) + 5 * g());
  const bands = fitBands(wn, y, refs), used = bands.filter(b => b.used);
  assert.equal(used.length, refs.length, `all eleven bands: ${bands.filter(b => !b.used).map(b => `${b.ref} ${b.why}`)}`);
  const fit = fitCorrection(bands, "auto");
  assert.equal(fit.order, 1);
  const fixed = correctAxis(Float64Array.from(used, b => b.pos), fit.coef);
  used.forEach((b, i) => close(fixed[i], b.ref, 0.15, `band ${b.ref}`));
  close(correctAxis([1500], fit.coef)[0] - 1500, 1.5, 0.05, "shift at 1500");
  assert.ok(fit.rms < 0.1, `rms ${fit.rms}`);
  // the Toolbox step moves the axis by the same correction
  const r = applyOps(wn, y, [{ op: "axis", args: fit.coef }]);
  close(r.nu[1300] - wn[1300], correctAxis([wn[1300]], fit.coef)[0] - wn[1300], 1e-12, "the step");
  // silicon: one band, an offset; a band off the spectrum is reported, not used
  const ys = Float64Array.from(wn, v => 50 + 3000 * lor(v, 519.9, 4) + 4 * g());
  const bs = fitBands(wn, ys, [520.7, 3500]), fs = fitCorrection(bs);
  assert.equal(fs.order, 0); close(fs.coef.shift, 0.8, 0.02, "silicon offset");
  assert.equal(bs.find(b => b.ref === 3500).why, "outside the spectrum");
  assert.deepEqual(referenceBands("custom", "3054.3; 1001.4, 1602.3"), [1001.4, 1602.3, 3054.3]);
});
