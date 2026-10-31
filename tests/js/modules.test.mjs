// Graphene, Silicon and Identify:  node --test tests/js/modules.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { compile, fitOne, cut } from "../../web/js/peakfit.js";
import { GRAPHENE_DEFAULTS, models as grModels, readOne as grRead, defectType } from "../../web/js/graphene.js";
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
