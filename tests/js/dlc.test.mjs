// node --test tests/js/dlc.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { DLC_DEFAULTS, DLC_PARAMS, SE_KEYS, bwf, bwfFwhm, bwfArea, lorentzArea, prepare, makeFitter, fitOne, derive, fitMean,
         fitSpectra, components, qcTest, paramStats, homogeneity } from "../../web/js/dlc.js";
import { noiseSigma } from "../../web/js/d3g-calibration.js";
import { rng } from "../../web/js/pf-analysis.js";

const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ""} ${a} vs ${b}`);
const gauss = R => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
const WN = Float64Array.from({ length: 801 }, (_, i) => 800 + 1.5 * i);
const S = structuredClone(DLC_DEFAULTS);
const at = key => DLC_PARAMS.findIndex(p => p.key === key);

// an a-C:H spectrum: Lorentzian D, BWF G, a background rising with the shift (photoluminescence)
const TRUE = { hd: 600, cd: 1370, wd: 280, hg: 1000, cg: 1545, wg: 160, q: -0.18, b: 300, m: 0.15 };
function dlcSpectrum(o = {}, sd = 0, R = null) {
  const T = { ...TRUE, ...o };
  return Float64Array.from(WN, x => T.b + T.m * (x - 1400) + T.hd / (1 + (2 * (x - T.cd) / T.wd) ** 2) + bwf(x, T.hg, T.cg, T.wg, T.q)
    + (sd && R ? sd * gauss(R) : 0));
}
const hOf = (m, hg) => S.hA + S.hB * Math.log10(m / hg * 1e4);

test("the BWF peaks at its height, its FWHM and area have closed forms, and q = 0 is a Lorentzian", () => {
  const h = 1000, c = 1545, w = 160, q = -0.2;
  let best = -Infinity, at0 = 0;
  for (let x = 1400; x <= 1700; x += 0.001) { const v = bwf(x, h, c, w, q); if (v > best) { best = v; at0 = x; } }
  near(best, h, 1e-6, "height"); near(at0, c, 2e-3, "maximum at c");
  // half-maximum crossings, found numerically
  const cross = (dir) => { let x = c; while (bwf(x, h, c, w, q) > h / 2) x += dir * 0.001; return x; };
  near(cross(1) - cross(-1), bwfFwhm(w, q), 0.01, "FWHM");
  let s = 0; const a = 900, b = 1900, dx = 0.01;
  for (let x = a + dx / 2; x < b; x += dx) s += bwf(x, h, c, w, q) * dx;
  near(bwfArea(h, c, w, q, a, b), s, 1e-3 * s, "area");
  for (const x of [1300, 1500, 1545, 1600]) near(bwf(x, h, c, w, 0), h / (1 + (2 * (x - c) / w) ** 2), 1e-9);
  near(bwfFwhm(w, 0), w, 1e-12);
  let L = 0; for (let x = a + dx / 2; x < b; x += dx) L += h / (1 + (2 * (x - c) / w) ** 2) * dx;
  near(lorentzArea(h, c, w, a, b), L, 1e-3 * L);
});

test("the analytic Jacobian matches finite differences", () => {
  const F = makeFitter(WN, S), th = Float64Array.of(310, 85, 600, 1370, 280, 1000, 1545, 160, -0.18);
  const f0 = new Float64Array(F.N), f1 = new Float64Array(F.N), f2 = new Float64Array(F.N);
  for (let k = 0; k < 9; k++) {
    const h = Math.max(1e-6, Math.abs(th[k]) * 1e-6), a = Float64Array.from(th), b = Float64Array.from(th);
    a[k] += h; b[k] -= h; F.evaluate(a, f1, false); F.evaluate(b, f2, false);
    F.evaluate(th, f0, true);
    let worst = 0, scale = 0;
    for (let n = 0; n < F.N; n += 7) {
      const fd = (f1[n] - f2[n]) / (2 * h);
      scale = Math.max(scale, Math.abs(fd));
      worst = Math.max(worst, Math.abs(fd - F.J[n * 9 + k]));
    }
    assert.ok(worst <= 1e-6 * Math.max(scale, 1), `parameter ${k}: ${worst} against ${scale}`);
  }
});

test("a noiseless a-C:H spectrum is recovered from its own start", () => {
  const F = makeFitter(WN, S), y = dlcSpectrum(), fit = fitOne(F, y, null), D = derive(F, fit, 1, S);
  near(D.v[at("idig")], TRUE.hd / TRUE.hg, 1e-6, "I(D)/I(G)");
  near(D.v[at("gpos")], TRUE.cg, 1e-4, "G position");
  near(D.v[at("gfwhm")], bwfFwhm(TRUE.wg, TRUE.q), 1e-3, "G FWHM");
  near(D.v[at("dpos")], TRUE.cd, 1e-3); near(D.v[at("dfwhm")], TRUE.wd, 1e-3);
  near(D.v[at("q")], TRUE.q, 1e-6);
  near(D.v[at("slope")], TRUE.m, 1e-6, "slope m");
  near(D.v[at("mig")], TRUE.m / TRUE.hg * 1e4, 1e-5, "m/I(G) in µm");
  near(D.v[at("h")], hOf(TRUE.m, TRUE.hg), 1e-4, "hydrogen");
  near(D.v[at("la")], 0.1 * Math.sqrt(0.6 / 0.0055), 1e-5, "La");
  assert.equal(D.flags, 0);
  const C = components(F, fit.th);
  near(Math.max(...C.g), TRUE.hg, 0.5);
});

test("noisy fits: estimates unbiased and the errors match the scatter of 200 replicates", () => {
  const R = rng(11), F = makeFitter(WN, S), keys = ["idig", "gpos", "gfwhm", "h"], got = Object.fromEntries(keys.map(k => [k, []])), pred = Object.fromEntries(keys.map(k => [k, []]));
  const shapes = fitOne(F, dlcSpectrum(), null).th;
  for (let r = 0; r < 200; r++) {
    const y = dlcSpectrum({}, 25, R), fit = fitOne(F, y, shapes), D = derive(F, fit, noiseSigma(y), S);
    for (const k of keys) { got[k].push(D.v[at(k)]); pred[k].push(D.s[SE_KEYS.indexOf(k)]); }
  }
  const mean = a => a.reduce((t, v) => t + v, 0) / a.length, sd = a => Math.sqrt(a.reduce((t, v) => t + (v - mean(a)) ** 2, 0) / (a.length - 1));
  const truth = { idig: 0.6, gpos: TRUE.cg, gfwhm: bwfFwhm(TRUE.wg, TRUE.q), h: hOf(TRUE.m, TRUE.hg) };
  for (const k of keys) {
    near(mean(got[k]), truth[k], 3 * sd(got[k]) / Math.sqrt(200) + 1e-9, `${k} bias`);
    const ratio = mean(pred[k]) / sd(got[k]);
    assert.ok(ratio > 0.8 && ratio < 1.25, `${k}: predicted SE / scatter = ${ratio}`);
  }
});

// a 12 x 10 map with G shifting across it, two unmeasured points and one without carbon
function map({ shift = 0, sd = 20, seed = 3 } = {}) {
  const R = rng(seed), nx = 12, ny = 10, P = nx * ny, K = WN.length, cube = new Float32Array(P * K);
  for (let p = 0; p < P; p++) {
    const j = p % nx, y = dlcSpectrum({ cg: TRUE.cg + shift * (j / (nx - 1) - 0.5) }, sd, R);
    cube.set(y, p * K);
  }
  cube.fill(NaN, 5 * K, 6 * K);                 // a map stopped early
  cube.fill(0, 7 * K, 8 * K);                   // an unmeasured particle reads zeros
  for (let c = 0; c < K; c++) cube[9 * K + c] = 300 + 0.15 * (WN[c] - 1400) + sd * gauss(R);   // background only
  return { cube, nx, ny, P, K };
}
function fitMap(m, settings = S) {
  const prep = prepare(m.cube, m.K, WN, m.P, settings), mean = fitMean(prep, settings);
  const r = fitSpectra({ nu: prep.nu, N: prep.N, Y: prep.Y, settings, shapes: mean.fit.th }, Array.from({ length: m.P }, (_, p) => p));
  for (let p = 0; p < m.P; p++) if (!prep.measured[p]) r.status[p] = 0;
  const col = key => { const j = at(key), out = new Float64Array(m.P); for (let p = 0; p < m.P; p++) out[p] = r.cols[p * DLC_PARAMS.length + j]; return out; };
  const scol = key => { const j = SE_KEYS.indexOf(key), out = new Float64Array(m.P); for (let p = 0; p < m.P; p++) out[p] = r.ses[p * SE_KEYS.length + j]; return out; };
  return { prep, mean, r, col, scol };
}

test("a map: status per spectrum, the parameters per pixel", () => {
  const m = map({ shift: 20 }), { prep, r, col } = fitMap(m);
  assert.equal(prep.measured[5], 0); assert.equal(prep.measured[7], 0); assert.equal(r.status[5], 0); assert.equal(r.status[7], 0);
  assert.equal(r.status[9], 2, "a spectrum without a G band is below the signal floor");
  const ok = Uint8Array.from(r.status, s => (s === 1 ? 1 : 0));
  assert.equal(ok.reduce((t, v) => t + v, 0), m.P - 3);
  const g = col("gpos");
  near(g[0], TRUE.cg - 10, 2, "G at the left edge"); near(g[11], TRUE.cg + 10, 2, "G at the right edge");
  const st = paramStats(col("idig"), ok, null);
  near(st.median, 0.6, 0.01);
});

test("the quality test: pass and fail per spectrum, the share in spec with its interval, Ppk", () => {
  const m = map({ shift: 20 }), { r, col } = fitMap(m);
  const limits = [{ key: "gpos", lo: 1540, hi: 1560, use_lo: true, use_hi: true }, { key: "idig", lo: 0, hi: 0.8, use_lo: false, use_hi: true }];
  const Q = qcTest(col, r.status, limits), g = col("gpos");
  let expect = 0; for (let p = 0; p < m.P; p++) if (r.status[p] === 1 && g[p] >= 1540 && g[p] <= 1560) expect++;
  assert.equal(Q.nPass, expect);
  assert.equal(Q.nPass + Q.nFail, m.P - 3);
  assert.ok(Q.ci[0] <= Q.share && Q.share <= Q.ci[1]);
  for (let p = 0; p < m.P; p++) if (r.status[p] === 1) assert.equal(Q.pass[p], g[p] >= 1540 && g[p] <= 1560 ? 1 : 2);
  assert.equal(Q.byKey.idig, 0, "I(D)/I(G) 0.6 passes everywhere");
  // no limits in use: nothing judged
  assert.equal(qcTest(col, r.status, [{ key: "h", lo: 1, hi: 2, use_lo: false, use_hi: false }]).n, 0);
  // Ppk against a two-sided limit
  const ok = Uint8Array.from(r.status, s => (s === 1 ? 1 : 0)), st = paramStats(g, ok, limits[0]);
  near(st.ppk, Math.min(1560 - st.mean, st.mean - 1540) / (3 * st.sd), 1e-12);
});

test("homogeneity: an even map reads as noise, a gradient as real variation", () => {
  for (const [shift, even] of [[0, true], [30, false]]) {
    const m = map({ shift, seed: 7 }), { r, col, scol } = fitMap(m), ok = Uint8Array.from(r.status, s => (s === 1 ? 1 : 0));
    const H = homogeneity(col("gpos"), scol("gpos"), ok, { grid: { nx: m.nx, ny: m.ny } });
    if (even) { assert.ok(H.I2 < 0.3, `even map I² ${H.I2}`); assert.ok(H.p > 0.01, `even map p ${H.p}`); }
    else { assert.ok(H.I2 > 0.9, `gradient I² ${H.I2}`); near(H.tau, 30 / Math.sqrt(12) * Math.sqrt(12 / 11), 3, "tau = SD of the gradient"); assert.ok(H.moran > 0.5, `Moran ${H.moran}`); }
  }
});

test("a narrow fit range is refused rather than fitted", () => {
  assert.throws(() => prepare(new Float32Array(WN.length), WN.length, WN, 1, { ...S, range: [1500, 1510] }), /widen the fit range/);
});
