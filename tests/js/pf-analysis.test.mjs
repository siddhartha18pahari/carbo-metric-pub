// node --test tests/js/pf-analysis.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { rng, wilson, weightedQuantile, laCancado, robustScale, gmm, bestGmm, bootstrapShares, median } from "../../web/js/pf-analysis.js";

const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ""} ${a} vs ${b}`);

test("Wilson interval matches the textbook values", () => {
  const [lo, hi] = wilson(5, 10);
  near(lo, 0.2366, 1e-4); near(hi, 0.7634, 1e-4);
  const [l0, h0] = wilson(0, 20);                 // no successes: lower bound 0, upper bound > 0
  assert.equal(l0, 0); near(h0, 0.1611, 1e-4);
  assert.deepEqual(wilson(0, 0).map(Number.isNaN), [true, true]);
});

test("weighted quantile: equal weights give the median, weight moves it", () => {
  near(weightedQuantile([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], 0.5), 3, 1e-12);
  // nearly all the weight on the largest value: D50 by weight sits just below it
  // (each weight is centred on its value: 2 + (0.5 - 0.015) / (0.51 - 0.015))
  near(weightedQuantile([1, 2, 3], [1, 1, 98], 0.5), 2 + 0.485 / 0.495, 1e-12);
  // a particle of zero weight does not count
  near(weightedQuantile([1, 2, 100], [1, 1, 0], 0.5), 1.5, 1e-12);
});

test("La from Cancado: 19.2 nm at A_D = A_G with a 532 nm laser, inversely proportional", () => {
  near(laCancado(1, 1, 532), 2.4e-10 * 532 ** 4, 1e-9);
  near(laCancado(1, 1, 532), 19.23, 0.01);
  near(laCancado(2, 1, 532), laCancado(1, 1, 532) / 2, 1e-9);
  assert.ok(Number.isNaN(laCancado(0, 1, 532)));   // no D band: no finite size
});

test("robust scaling ignores one wild value", () => {
  const [z] = robustScale([[1, 2, 3, 4, 5, 6, 7, 1000]]);
  near(median(z), 0, 1e-12);
  assert.ok(Math.abs(z[1] - z[0]) < 1);           // the bulk keeps unit-ish spacing
});

// two well-separated populations, 300 and 100 particles, in two features
function twoPops(seed = 3) {
  const R = rng(seed), gauss = () => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
  const a = [], b = [], truth = [];
  for (let i = 0; i < 400; i++) {
    const c = i < 300 ? 0 : 1;
    a.push((c ? 4 : 0) + gauss() * 0.5); b.push((c ? -3 : 1) + gauss() * 0.4); truth.push(c);
  }
  return { cols: [a, b], truth };
}

test("the mixture finds two populations, sized right, and BIC prefers k = 2", () => {
  const { cols, truth } = twoPops();
  const best = bestGmm(cols, 4, 1);
  assert.equal(best.k, 2);
  const big = best.weights[0] > best.weights[1] ? 0 : 1;
  near(best.weights[big], 0.75, 0.02);
  let agree = 0;
  for (let i = 0; i < truth.length; i++) agree += (best.labels[i] === big) === (truth[i] === 0);
  assert.ok(agree / truth.length > 0.99, `agreement ${agree}`);
  // the same seed gives the same answer
  assert.deepEqual(Array.from(bestGmm(cols, 4, 1).labels), Array.from(best.labels));
});

test("one population stays one", () => {
  const R = rng(9), g = () => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
  const cols = [Array.from({ length: 300 }, g), Array.from({ length: 300 }, g)];
  assert.equal(bestGmm(cols, 4, 1).k, 1);
});

test("a mixture log-likelihood never falls with more components from the same start", () => {
  const { cols } = twoPops(5);
  assert.ok(gmm(cols, 2, { seed: 2 }).logLik > gmm(cols, 1, { seed: 2 }).logLik);
});

test("bootstrap shares: the point share sits inside its interval, and weights matter", () => {
  const labels = Int16Array.from({ length: 200 }, (_, i) => (i < 150 ? 0 : 1));
  const even = bootstrapShares(labels, new Float64Array(200).fill(1), 2);
  near(even[0][0], 0.75, 1e-12);
  assert.ok(even[0][1] < 0.75 && even[0][2] > 0.75);
  // the 50 particles of population 1 are 9 times heavier: 75 % by volume
  const w = Float64Array.from(labels, l => (l ? 9 : 1));
  near(bootstrapShares(labels, w, 2)[1][0], 450 / 600, 1e-12);
  // particles left out (label -1) still count in the total
  const some = Int16Array.from({ length: 100 }, (_, i) => (i < 50 ? 0 : -1));
  near(bootstrapShares(some, new Float64Array(100).fill(1), 1)[0][0], 0.5, 1e-12);
});

import { laserNm } from "../../web/js/engine/readers.js";

// a "Laser" record as LabSpec 6 saves it: name, padding, "pos" 40 bytes in, the double 8 bytes after
function laserRecord(nm) {
  const b = new Uint8Array(6 + 34 + 3 + 5 + 8 + 8);
  b.set([76, 97, 115, 101, 114, 0], 0); b.set([112, 111, 115], 40);
  new DataView(b.buffer).setFloat64(48, nm, true);
  return b;
}
const cat = (...parts) => { const n = parts.reduce((s, p) => s + p.length, 0), o = new Uint8Array(n); let i = 0; for (const p of parts) { o.set(p, i); i += p.length; } return o; };

test("laser wavelength: the last plausible acquisition record, none when only 'Off' is saved", () => {
  assert.equal(laserNm(cat(new Uint8Array(100), laserRecord(785), new Uint8Array(50), laserRecord(532.12), new Uint8Array(20))), 532.12);
  assert.equal(laserNm(cat(new Uint8Array(10), laserRecord(0))), null);
  assert.equal(laserNm(new Uint8Array(300)), null);
});

test("a continuous spread is one population, not slices of it", () => {
  // uniform in a square: a mixture fits it better with several pieces, but none is separated
  const R = rng(11), cols = [Array.from({ length: 2000 }, () => R()), Array.from({ length: 2000 }, () => R())];
  const best = bestGmm(cols, 4, 1);
  assert.equal(best.k, 1);
});

import { heightCovariances, ratioVariance, heterogeneity, moranI, chi2Upper, chi2Quantile, spearmanCI } from "../../web/js/pf-analysis.js";
import { fitSpectra } from "../../web/js/engine/core.js";
import { DEFAULT_CONFIG } from "../../web/js/engine/config.js";

// replicate spectra of one carbon with a weak, broad D3 (2.5 counts over 150 cm-1), noise 1.2 per channel
function replicates(P = 400, sigma = 1.2, seed = 21) {
  const R = rng(seed), g = () => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
  const wn = Float64Array.from({ length: 297 }, (_, k) => 1000 + k * 800 / 296), K = wn.length;
  const lor = (x, c, w) => 1 / (1 + (2 * (x - c) / w) ** 2), gau = (x, c, w) => Math.exp(-2.772588722239781 * ((x - c) / w) ** 2);
  const truth = x => { const t = (x - 1400) / 400; return 40 + 5 * t - 3 * t * t + 60 * lor(x, 1581, 22) + 15 * lor(x, 1348, 55) + 5 * lor(x, 1618, 30) + 2.5 * gau(x, 1505, 150) + 3 * lor(x, 1200, 80); };
  const cube = new Float32Array(P * K);
  for (let p = 0; p < P; p++) for (let k = 0; k < K; k++) cube[p * K + k] = truth(wn[k]) + sigma * g();
  return { cube, wn, P };
}
const sd = a => { const m = a.reduce((s, v) => s + v, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1)); };
const rms = a => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);

test("band-height standard errors match the scatter of repeated fits", () => {
  const { cube, wn, P } = replicates();
  const cfg = structuredClone(DEFAULT_CONFIG), fit = fitSpectra(cube, P, wn, cfg), m = fit.m;
  const cov = heightCovariances(fit, cfg, wn);
  const j3 = fit.names.indexOf("D3"), jG = fit.names.indexOf("G");
  const A3 = [], se3 = [], AG = [], seG = [], R = [], seR = [];
  for (let p = 0; p < P; p++) {
    const o = p * m * m, a3 = fit.intensity[p * m + j3], aG = fit.intensity[p * m + jG];
    A3.push(a3); se3.push(Math.sqrt(cov[o + j3 * m + j3])); AG.push(aG); seG.push(Math.sqrt(cov[o + jG * m + jG]));
    R.push(a3 / aG); seR.push(Math.sqrt(ratioVariance(cov, o, m, j3, jG, a3, aG)));
  }
  // the predicted errors agree with the observed spread to within 15 %
  for (const [obs, pred, what] of [[sd(A3), rms(se3), "D3 height"], [sd(AG), rms(seG), "G height"], [sd(R), rms(seR), "D3/G"]])
    assert.ok(Math.abs(pred / obs - 1) < 0.15, `${what}: predicted ${pred} vs observed ${obs}`);
  // a broad band is well determined although its height is twice the noise of one channel
  assert.ok(rms(se3) < 0.5, `D3 standard error ${rms(se3)}`);
  // and replicates of one carbon look uniform
  const h = heterogeneity(R, seR);
  assert.ok(h.p > 0.01 && h.I2 < 0.25, `uniform replicates: p ${h.p}, I2 ${h.I2}`);
});

test("heterogeneity: no spread beyond the errors, then a known 30 % spread recovered", () => {
  const Rg = rng(5), g = () => Math.sqrt(-2 * Math.log(Rg() + 1e-12)) * Math.cos(2 * Math.PI * Rg());
  const n = 1500, se = Array.from({ length: n }, () => 0.004 + 0.006 * Rg());
  const uniform = se.map(s => 0.05 + s * g());
  const h0 = heterogeneity(uniform, se);
  assert.ok(h0.p > 0.01 && h0.I2 < 0.1 && h0.tauLo === 0, JSON.stringify(h0));
  near(h0.mu, 0.05, 0.001);
  const dev = se.map(() => 0.015 * g());                            // true SD 0.015: CV 30 %
  const varied = se.map((s, i) => 0.05 + dev[i] + s * g());
  const h1 = heterogeneity(varied, se), real = sd(dev);              // the spread actually drawn
  assert.ok(h1.p < 1e-6 && h1.I2 > 0.75, JSON.stringify(h1));
  assert.ok(h1.tauLo < real && h1.tauHi > real, `tau ${h1.tau} [${h1.tauLo}, ${h1.tauHi}] vs ${real}`);
  near(h1.tau, real, 0.1 * real);
});

test("chi-square tail and quantile are consistent", () => {
  near(chi2Upper(chi2Quantile(1.959964, 200), 200), 0.025, 0.002);
  near(chi2Upper(3.84, 1), 0.05, 0.02);            // rough at df = 1
  near(chi2Upper(1000, 1000), 0.5, 0.02);
});

test("Moran's I: random placement near zero, a gradient across the sample clusters", () => {
  const R = rng(8), n = 600, x = [], y = [];
  for (let i = 0; i < n; i++) { x.push(R() * 100); y.push(R() * 100); }
  const noise = x.map(() => R());
  const m0 = moranI(noise, x, y);
  assert.ok(Math.abs(m0.I) < 0.08 && m0.p > 0.01, JSON.stringify(m0));
  const grad = x.map((v, i) => v / 100 + 0.2 * noise[i]);
  const m1 = moranI(grad, x, y);
  assert.ok(m1.I > 0.5 && m1.p <= 0.01, JSON.stringify(m1));
});

test("Spearman interval narrows with n and stays inside -1..1", () => {
  const [a, b] = spearmanCI(0.3, 1000), [c, d] = spearmanCI(0.3, 50);
  assert.ok(a > 0.24 && b < 0.36 && c < a && d > b && c > -1 && d < 1);
});

import { coverage } from "../../web/js/pf-analysis.js";

// a map's particles as index groups: n pixels each, true value mu + between offset + within deviation, seen with noise se
function particlesOf({ k, n, mu = 0.05, sdB = 0, sdW = 0, seed = 13 }) {
  const R = rng(seed), g = () => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
  const y = [], se = [], groups = [], b = [], w = [];
  for (let i = 0; i < k; i++) {
    const off = sdB * g(), idx = [];
    b.push(off);
    for (let j = 0; j < n; j++) { const s = 0.004 + 0.006 * R(), dev = sdW * g(); w.push([i, dev]); idx.push(y.length); se.push(s); y.push(mu + off + dev + s * g()); }
    groups.push(idx);
  }
  // the spreads actually drawn: between, and pooled within (about each particle's own mean)
  const mw = Array.from({ length: k }, (_, i) => w.filter(v => v[0] === i).reduce((s, v) => s + v[1], 0) / n);
  const realW = Math.sqrt(w.reduce((s, [i, d]) => s + (d - mw[i]) ** 2, 0) / (k * (n - 1)));
  return { y, se, groups, realB: sd(b), realW };
}

test("coverage: an even coating on every particle shows no spread within or between", () => {
  const { y, se, groups } = particlesOf({ k: 20, n: 120 });
  const c = coverage(y, se, groups);
  assert.equal(c.particles, 20);
  assert.ok(c.within.p > 0.01 && c.within.I2 < 0.1 && c.within.tauLo === 0, JSON.stringify(c.within));
  assert.ok(c.between.p > 0.01 && c.between.tauLo === 0, JSON.stringify(c.between));
  near(c.mu, 0.05, 0.001);
  // each particle tested on its own: about 5 % flagged by chance
  assert.ok(c.parts.filter(h => h.p < 0.05).length <= 4);
  // errors 30 % too large: the pixels scatter less than they predict, the interval
  // collapses to 0, and only the plain scatter still bounds a real spread
  const big = coverage(y, se.map(s => 1.3 * s), groups);
  assert.ok(big.within.p > 0.975 && big.within.tauHi === 0, JSON.stringify(big.within));
  near(big.within.sd, c.within.noise, 0.1 * c.within.noise);
});

test("coverage: known spreads within and between particles are recovered and split", () => {
  const { y, se, groups, realB, realW } = particlesOf({ k: 40, n: 150, sdB: 0.006, sdW: 0.01 });
  const c = coverage(y, se, groups);
  assert.ok(c.within.p < 1e-6 && c.within.tauLo < realW && c.within.tauHi > realW, `within ${JSON.stringify(c.within)} vs ${realW}`);
  near(c.within.tau, realW, 0.1 * realW);
  assert.ok(c.between.p < 1e-6 && c.between.tauLo < realB && c.between.tauHi > realB, `between ${JSON.stringify(c.between)} vs ${realB}`);
  near(c.between.tau, realB, 0.25 * realB);
  near(c.shareWithin, realW ** 2 / (realW ** 2 + realB ** 2), 0.1);
  // the plain scatter inside the particles is the noise and the real spread together
  near(c.within.sd ** 2, c.within.noise ** 2 + realW ** 2, 0.1 * c.within.sd ** 2);
  // each particle's own within-particle SD, around the common one
  const taus = c.parts.map(h => h.tau);
  near(taus.reduce((s, v) => s + v, 0) / taus.length, 0.01, 0.0015);
});

test("coverage: particles that differ but are each evenly coated", () => {
  const { y, se, groups } = particlesOf({ k: 15, n: 100, sdB: 0.01 });
  const c = coverage(y, se, groups);
  assert.ok(c.within.p > 0.01 && c.within.tauLo === 0, JSON.stringify(c.within));
  assert.ok(c.between.p < 1e-6 && c.between.I2 > 0.9, JSON.stringify(c.between));
  assert.ok(c.shareWithin < 0.05);
});

test("coverage: one patchy particle stands out among even ones", () => {
  const { y, se, groups } = particlesOf({ k: 10, n: 120 });
  groups[3].forEach((i, j) => { if (j % 2) y[i] += 0.02; });     // half its pixels carry 40 % more coating
  const c = coverage(y, se, groups);
  assert.ok(c.parts[3].p < 1e-6 && c.parts[3].I2 > 0.5, JSON.stringify(c.parts[3]));
  for (const [k, h] of c.parts.entries()) if (k !== 3) assert.ok(h.I2 < 0.3, `particle ${k}: I2 ${h.I2}`);
  assert.ok(c.within.p < 1e-6);
});

test("coverage through the fit: an evenly coated particle passes, a patchy one does not", () => {
  // two particles of 200 spectra each; in the second, D3 is 2.5 counts on half the pixels and 6 on the rest
  const R = rng(31), g = () => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
  const wn = Float64Array.from({ length: 297 }, (_, k) => 1000 + k * 800 / 296), K = wn.length, P = 400;
  const lor = (x, c, w) => 1 / (1 + (2 * (x - c) / w) ** 2), gau = (x, c, w) => Math.exp(-2.772588722239781 * ((x - c) / w) ** 2);
  const spec = (x, d3) => { const t = (x - 1400) / 400; return 40 + 5 * t - 3 * t * t + 60 * lor(x, 1581, 22) + 15 * lor(x, 1348, 55) + 5 * lor(x, 1618, 30) + d3 * gau(x, 1505, 150) + 3 * lor(x, 1200, 80); };
  const cube = new Float32Array(P * K);
  for (let p = 0; p < P; p++) { const d3 = p < 200 || p % 2 ? 2.5 : 6; for (let k = 0; k < K; k++) cube[p * K + k] = spec(wn[k], d3) + 1.2 * g(); }
  const cfg = structuredClone(DEFAULT_CONFIG), fit = fitSpectra(cube, P, wn, cfg), m = fit.m;
  const cov = heightCovariances(fit, cfg, wn), j3 = fit.names.indexOf("D3"), jG = fit.names.indexOf("G");
  const y = [], se = [];
  for (let p = 0; p < P; p++) {
    const a3 = fit.intensity[p * m + j3], aG = fit.intensity[p * m + jG];
    y.push(a3 / aG); se.push(Math.sqrt(ratioVariance(cov, p * m * m, m, j3, jG, a3, aG)));
  }
  const groups = [Array.from({ length: 200 }, (_, i) => i), Array.from({ length: 200 }, (_, i) => 200 + i)];
  const c = coverage(y, se, groups);
  assert.ok(c.parts[0].p > 0.01 && c.parts[0].I2 < 0.25, `even particle: p ${c.parts[0].p}, I2 ${c.parts[0].I2}`);
  assert.ok(c.parts[1].p < 1e-6 && c.parts[1].I2 > 0.5, `patchy particle: p ${c.parts[1].p}, I2 ${c.parts[1].I2}`);
  // the patchy particle's true spread: D3/G of 2.5/60 and 6/60 in equal parts, SD 1.75/60
  near(c.parts[1].tau, 1.75 / 60, 0.25 * 1.75 / 60);
  assert.equal(c.between, null);                                 // two particles cannot be compared this way
});

import { pairRatio } from "../../web/js/pf-analysis.js";

test("neighbour pairs: 1 for noise alone, rising with distance over a real pattern, never across particles", () => {
  const R = rng(17), g = () => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
  const nx = 60, ny = 60, N = nx * ny, se = Float64Array.from({ length: N }, () => 0.004 + 0.006 * R());
  // two particles side by side with very different means: pairs across the border would inflate the ratio
  const labels = Int32Array.from({ length: N }, (_, p) => ((p % nx) < 30 ? 1 : 2));
  const flat = Float64Array.from({ length: N }, (_, p) => (labels[p] === 1 ? 0.03 : 0.08) + se[p] * g());
  const r1 = pairRatio(flat, se, labels, nx, ny, 1);
  near(r1.ratio, 1, 0.06);
  assert.equal(r1.n, ny * (nx - 2) + nx * (ny - 1));              // rows lose the pair across the border
  // a patch pattern 20 pixels across (SD 0.007): little change between neighbours, more further apart
  const patch = Float64Array.from({ length: N }, (_, p) => flat[p] + 0.007 * Math.SQRT2 * Math.sin(2 * Math.PI * (p % nx) / 20) * Math.sin(2 * Math.PI * Math.floor(p / nx) / 20));
  const [a, b, c] = [1, 3, 6].map(L => pairRatio(patch, se, labels, nx, ny, L).ratio);
  assert.ok(a < 1.1 && b > a + 0.1 && c > b + 0.3, `${a} ${b} ${c}`);
});

test("band-height covariances for a subset of spectra match the full computation", () => {
  const { cube, wn } = replicates(30);
  const cfg = structuredClone(DEFAULT_CONFIG), fit = fitSpectra(cube, 30, wn, cfg), mm = fit.m * fit.m;
  const all = heightCovariances(fit, cfg, wn), out = new Float64Array(30 * mm).fill(NaN);
  heightCovariances(fit, cfg, wn, [4, 17], out);
  for (const p of [4, 17]) for (let k = 0; k < mm; k++) assert.equal(out[p * mm + k], all[p * mm + k]);
  assert.ok(Number.isNaN(out[5 * mm]));                           // the others are left alone
});

import { binomLower, certifyCount, detectionPower, signalFloor, normInv, gini, noiseGini, logCV, lognormalPreferred,
         gridMoran, effectiveN, variogram, fitVariogram, hotSpots, blockShuffle } from "../../web/js/pf-analysis.js";

test("exact binomial lower bounds and the particle counts that certify coverage", () => {
  near(binomLower(100, 3), 0.924289, 1e-6); near(binomLower(50, 10), 0.684404, 1e-6);   // scipy beta.ppf
  near(binomLower(143, 0), 0.05 ** (1 / 143), 1e-12);
  assert.equal(binomLower(10, 10), 0);
  // zero failures: 59 particles certify 95 %, 299 certify 99 %, 598 certify 99.5 %
  assert.deepEqual([0.95, 0.99, 0.995].map(p => certifyCount(p)), [59, 299, 598]);
  // at 99 % detection power and 1 % false detections a count first passes at 529 for 99 % but fails
  // again until 660 (the rounded failures): every count from 661 passes, and 1 867 for 99.5 %
  assert.deepEqual([0.95, 0.99, 0.995].map(p => certifyCount(p, 0.99, 0.01)), [79, 661, 1867]);
  for (const n of [661, 700, 999]) assert.ok(binomLower(n, Math.round(0.01 * n)) >= 0.99 * 0.99 + 0.01 * 0.01, `n ${n}`);
  assert.ok(binomLower(660, 7) < 0.99 * 0.99 + 0.01 * 0.01);         // the last count that fails
  assert.ok(binomLower(529, 5) >= 0.99 * 0.99 + 0.01 * 0.01);        // the first that passes, before failures return
  // certifiable with enough particles whenever the power beats the false detections; never when it does not
  assert.equal(certifyCount(0.99, 0.01, 0.01), null);
  assert.equal(certifyCount(0.99, 0.5, 0.01), null);                // possible, but beyond 20 000 particles
});

test("the signal floor gives exactly the detection power asked for, and more above it", () => {
  near(normInv(0.95), 1.6448536, 1e-6); near(normInv(0.025), -1.959964, 1e-6);
  const g = signalFloor(0.06, 0.02, 1.0, 2.0, 0.95);
  near(detectionPower(0.06, 0.02, 1.0, 2.0, g), 0.95, 1e-6);
  assert.ok(detectionPower(0.06, 0.02, 1.0, 2.0, 2 * g) > 0.99);
  assert.equal(signalFloor(0.02, 0.03, 1, 1), Infinity);           // a coating below the threshold is never found
});

test("inequality and distribution shape: Gini, its noise-only value, lognormal against normal", () => {
  near(gini([2, 2, 2, 2]), 0, 1e-12); near(gini([0, 0, 0, 1]), 0.75, 1e-12);
  const R = rng(4), g = () => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
  const se = Array.from({ length: 2000 }, () => 0.005), flat = se.map(s => 0.05 + s * g());
  near(gini(flat), noiseGini(0.05, se), 0.005);                       // noise alone: no excess
  const varied = flat.map(v => v * Math.exp(0.3 * g()));
  assert.ok(gini(varied) - noiseGini(0.05, se) > 0.1);
  assert.equal(lognormalPreferred(flat.map(v => Math.exp(10 * (v - 0.05) * 20))), true);
  assert.equal(lognormalPreferred(flat), false);
  near(logCV(Array.from({ length: 500 }, () => Math.exp(0.2 * g()))), Math.sqrt(Math.expm1(0.04)), 0.02);
});

// a smooth field (a few broad patches) plus white noise of known size, on a disc
function patchField(noise = 0.004, seed = 9) {
  const R = rng(seed), g = () => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
  const nx = 70, ny = 60, img = new Float64Array(nx * ny), ok = new Uint8Array(nx * ny);
  const bumps = Array.from({ length: 12 }, () => [R() * nx, R() * ny, 0.004 * g()]);
  for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
    let s = 0.05; for (const [x, y, a] of bumps) s += a * Math.exp(-((j - x) ** 2 + (i - y) ** 2) / 60);
    img[i * nx + j] = s + noise * g();
    ok[i * nx + j] = (i - 30) ** 2 / 26 ** 2 + (j - 35) ** 2 / 30 ** 2 <= 1 ? 1 : 0;
  }
  return { img, ok, nx, ny };
}

test("the variogram nugget recovers the white noise, and Moran's I sees the patches", () => {
  const { img, ok, nx, ny } = patchField(0.004);
  const vg = variogram(img, ok, nx, ny, 15);
  assert.ok(vg.classical[0] < vg.classical[vg.classical.length - 1]);   // the field varies more at a distance
  const fit = fitVariogram(vg.lags, vg.classical, 15);
  assert.ok(fit.nugget / 0.004 ** 2 > 0.5 && fit.nugget / 0.004 ** 2 < 2, `nugget ratio ${fit.nugget / 0.004 ** 2}`);
  const white = patchField(0.004, 3); for (let p = 0; p < white.img.length; p++) white.img[p] = 0.05 + 0.004 * (rng(p + 1)() - 0.5) * 3.46;
  assert.ok(Math.abs(gridMoran(white.img, white.ok, nx, ny)) < 0.05);
  assert.ok(gridMoran(img, ok, nx, ny) > 0.2);
  near(effectiveN(100, 0), 100, 1e-12); near(effectiveN(100, 0.5), 100 / 3, 1e-9);
});

test("locally thin places: a planted thin patch is found, and noise alone gives none", () => {
  const R = rng(12), g = () => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
  const nx = 70, ny = 60, img = new Float64Array(nx * ny), ok = new Uint8Array(nx * ny);
  for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
    img[i * nx + j] = 0.05 + 0.004 * g();
    ok[i * nx + j] = (i - 30) ** 2 / 26 ** 2 + (j - 35) ** 2 / 30 ** 2 <= 1 ? 1 : 0;
  }
  assert.ok(hotSpots(img, ok, nx, ny).cold < 0.005);                 // white noise: no thin place
  const disc = (i, j) => (i - 30) ** 2 + (j - 20) ** 2 <= 36, thin = Float64Array.from(img);
  let inDisc = 0; for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) if (disc(i, j)) { thin[i * nx + j] -= 0.02; inDisc += ok[i * nx + j]; }
  const found = hotSpots(thin, ok, nx, ny), sh = blockShuffle(thin, ok, nx, ny, 8), nul = hotSpots(sh.img, sh.ok, sh.nx, sh.ny);
  // the 5 x 5 neighbourhood spreads a thin place into its surroundings, so its extent is overstated
  const share = inDisc / found.n;
  assert.ok(found.cold >= 0.9 * share && found.cold <= 2 * share, `flagged ${found.cold} vs patch ${share}`);
  // moving whole 8 x 8 blocks keeps a patch no bigger than a few blocks largely together, so the
  // block null finds it too: only structure larger than the blocks shows as an excess over it
  assert.ok(Math.abs(nul.cold - found.cold) < 0.5 * found.cold, JSON.stringify({ found, nul }));
  const sh0 = blockShuffle(img, ok, nx, ny, 8);
  assert.ok(hotSpots(sh0.img, sh0.ok, sh0.nx, sh0.ny).cold < 0.005);   // and noise moved in blocks is still noise
});

import { floorWithReference } from "../../web/js/pf-analysis.js";

test("floor and threshold together: the reference's scatter at the brightness of the floor", () => {
  // an uncoated reference whose index scatters as 1/G (D3 noise 1 count, G noise 2 counts)
  const R = rng(21), g = () => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
  const G = Array.from({ length: 4000 }, () => 20 + 380 * R()), r = G.map(v => 0.01 + Math.sqrt(1 + 0.0004) / v * g());
  const all = Float64Array.from(r).sort(), tauAll = all[Math.floor(0.99 * (all.length - 1))];
  const t = floorWithReference(G, r, 0.03, 1, 2, 0.95);
  assert.ok(t && t.g >= signalFloor(0.03, t.tau, 1, 2, 0.95), JSON.stringify(t));
  assert.ok(t.tau < tauAll && t.n >= 100);                     // bright spectra scatter less than all of them
  // a lower floor would not hold: its threshold, taken over dimmer spectra, needs a higher floor than itself
  const lower = G.filter(v => v >= 0.8 * t.g), idx = G.map((v, i) => i).filter(i => G[i] >= 0.8 * t.g);
  const rl = Float64Array.from(idx, i => r[i]).sort(), tauL = rl[Math.floor(0.99 * (rl.length - 1))];
  assert.ok(0.8 * t.g < signalFloor(0.03, tauL, 1, 2, 0.95), `${lower.length}`);
  assert.equal(floorWithReference(G, r, 0.011, 1, 2, 0.95), null); // a coating barely above uncoated is never found
});

import { hexBins } from "../../web/js/pf-analysis.js";

test("hexagon bins: every point counted once, in the hexagon whose centre is nearest", () => {
  const R = rng(6), xs = [], ys = [];
  for (let i = 0; i < 3000; i++) { xs.push(R() * 100); ys.push(R() * 60); }
  const s = 4, cells = hexBins(xs, ys, s);
  assert.equal(cells.reduce((t, c) => t + c.n, 0), 3000);
  // each point lies within its hexagon: no centre is nearer than the one it was counted in
  for (let i = 0; i < 300; i++) {
    const d = cells.map(c => Math.hypot(c.x - xs[i], c.y - ys[i])), best = Math.min(...d);
    assert.ok(best <= s + 1e-9, `point ${i} is ${best} from the nearest centre`);
  }
  // a uniform cloud fills its hexagons evenly: about 3000 / (6000 / (2.598 s²)) points each
  const counts = cells.filter(c => c.x > 8 && c.x < 92 && c.y > 8 && c.y < 52).map(c => c.n);
  near(counts.reduce((t, v) => t + v, 0) / counts.length, 3000 / (6000 / (1.5 * Math.sqrt(3) * s * s)), 4);
  assert.deepEqual(hexBins([1, NaN], [1, 2], 2).map(c => c.n), [1]);
});

import { robustSD, boundedCoverage, lowShare, qcd, qRatio, evenness, moments, gridGeary, spatialPermutation, neighbourNoise,
         blockBootstrap, ksTest, fitFreeHeights, nestedSize } from "../../web/js/pf-analysis.js";

const normal = R => () => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());

test("distribution measures match the reference implementation (numpy / scipy)", () => {
  const x = [1, 2, 3, 4, 10, 0.5, 7, 2.2];
  near(gini(x), 0.4347643097643099, 1e-12);
  near(lowShare(x, 0.1), 0.016835016835016835, 1e-12);
  near(qcd(x), 0.46153846153846156, 1e-12);
  near(qRatio(x), 9.294117647058822, 1e-9);
  near(evenness(x, 16), 0.6875, 1e-12);
  const m = moments(x);
  near(m.skew, 0.9905431095842289, 1e-12); near(m.kurt, -0.2783227148162033, 1e-12);
  near(lowShare(Array(50).fill(3), 0.1), 0.1, 1e-12);            // equal values: the lowest tenth holds a tenth
  assert.equal(evenness(Array(20).fill(2), 16), 0);                // all in one bin
  near(robustSD([1, 2, 3, 4, 100]), 1.4826, 1e-12);                // MAD 1, the outlier ignored
});

test("Geary's c matches the reference value, and the permutation test tells clustering from noise", () => {
  const img = [1, 2, 3, 2, 5, 4, 3, 4, 9], ok = [0, 1, 1, 1, 1, 1, 1, 1, 1];
  near(gridGeary(img, ok, 3, 3), 0.7194444444444444, 1e-12);
  const R = rng(8), g = normal(R), n = 30, noise = new Float64Array(n * n), smooth = new Float64Array(n * n), all = new Uint8Array(n * n).fill(1);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { noise[i * n + j] = 5 + g(); smooth[i * n + j] = 5 + Math.sin(i / 5) + Math.cos(j / 6) + 0.3 * g(); }
  const a = spatialPermutation(noise, all, n, n, 99), b = spatialPermutation(smooth, all, n, n, 99);
  assert.ok(a.pI > 0.05 && a.pC > 0.05 && Math.abs(a.c - 1) < 0.15, `noise: ${JSON.stringify(a)}`);
  assert.ok(b.pI <= 0.01 && b.pC <= 0.01 && b.c < 0.5, `smooth: ${JSON.stringify(b)}`);
});

test("neighbouring-pixel noise recovers the noise under a smooth field", () => {
  const R = rng(4), g = normal(R), n = 40, img = new Float64Array(n * n), ok = new Uint8Array(n * n).fill(1);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) img[i * n + j] = 100 + 20 * Math.sin(i / 12) * Math.cos(j / 15) + 3 * g();
  const r = neighbourNoise(img, ok, n, n);
  near(r.sd, 3, 0.3); assert.equal(r.n, 2 * n * (n - 1));
});

test("block bootstrap: noise alone leaves no excess CV, real patches do, and coverage gets an interval", () => {
  const R = rng(9), g = normal(R), n = 48, ok = new Uint8Array(n * n).fill(1), sig = new Float64Array(n * n).fill(0.02);
  const flat = new Float64Array(n * n), patchy = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    flat[i * n + j] = 0.2 + 0.02 * g();
    patchy[i * n + j] = 0.2 * (1 + 0.3 * Math.sin(i / 4) * Math.sin(j / 5)) + 0.02 * g();   // patches of about 6 pixels
  }
  const a = blockBootstrap(flat, sig, ok, n, n, { tau: 0.15 }), b = blockBootstrap(patchy, sig, ok, n, n, { tau: 0.15 });
  assert.equal(a.blocks, 36);
  assert.ok(a.est.cvExcess < 0.03 && a.ci.cvExcess[0] === 0, `flat ${JSON.stringify(a.ci.cvExcess)}`);
  assert.ok(b.ci.cvExcess[0] > 0.08, `patchy ${JSON.stringify(b.ci.cvExcess)}`);
  assert.ok(a.ci.cover[0] > 0.98 && b.ci.cover[0] < b.est.cover && b.est.cover < b.ci.cover[1]);
  assert.ok(b.est.U > a.est.U && b.ci.U[0] <= b.est.U && b.est.U <= b.ci.U[1]);
  assert.equal(blockBootstrap(flat, sig, ok, 6, 6), null);        // one block of 8 x 8: fewer than 4
});

test("bounded coverage counts the spectra left out both ways and certifies on the worst case", () => {
  const r = boundedCoverage(90, 100, 10);
  near(r.c, 0.9, 1e-12); near(r.lo, 90 / 110, 1e-12); near(r.hi, 100 / 110, 1e-12);
  near(r.lower, binomLower(110, 20), 1e-12);
  assert.ok(r.lower < binomLower(100, 10));
});

test("KS distance: the exact statistic, a small p for shifted samples and a large one for the same", () => {
  near(ksTest([0.1, 0.5, 0.9, 1.3, 2.0, 2.2, 3.1], [0.4, 1.1, 1.2, 2.5, 3.3, 3.9, 4.4, 5.0]).d, 0.5, 1e-12);
  const R = rng(2), g = normal(R), a = [], b = [], c = [];
  for (let i = 0; i < 400; i++) { a.push(g()); b.push(g()); c.push(g() + 0.5); }
  assert.ok(ksTest(a, b).p > 0.05 && ksTest(a, c).p < 1e-6);
});

test("fit-free heights recover the band heights and the channel noise of synthetic spectra", () => {
  const wn = Float64Array.from({ length: 801 }, (_, k) => 1000 + k), K = wn.length, R = rng(12), g = normal(R), n = 300;
  const cube = new Float32Array(n * K), hG = [], hC = [];
  for (let p = 0; p < n; p++) {
    const G = 800 + 400 * R(), C = 0.2 * G;
    hG.push(G); hC.push(C);
    for (let k = 0; k < K; k++) {
      const x = wn[k];
      cube[p * K + k] = 300 + 0.1 * (x - 1000) + G / (1 + ((x - 1580) / 8) ** 2) + C * Math.exp(-0.5 * ((x - 1500) / 40) ** 2) + 5 * g();
    }
  }
  const list = Array.from({ length: n }, (_, p) => p), r = fitFreeHeights(cube, K, wn, list, [1580, 1500]);
  near(r.sigma, 5, 0.3);
  // a 7-channel mean of a Lorentzian 8 cm-1 wide reads about 3 % low; D3 carries G's tail (1 % of G at 1500, less
  // the 0.2 % the same tail adds to the baseline through 1700-1740)
  for (let p = 0; p < n; p += 37) { near(r.heights[0][p] / hG[p], 0.97, 0.03); near((r.heights[1][p] - 0.0077 * hG[p]) / hC[p], 1, 0.03); }
  assert.ok(r.se[0] >= 5 * Math.sqrt(1 / 7) && r.se[0] < 5);
  assert.equal(fitFreeHeights(cube, K, Float64Array.from(wn, v => v + 300), list, [1580, 1500]), null);   // no 1030-1090 window
});

test("nested sample size: the between-particle term sets a floor no number of pixels removes", () => {
  const s = nestedSize(0.05, 0.1, 0.2, 0.1, 25);
  assert.equal(s.population, Math.ceil((0.01 + 0.04 + 0.01) / 0.0025));
  assert.equal(s.maps, Math.ceil((0.01 + 0.05 / 25) / 0.0025));
  assert.equal(s.floor, 4);
  assert.equal(nestedSize(0.05, 0.1, 0.2, 0.1, Infinity).maps, s.floor);
});

import { dispersion, nmf, nmfProject, fitSplats, splatScale, gaussSmooth } from "../../web/js/pf-analysis.js";

test("dispersion: Clark-Evans 2 on a square lattice, about 1 at random, low when clumped; touching and merged objects", () => {
  const xs = [], ys = [];
  for (let i = 0; i < 20; i++) for (let j = 0; j < 20; j++) { xs.push(5 + 10 * j); ys.push(5 + 10 * i); }
  const d4 = Array(400).fill(4), a = Array(400).fill(12.6);
  const lat = dispersion(xs, ys, d4, a, 200 * 200);
  near(lat.clarkEvans, 2, 1e-12); near(lat.nnMedian, 10, 1e-12); assert.equal(lat.touching, 0); assert.equal(lat.merged, 0);
  const R = rng(5), rx = [], ry = [];
  for (let i = 0; i < 2000; i++) { rx.push(1000 * R()); ry.push(1000 * R()); }
  const ran = dispersion(rx, ry, Array(2000).fill(1), Array(2000).fill(1), 1e6);
  assert.ok(ran.clarkEvans > 0.97 && ran.clarkEvans < 1.08, `random ${ran.clarkEvans}`);       // a few % high: no edge correction
  const cx = [], cy = [];
  for (let c = 0; c < 40; c++) { const x0 = 1000 * R(), y0 = 1000 * R(); for (let i = 0; i < 25; i++) { cx.push(x0 + 8 * R()); cy.push(y0 + 8 * R()); } }
  assert.ok(dispersion(cx, cy, Array(1000).fill(1), Array(1000).fill(1), 1e6).clarkEvans < 0.3);
  // two discs 6 across with centres 5 apart overlap; an object four times the median area counts as merged
  const t = dispersion([0, 5, 100, 200], [0, 0, 0, 0], [6, 6, 6, 6], [28, 28, 28, 120], 1e4);
  near(t.touching, 0.5, 1e-12); near(t.merged, 0.25, 1e-12); near(t.crowded, 0.75, 1e-12); near(t.gap[0], -1, 1e-12);
});

test("NMF recovers mixed component spectra and their amounts; projection onto fixed spectra is exact", () => {
  const K = 120, g = (c, w) => Array.from({ length: K }, (_, x) => Math.exp(-0.5 * ((x - c) / w) ** 2));
  // a narrow band and two broad ones that overlap it (a flat component would be absorbed by any other:
  // NMF's rotational ambiguity, which no starting point removes)
  const truth = [g(85, 3), g(40, 10), g(65, 16)];
  const R = rng(9), n = 600, k = 3, V = new Float64Array(n * K), Wt = [];
  for (let i = 0; i < n; i++) { const w = [R(), R() * 2, R() * 0.5]; Wt.push(w); for (let c = 0; c < K; c++) V[i * K + c] = w.reduce((t, a, j) => t + a * truth[j][c], 0) + 0.002 * R(); }
  // start from blurred, partly mixed versions of the truth
  const H0 = truth.map((h, j) => h.map((v, c) => 0.7 * v + 0.3 * truth[(j + 1) % 3][c] + 0.05));
  const { W, H } = nmf(V, n, K, H0, { iters: 200 });
  const cos = (a, b) => { let ab = 0, aa = 0, bb = 0; for (let c = 0; c < K; c++) { ab += a[c] * b[c]; aa += a[c] ** 2; bb += b[c] ** 2; } return ab / Math.sqrt(aa * bb); };
  for (let j = 0; j < k; j++) assert.ok(cos(H.subarray(j * K, (j + 1) * K), truth[j]) > 0.98, `component ${j}: ${cos(H.subarray(j * K, (j + 1) * K), truth[j])}`);
  // amounts: share of component 1 against the truth's share, in units of each component's sum
  const sum = truth.map(h => h.reduce((t, v) => t + v, 0)), share = (w, i) => w[i * k + 1] / (w[i * k] + w[i * k + 1] + w[i * k + 2]);
  const tShare = Wt.map(w => (w[1] * sum[1]) / (w[0] * sum[0] + w[1] * sum[1] + w[2] * sum[2]));
  const est = Array.from({ length: n }, (_, i) => share(W, i)), mE = est.reduce((a, b) => a + b) / n, mT = tShare.reduce((a, b) => a + b) / n;
  let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < n; i++) { sxy += (est[i] - mE) * (tShare[i] - mT); sxx += (est[i] - mE) ** 2; syy += (tShare[i] - mT) ** 2; }
  assert.ok(sxy / Math.sqrt(sxx * syy) > 0.98, `share r ${sxy / Math.sqrt(sxx * syy)}`);
  // with the true spectra fixed, the amounts come back
  const Ht = new Float64Array(k * K); truth.forEach((h, j) => h.forEach((v, c) => { Ht[j * K + c] = v; }));
  const Wp = nmfProject(V, n, K, Ht, k, { iters: 200 });
  for (let i = 0; i < n; i += 50) for (let j = 0; j < k; j++) near(Wp[i * k + j], Wt[i][j], 0.005, `pixel ${i} component ${j}`);
});

test("2-D splats: a field of two anisotropic Gaussians is recovered on points the fit never saw", async () => {
  const R = rng(17), gauss = () => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
  const truth = (x, y) => {
    const u1 = Math.cos(0.5) * (x - 0.35) + Math.sin(0.5) * (y - 0.4), u2 = -Math.sin(0.5) * (x - 0.35) + Math.cos(0.5) * (y - 0.4);
    return 0.5 + 2 * Math.exp(-0.5 * ((u1 / 0.12) ** 2 + (u2 / 0.05) ** 2)) + 1.2 * Math.exp(-0.5 * (((x - 0.75) / 0.08) ** 2 + ((y - 0.7) / 0.08) ** 2));
  };
  const mk = m => { const X = new Float64Array(2 * m), y = new Float64Array(m); for (let i = 0; i < m; i++) { X[2 * i] = R(); X[2 * i + 1] = R(); y[i] = truth(X[2 * i], X[2 * i + 1]) + 0.05 * gauss(); } return { X, y }; };
  const train = mk(1500), hold = mk(400);
  const { field, hist } = await fitSplats(train.X, train.y, { iters: 400, nMax: 32, seed: 2 });
  const p = field.predict(hold.X); let se = 0; for (let i = 0; i < 400; i++) se += (p[i] - hold.y[i]) ** 2;
  const rmse = Math.sqrt(se / 400);
  assert.ok(rmse < 0.075, `held-out RMSE ${rmse} against noise 0.05`);
  assert.ok(hist[hist.length - 1].rmse < 0.06 && hist[hist.length - 1].rmse < hist[0].rmse / 3, JSON.stringify(hist));
  const sc = splatScale(field, 101);
  assert.ok(sc.radius > 2 && sc.radius < 12, `scale ${sc.radius} px`);    // the true primitives: 7.7 and 8 px
});

test("Gaussian smoothing over a mask ignores the pixels outside it", () => {
  const nx = 12, ny = 10, img = new Float64Array(nx * ny).fill(3), ok = new Uint8Array(nx * ny).fill(1);
  for (let j = 0; j < 4; j++) { img[j] = 1000; ok[j] = 0; }
  const s = gaussSmooth(img, ok, nx, ny, 1);
  for (let p = 0; p < nx * ny; p++) near(s[p], 3, 1e-9);
});

import { gammaQ, kruskalWallis, brownForsythe, groupGap } from "../../web/js/pf-analysis.js";

test("between-map tests match the reference values (scipy)", () => {
  near(gammaQ(0.5, 1.6), 0.07363827012030257, 1e-10);           // chi-square, df 1
  near(gammaQ(1, 3.75), 0.023517745856009114, 1e-10);           // df 2
  near(gammaQ(12.5, 20), 0.029164395623152116, 1e-10);          // df 25
  near(gammaQ(2, 0.15), 0.9898141728888165, 1e-10);             // df 4
  const kw = kruskalWallis([[1.1, 2.5, 3.3, 4.0], [2.2, 5.1, 6.3, 7.7, 8.0], [0.5, 0.7, 1.9]]);
  near(kw.H, 7.317948717948717, 1e-12); near(kw.p, 0.025758918586256785, 1e-10); assert.equal(kw.df, 2);
  const bf = brownForsythe([[1, 2, 3, 4, 10], [2, 2.5, 3, 3.5, 4, 4.1], [0, 5, 10, 15]]);
  near(bf.F, 4.761898584905661, 1e-10); near(bf.p, 0.030031299933042023, 1e-9);
  // ties: KS-like data with repeats still gives a finite, corrected H
  assert.ok(Number.isFinite(kruskalWallis([[1, 1, 2, 2], [2, 3, 3, 3]]).H));
});

test("group gap: replicates closer than different samples give a small p; p never below one over the groupings", () => {
  // 3 samples x 3 replicate maps: distance 0.1 within a sample, 0.5 between samples
  const labels = ["A", "A", "A", "B", "B", "B", "C", "C", "C"], m = labels.length;
  const d = labels.map((a, i) => labels.map((b, j) => (i === j ? 0 : a === b ? 0.1 : 0.5)));
  const g = groupGap(d, labels, { nPerm: 2999 });
  near(g.within, 0.1, 1e-12); near(g.between, 0.5, 1e-12); near(g.gap, 0.4, 1e-12);
  assert.equal(g.groupings, 280);                                // 9! / (3!^3 3!)
  assert.ok(g.p < 0.01 && g.p >= 1 / 3000, `p ${g.p}`);
  // two maps of each of two samples: only 3 groupings, so even a perfect separation cannot reach p < 1/3
  const l2 = ["A", "A", "B", "B"], d2 = l2.map((a, i) => l2.map((b, j) => (i === j ? 0 : a === b ? 0.1 : 0.5)));
  const g2 = groupGap(d2, l2, { nPerm: 999 });
  assert.equal(g2.groupings, 3); assert.ok(g2.p > 0.25, `p ${g2.p}`);
  // no difference: p well above 0.05
  const R = rng(4), dn = labels.map(() => labels.map(() => 0)); for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) dn[i][j] = dn[j][i] = 0.3 + 0.05 * R();
  assert.ok(groupGap(dn, labels, { nPerm: 999 }).p > 0.05);
});

test("KS at the effective sample sizes: correlated pixels no longer look like a difference", () => {
  const R = rng(14), a = [], b = [];
  for (let i = 0; i < 2000; i++) { a.push(R()); b.push(R() * 1.03); }
  const raw = ksTest(a, b), eff = ksTest(a, b, 120, 120);
  near(raw.d, eff.d, 1e-15);
  assert.ok(eff.p > raw.p);
});

import { tTail2, tQuantile, thinTail, mixtureCoverage, olsHC3, nForMean, nForSpread, quantileSorted } from "../../web/js/pf-analysis.js";

test("t distribution matches the reference values (scipy)", () => {
  near(tQuantile(0.975, 5), 2.570581835636314, 1e-9); near(tQuantile(0.95, 30), 1.6972608865939574, 1e-9);
  near(tTail2(2.1, 12), 0.05754493873495118, 1e-10); near(tQuantile(0.025, 5), -2.570581835636314, 1e-9);
});

test("HC3 regression matches statsmodels: coefficients, robust errors, p-values and Holm", () => {
  const x1 = [2.040919121385182, -2.555665031314182, 0.418098846725779, -0.56776960612793, -0.452649292110446, -0.215597163089766, -2.019986129147251, -0.231932377644189, -0.865213076274942, 3.322999516644883, 0.225786613227922, -0.352630794341595, -0.28128741815135, -0.66804634610895, -1.055150551205121, -0.390800977234655, 0.481945388506786, -0.238553606573367, 0.957758702959764, -0.19980212906658, 0.024259565076665, 1.545820851212812, 0.545105522687645, -0.505228735614018, -0.182838974597735, 0.540525131754802, 1.935088034098853, -0.269620327341913, -0.243558679079105, 1.002313601275691, -0.886459943160587, -0.291720232439864, 0.882538967456484, 0.580350016190899, 0.091516703282352, 0.670104354828479, -2.828162306843763, 1.02130681750008, -0.959644759808142, -1.66861984265597, 0.276445759521, 0.70054488534939, -0.444767455682784, -1.076405840100808, 0.026124833534034, -0.052747308242879, 1.405598166018093, 0.74740798747935, 0.19381564626462, 1.111633205223992, -0.205523049905792, -0.925899573648368, 0.584058311025248, 0.58253841865569, -0.214828911126856, -0.782808577963966, 0.229153905213263, -2.49389427845799, 0.690124770162812, 0.491368260744991], x2 = [0.746860385649838, 0.752223418369277, 0.566977874452912, 0.92107967728293, 0.205775048229019, 0.850901124591651, 0.168987311327086, 0.96435772089428, 0.623692728106152, 0.606883788010015, 0.970558763132624, 0.787032713788771, 0.789917476088554, 0.054093757870965, 0.36928630609442, 0.084894772165468, 0.193527581465514, 0.213866990692147, 0.858641932165904, 0.126754979718305, 0.296757768389449, 0.492846978924333, 0.849460411189537, 0.965231420837375, 0.708144770687242, 0.213687219430548, 0.544982681458004, 0.705959042986887, 0.051882527726625, 0.679884167224071, 0.368281835139081, 0.589700286779208, 0.669533281141665, 0.669127496145936, 0.523049775525856, 0.554737432465089, 0.198149713613533, 0.495187391535402, 0.125409472645253, 0.480747001926064, 0.536244925569812, 0.774104816310318, 0.393664792994186, 0.01960041855161, 0.527749933213503, 0.205232492450088, 0.741257523194369, 0.388697230416737, 0.380473164132802, 0.909454579769297, 0.392970443955209, 0.348851577151557, 0.348019735358256, 0.480765570302582, 0.093292108751414, 0.546747407945715, 0.921427462033782, 0.562922064879737, 0.743910207350547, 0.947140663165677], x3 = [1.066983310895314, -1.302070858245795, -0.978548528621413, -0.801172010781182, 0.043295900283145, 0.640971064689471, 2.047886055357333, -0.197445429887671, 0.76750255890362, 0.155417810059435, 1.759926283908254, 0.742157861211569, 1.368550450874479, -1.077675189783463, -0.192240711525951, -0.81377242184204, 1.504947404008307, 0.657639903817773, -0.305144425777516, -0.452467887070401, 0.484664878206702, -0.701495530179553, -0.930588876212155, 0.481274492271532, 2.463132032105226, -0.246133554064813, -0.555865780793555, -1.171156834025543, -1.335010957559383, 0.524983234564226, 0.850803032778553, 0.00917472088249, 0.332575988676208, 0.115916577507104, 0.138654846047221, -1.526159052582847, -0.458118273049221, 0.111479307589948, -0.783166780563685, -0.476429743859817, -0.819120180327734, -0.333496674198441, 0.853108274325136, -0.406580350655598, -0.153870680197105, 0.813718359438118, 0.644770214156916, 1.695207558959633, -2.090485078128588, 0.856858944523106, -0.482284086773082, 0.134691818817881, 0.837719384899064, 1.083253155691345, 1.039350690308836, 0.155106691078253, 1.609662683018314, -0.282974227216137, -0.140981913070787, 0.799351188894086], y = [0.794651528206971, 3.151983488794485, 1.896656825174426, 2.562648506546084, 0.04354107631523, -0.35213302043487, -0.830919770909526, 0.844909321628059, -1.220831024350479, 3.986910584714679, 0.37076029210908, 1.882336882284186, -0.717911830541678, 1.344938842001387, -1.438456350545391, -0.986451797260489, -0.053964157751195, -1.087150409226134, 1.225048908301962, 1.436416650249616, 0.596688303823735, 2.695675087653405, 2.758274074366521, 0.144486592681683, 0.354289393058017, 0.518913653509454, -1.279711862998166, 0.934675486850821, -1.680772742446488, 2.077655301946989, -0.340320042804529, 1.380847652194455, 0.907495981244646, -0.298530506348269, 0.790586080562763, 0.121717781279984, -2.258819219558223, 1.23174148343957, -0.504448324290972, -1.325441860856361, -0.387224801864427, 0.57282232674711, -2.587786165171701, -0.609013485939227, -0.851130605667208, 1.546041543111712, 3.564531376683454, -1.698814685732657, 0.699870928827425, 0.919871102170196, -0.936587297531207, 0.4324067196386, 0.266324375723443, -1.460645222862078, 0.004827659023947, -0.496551109412593, 0.525520445534879, -4.430507506457238, 1.657056009939369, 1.525099632985614];
  const f = olsHC3([x1, x2, x3], y);
  [-0.476169707396239, 0.596397860830917, 1.589078888881523, -0.354641099298001].forEach((b, a) => near(f.beta[a], b, 1e-10));
  [0.395482065749168, 0.298931471684015, 0.759839330007008, 0.207273851415231].forEach((s, a) => near(f.se[a], s, 1e-10));
  near(f.r2, 0.32749390540719414, 1e-12); assert.equal(f.df, 56);
  [0.123139449485934, 0.123139449485934, 0.123139449485934].forEach((h, a) => near(f.holm[a + 1], h, 1e-9));
});

test("thinnest particles: the noise-free lower quantile, the prediction interval and the share below a limit", () => {
  const R = rng(31), g = normal(R), y = [], se = [];
  for (let i = 0; i < 400; i++) { se.push(0.1); y.push(1 + 0.2 * g() + 0.1 * g()); }
  const t = thinTail(y, se);
  near(t.value, 1 - 1.6449 * 0.2, 0.04); assert.ok(t.ci[0] < t.value && t.value < t.ci[1]);
  near(t.pred[0], 1 - 1.966 * 0.2, 0.05); near(t.pred[1], 1 + 1.966 * 0.2, 0.05);
  const b = t.below(0.7); near(b.share, 0.0668, 0.03); near(b.cpk, 0.5, 0.08); assert.ok(b.ci[0] <= b.share && b.share <= b.ci[1]);
  // the plain 5th percentile of the measured values is lower: the noise widens it
  assert.ok(quantileSorted(Float64Array.from(y).sort(), 0.05) < t.value - 0.01);
  // lognormal: the quantile on the log scale
  const yl = [], sl = []; for (let i = 0; i < 400; i++) { const v = Math.exp(0.3 * g()); yl.push(v * Math.exp(0.05 * g())); sl.push(0.05 * v); }
  near(thinTail(yl, sl, { log: true }).value, Math.exp(-1.6449 * 0.3), 0.05);
});

test("coverage without a threshold: an uncoated group is found when present, and not invented when absent", () => {
  const R = rng(37), g = normal(R), mk = (n, frac) => { const y = [], se = []; for (let i = 0; i < n; i++) { const u = R() < frac; se.push(0.003); y.push((u ? 0.01 + 0.005 * g() : 0.05 + 0.01 * g()) + 0.003 * g()); } return { y, se }; };
  const a = mk(500, 0.3), fa = mixtureCoverage(a.y, a.se, { B: 60 });
  assert.ok(fa.two && Math.abs(fa.uncoated - 0.3) < 0.06 && fa.ci[0] < fa.uncoated && fa.uncoated < fa.ci[1], JSON.stringify(fa.ci));
  const one = [], so = []; for (let i = 0; i < 500; i++) { so.push(0.003); one.push(0.05 + 0.01 * g() + 0.003 * g()); }
  assert.equal(mixtureCoverage(one, so, { B: 20 }).two, false);
  // with the uncoated reference's own values (noise included), the uncoated share and a test of it
  const ref = Array.from({ length: 800 }, () => 0.01 + Math.hypot(0.005, 0.003) * g());
  const b = mk(400, 0.2), fb = mixtureCoverage(b.y, b.se, { ref, B: 60 });
  assert.ok(Math.abs(fb.uncoated - 0.2) < 0.06 && fb.p < 0.001, JSON.stringify({ u: fb.uncoated, p: fb.p }));
  const c = mk(400, 0), fc = mixtureCoverage(c.y, c.se, { ref, B: 30 });
  assert.ok(fc.uncoated < 0.02 && fc.p > 0.05, JSON.stringify({ u: fc.uncoated, p: fc.p }));
});

test("power: samples per group for a difference in the mean and a ratio of spreads", () => {
  assert.equal(nForMean(0.3, 0.1), Math.ceil(2 * ((1.959964 + 0.841621) * 3) ** 2));
  assert.equal(nForSpread(1.5), Math.ceil(1 + ((1.959964 + 0.841621) / Math.log(1.5)) ** 2));
  assert.ok(nForMean(0.3, 0.05) > 3.9 * nForMean(0.3, 0.1));
});

test("coverage without a threshold: a right-skewed coated population is not split into a bulk and a tail", () => {
  const R = rng(41), g = normal(R), y = [], se = [], y2 = [], s2 = [];
  // lognormal coated particles (log-scale CV about 55 %), measured with errors of 0.006
  for (let i = 0; i < 400; i++) { se.push(0.006); y.push(0.05 * Math.exp(0.52 * g()) + 0.006 * g()); }
  const one = mixtureCoverage(y, se, { B: 20 });
  assert.equal(one.two, false, JSON.stringify({ scale: one.scale, D: one.D, b1: one.bic1, b2: one.bic2 }));
  // the same with a quarter of uncoated particles near zero
  for (let i = 0; i < 400; i++) { s2.push(0.006); y2.push(R() < 0.25 ? 0.004 + 0.003 * g() + 0.006 * g() : 0.05 * Math.exp(0.52 * g()) + 0.006 * g()); }
  const two = mixtureCoverage(y2, s2, { B: 40 });
  assert.ok(two.two && Math.abs(two.uncoated - 0.25) < 0.07, JSON.stringify({ two: two.two, u: two.uncoated, D: two.D }));
});
