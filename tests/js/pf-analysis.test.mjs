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
