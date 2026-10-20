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
