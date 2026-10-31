// Peak fit and spectrum processing:  node --test tests/js/peakfit.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { faddeeva, fwhmOf, areaOf, newPeak, compile, fitOne, fitMany, evaluate, findPeaks, MODEL_DEFAULTS, OUT } from "../../web/js/peakfit.js";
import { modpoly, als, snip, rubberband, savgol, normalise, applyOps, interp, evenAxis, trapz } from "../../web/js/spectra.js";
import { erf } from "../../web/js/engine/core.js";

const close = (a, b, tol, what = "") => assert.ok(Math.abs(a - b) <= tol, `${what} ${a} vs ${b}`);
function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function gauss(r) { return () => Math.sqrt(-2 * Math.log(r() + 1e-300)) * Math.cos(2 * Math.PI * r()); }
const model = (peaks, extra = {}) => ({ ...structuredClone(MODEL_DEFAULTS), range: [400, 1200], baseline: { kind: "poly", order: 1 }, despike: false, peaks, ...extra });
const wn = Float64Array.from({ length: 801 }, (_, k) => 400 + k);

test("Faddeeva function: w(iy) = erfcx(y), and a value from the literature", () => {
  for (const y of [0.05, 0.7, 2]) close(faddeeva(0, y)[0], Math.exp(y * y) * (1 - erf(y)), 1e-11, `erfcx(${y})`);
  close(faddeeva(0, 5)[0], 0.1107046377330686, 1e-13, "erfcx(5)");
  const w = faddeeva(1, 1);
  close(w[0], 0.30474420525691259, 1e-13); close(w[1], 0.20821893820283163, 1e-13);
});

test("each line shape: analytic derivatives, area and FWHM", () => {
  const fine = Float64Array.from({ length: 400001 }, (_, k) => 780 - 40000 + 0.2 * k);
  for (const shape of ["gauss", "lorentz", "pv", "voigt", "bwf"]) {
    const pk = newPeak(shape, 780, 100, 30, "A");
    if (pk.eta) pk.eta.v = 0.3; if (pk.wl) pk.wl.v = 12; if (pk.q) pk.q.v = -0.2;
    const C = compile(model([pk]), wn), x = Float64Array.from(C.x0); x[0] = 5; x[1] = 2;
    const f = new Float64Array(C.N), J = new Float64Array(C.N * C.nx);
    evaluate(C, x, f, J);
    for (let i = 0; i < C.nx; i++) {
      const d = 1e-6 * Math.max(1, Math.abs(x[i])), u = Float64Array.from(x), v = Float64Array.from(x); u[i] += d; v[i] -= d;
      const fu = evaluate(C, u, new Float64Array(C.N), null), fv = evaluate(C, v, new Float64Array(C.N), null);
      for (let n = 0; n < C.N; n += 7) close(J[n * C.nx + i], (fu[n] - fv[n]) / (2 * d), 1e-6 * (1 + Math.abs(J[n * C.nx + i])), `${shape} dF/dx${i}`);
    }
    const e = pk.eta ? pk.eta.v : pk.wl ? pk.wl.v : pk.q ? pk.q.v : 0;
    if (shape !== "bwf") {
      // the integral over all shifts: the numerical one over ±40 000 cm⁻¹ plus the Lorentzian tails beyond
      const C2 = compile(model([pk], { range: [fine[0], fine[fine.length - 1]], baseline: { kind: "none" } }), fine);
      const g = evaluate(C2, C2.x0, new Float64Array(C2.N), null);
      const wl = shape === "lorentz" ? 30 : shape === "pv" ? 30 : shape === "voigt" ? 12 : 0, hl = shape === "pv" ? 0.3 * 100 : shape === "lorentz" ? 100 : 0;
      let tail = hl * wl / 2 * 2 * (Math.PI / 2 - Math.atan(2 * 40000 / wl));
      if (shape === "voigt") tail = areaOf("voigt", 100, 780, pk.w.v, 12) * 2 * (Math.PI / 2 - Math.atan(2 * 40000 / 12)) / Math.PI;
      close(trapz(C2.nu, g) + tail, areaOf(shape, 100, 780, pk.w.v, e), 2e-3, `${shape} area`);
    }
    // the FWHM: where the band falls to half its height (found by bisection on the model)
    const C3 = compile(model([pk], { range: [0, 2000], baseline: { kind: "none" } }), Float64Array.from({ length: 2001 }, (_, k) => k));
    const at = v => evaluate({ ...C3, N: 1, nu: Float64Array.of(v), t: Float64Array.of(0) }, C3.x0, new Float64Array(1), null)[0];
    const half = (a, b) => { for (let i = 0; i < 80; i++) { const m = (a + b) / 2; (at(m) > 50) === (at(a) > 50) ? (a = m) : (b = m); } return (a + b) / 2; };
    close(half(780, 900) - half(600, 780), fwhmOf(shape, pk.w.v, e), shape === "voigt" ? 0.0003 * 30 : 1e-6, `${shape} FWHM`);
  }
});

test("three overlapping bands over a sloping background: values within their errors, errors honest", () => {
  const truth = [["lorentz", 700, 300, 18], ["voigt", 760, 150, 24], ["bwf", 900, 220, 60]];
  const make = (r, n) => {
    const peaks = truth.map(([s, c, h, w], j) => newPeak(s, c, h, w, `B${j}`));
    peaks[1].wl.v = 10; peaks[1].w.v = 16; peaks[2].q.v = -0.25;
    const C = compile(model(peaks), wn), x = Float64Array.from(C.x0); x[0] = 50; x[1] = -12;
    const y = evaluate(C, x, new Float64Array(C.N), null);
    return { C, x, y: Float64Array.from(y, v => v + n * r()) };
  };
  const g = gauss(rng(7)), z = [];
  for (let rep = 0; rep < 120; rep++) {
    const { C, x, y } = make(g, 4);
    // start away from the truth: positions off by 4, widths by 20 %
    const start = Float64Array.from(C.x0);
    C.xs.forEach((v, i) => { if (v.kind === "c") start[i] += 4; if (v.kind === "w") start[i] *= 1.2; });
    const r = fitOne(C, y, start);
    assert.equal(r.status, 1);
    const P = ["c", "w"].flatMap(k => C.xs.map((v, i) => [v, i]).filter(([v]) => v.kind === k));
    for (const [, i] of P) z.push((r.fit.x[i] - x[i]) / Math.sqrt(r.fit.cov[i * C.nx + i]));
    if (rep === 0) { close(r.D.vals[0][1], 700, 0.5, "position"); close(r.D.rn, 1, 0.15, "residual / noise"); }
  }
  const m = z.reduce((s, v) => s + v, 0) / z.length, sd = Math.sqrt(z.reduce((s, v) => s + (v - m) ** 2, 0) / (z.length - 1));
  close(m, 0, 0.12, "mean z"); close(sd, 1, 0.12, "SD of z (the errors are the real scatter)");
});

test("links: an equal width and a fixed offset are kept, and the errors follow them", () => {
  const a = newPeak("lorentz", 600, 200, 20, "A"), b = newPeak("lorentz", 640, 120, 20, "B");
  b.w = { v: 20, mode: "link", to: 0, rel: "equal" }; b.c = { v: 640, mode: "link", to: 0, rel: "offset", k: 40 };
  const C = compile(model([a, b]), wn);
  assert.equal(C.nx, 2 + 4);                                     // background 2, A's three, B's height
  const x = Float64Array.from(C.x0); x[0] = 10;
  const truth = evaluate(C, x, new Float64Array(C.N), null), g = gauss(rng(3));
  const r = fitOne(C, Float64Array.from(truth, v => v + 2 * g()));
  close(r.D.vals[1][1] - r.D.vals[0][1], 40, 1e-9, "offset"); close(r.D.vals[1][2], r.D.vals[0][2], 1e-9, "equal widths");
  close(r.D.ses[1][1], r.D.ses[0][1], 1e-9, "B's position error is A's");
});

test("many spectra at once, each from the model in view", () => {
  const pk = newPeak("gauss", 800, 100, 25, "G"), C = compile(model([pk]), wn), N = C.N, n = 6, Y = new Float32Array(n * N), g = gauss(rng(11));
  for (let i = 0; i < n; i++) {
    const x = Float64Array.from(C.x0); x[0] = 20; x[2] = 50 + 20 * i; x[3] = 795 + 2 * i;
    const y = evaluate(C, x, new Float64Array(N), null);
    for (let k = 0; k < N; k++) Y[i * N + k] = y[k] + g();
  }
  Y.fill(NaN, 5 * N, 6 * N);                                    // a point never measured
  const r = fitMany({ model: model([pk]), wn, x0: C.x0 }, Y);
  assert.deepEqual(Array.from(r.status), [1, 1, 1, 1, 1, 0]);
  for (let i = 0; i < 5; i++) close(r.vals[i * OUT.length + 1], 795 + 2 * i, 0.3, `position ${i}`);
});

test("peak search finds the bands and their widths", () => {
  const g = gauss(rng(5)), y = Float64Array.from(wn, v => 30 + 400 / (1 + (2 * (v - 520) / 6) ** 2) + 90 * Math.exp(-4 * Math.LN2 * ((v - 950) / 40) ** 2) + 3 * g());
  const p = findPeaks(wn, y, { prominence: 8 });
  assert.equal(p.length, 2);
  close(p[0].c, 520, 1); close(p[0].w, 6, 1.5); close(p[1].c, 950, 3); close(p[1].w, 40, 6);
});

test("backgrounds: each one follows a curved background under bands", () => {
  const base = Float64Array.from(wn, v => 200 + 0.3 * (v - 400) - 2e-4 * (v - 400) ** 2);
  // Gaussian bands, so the background under them is the only background (a Lorentzian's wings are not)
  const y = Float64Array.from(wn, (v, k) => base[k] + 500 * Math.exp(-4 * Math.LN2 * ((v - 700) / 15) ** 2) + 300 * Math.exp(-4 * Math.LN2 * ((v - 1000) / 25) ** 2));
  const rms = b => Math.sqrt(b.reduce((s, v, k) => s + ((Math.abs(wn[k] - 700) > 60 && Math.abs(wn[k] - 1000) > 80) ? (v - base[k]) ** 2 : 0), 0) / wn.length);
  assert.ok(rms(modpoly(wn, y, 2)) < 0.05, "modified polynomial");
  // with noise of 3 counts the background stays within a tenth of the noise on average (clipping at the polynomial itself sinks it by about 2 SD)
  const g = gauss(rng(5)), yn = Float64Array.from(y, v => v + 3 * g()), bn = modpoly(wn, yn, 2);
  close(bn.reduce((t, v, k) => t + (v - base[k]) / wn.length, 0), 0, 0.3 * 3 / 3 + 0.05, "noisy background offset");
  assert.ok(rms(als(y, 1e6, 0.001)) < 2, "asymmetric least squares");
  assert.ok(rms(snip(y, 30)) < 3, "SNIP");
  const rb = rubberband(wn, y);
  assert.ok(rb.every((v, k) => v <= y[k] + 1e-9), "rubber band stays below the spectrum");
});

test("Savitzky-Golay is exact on polynomials, derivatives included; normalise; operations in order", () => {
  const x = Float64Array.from({ length: 50 }, (_, k) => k * 0.5), y = Float64Array.from(x, v => 3 - 2 * v + 0.5 * v * v);
  const s = savgol(y, 7, 2), d = savgol(y, 7, 2, 1, 0.5), d2 = savgol(y, 9, 3, 2, 0.5);
  for (let k = 0; k < 50; k++) { close(s[k], y[k], 1e-9); close(d[k], -2 + x[k], 1e-9); close(d2[k], 1, 1e-8); }
  close(Math.max(...normalise(x, y, "max")), 1, 1e-12);
  close(trapz(x, normalise(x, y, "area")), 1, 1e-12);
  const out = applyOps(wn, Float64Array.from(wn, v => 2 * v), [{ op: "crop", args: { lo: 500, hi: 600 } }, { op: "resample", args: { step: 2.5 } }, { op: "offset", args: { add: 1, mul: 0.5 } }]);
  assert.equal(out.nu.length, 41); close(out.y[0], 501, 1e-9);
  const sub = applyOps(wn, Float64Array.from(wn, v => 3 * Math.sin(v / 30)), [{ op: "subtract", args: { scale: "fit" } }], { ref: { nu: wn, y: Float64Array.from(wn, v => Math.sin(v / 30)) } });
  close(sub.k, 3, 1e-9, "reference scale");
  assert.deepEqual(Array.from(interp(Float64Array.of(0, 1, 2), Float64Array.of(0, 10, 20), Float64Array.of(0.5, 1.5))), [5, 15]);
  assert.deepEqual(Array.from(evenAxis(Float64Array.of(0.3, 4.1), 1)), [1, 2, 3, 4]);
});

test("cosmic rays: a spike goes, the top of a band eight channels wide stays", async () => {
  const { despikeSpectrum } = await import("../../web/js/spectra.js");
  const g = gauss(rng(9)), x = Float64Array.from({ length: 400 }, (_, k) => 350 + 0.5 * k);
  const clean = Float64Array.from(x, v => 200 + 3000 / (1 + (2 * (v - 520.7) / 4) ** 2) + 8 * g()), y = Float64Array.from(clean);
  y[100] += 2500; y[300] += 900; y[301] += 700;                  // one- and two-channel spikes
  const r = despikeSpectrum(y);
  assert.equal(r.spikes, 2);
  for (let k = 0; k < 400; k++) if (k !== 100 && k !== 300 && k !== 301) assert.equal(r.y[k], y[k], `channel ${k} changed`);
  assert.ok(Math.abs(r.y[100] - clean[100]) < 40 && Math.abs(r.y[300] - clean[300]) < 40);
});
