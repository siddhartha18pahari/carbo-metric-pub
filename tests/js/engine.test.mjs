// Browser engine unit tests:  node --test tests/js
import test from "node:test";
import assert from "node:assert/strict";
import { erf, windowedArea, shapeEval, fitSpectra } from "../../web/js/engine/core.js";
import { readText } from "../../web/js/engine/readers.js";
import { DEFAULT_CONFIG, toIni, fromIni } from "../../web/js/engine/config.js";
import { g6 } from "../../web/js/engine/local-api.js";

const close = (a, b, tol) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`);

test("erf to double precision", () => {
  close(erf(0.5), 0.5204998778130465, 1e-15);
  close(erf(1.0), 0.8427007929497149, 1e-15);
  close(erf(2.4), 0.9993114861033549, 1e-15);
  close(erf(3.0), 0.9999779095030014, 1e-15);
  close(erf(-1.7), -0.9837904585907746, 1e-15);
});

test("area factors match the vendor exports", () => {
  close(windowedArea(1, 1500, 100, "Gauss", 1000, 1800) / 100, 1.0645, 1e-4);
  close(windowedArea(1, 1580, 19, "Lorentz", 1000, 1800) / 19, 1.541, 2e-3);
});

test("Python's %.6g", () => {
  assert.equal(g6(1234567), "1.23457e+06");
  assert.equal(g6(0.000123456), "0.000123456");
  assert.equal(g6(162.468), "162.468");
  assert.equal(g6(10), "10");
  assert.equal(g6(0), "0");
  assert.equal(g6(1e-5), "1e-05");
});

test("INI round trip", () => {
  const c = structuredClone(DEFAULT_CONFIG);
  c.peaks[3].shape = "Lorentz"; c.ratio_num = "D1"; c.auto_shapes = false;
  assert.deepEqual(fromIni(toIni(c)), c);
});

test("LabSpec matrix text: X is the faster-varying column", () => {
  const K = 40, wn = Array.from({ length: K }, (_, k) => 1000 + 20 * k);
  const lines = ["\t\t" + wn.join("\t")];
  for (let iy = 0; iy < 3; iy++) for (let ix = 0; ix < 4; ix++)
    lines.push([-5 + 2 * iy, 10 + 1.5 * ix, ...wn.map((_, k) => 100 * iy + ix + 0.001 * k)].join("\t"));
  const m = readText(lines.join("\n"), "map.txt");
  assert.equal(m.ny, 3); assert.equal(m.nx, 4);
  assert.deepEqual(Array.from(m.x), [10, 11.5, 13, 14.5]);
  close(m.cube[(2 * 4 + 1) * K], 201, 1e-4);
});

test("x / y / shift / intensity table, shift descending", () => {
  const rows = ["#X\t#Y\t#Wave\t#Intensity"];
  for (let iy = 0; iy < 2; iy++) for (let ix = 0; ix < 3; ix++)
    for (let k = 0; k < 30; k++) rows.push([ix * 0.5, iy * 0.5, 1800 - k * 800 / 29, 10 * iy + ix].join("\t"));
  const m = readText(rows.join("\n"), "map.txt");
  assert.equal(m.ny, 2); assert.equal(m.nx, 3); assert.ok(m.wn[0] < m.wn[29]);
  assert.equal(m.cube[(1 * 3 + 2) * 30], 12);
});

test("two-column spectrum with labels, and decimal commas", () => {
  const wn = Array.from({ length: 200 }, (_, k) => 1000 + 4 * k);
  const y = wn.map(v => 20 + 80 * shapeEval(v, 1580, 20, "Lorentz") + 30 * shapeEval(v, 1350, 60, "Lorentz"));
  const m = readText("Raman shift (cm-1),Intensity\n" + wn.map((v, k) => `${v},${y[k]}`).join("\n"), "s.csv");
  assert.equal(m.nx * m.ny, 1);
  const r = fitSpectra(m.cube, 1, m.wn, structuredClone(DEFAULT_CONFIG));
  close(r.position[0], 1580, 1);
  const s = readText(wn.map((v, k) => `${v.toFixed(2)};${y[k].toFixed(3)}`.replaceAll(".", ",")).join("\n"), "s.csv");
  close(s.wn[0], 1000, 1e-9);
});

test("point lists are refused, not guessed", () => {
  const wn = Array.from({ length: 20 }, (_, k) => 1000 + 40 * k);
  const lines = ["\t\t" + wn.join("\t"), ...[[0, 0], [0, 1], [1, 0]].map(([y, x]) => [y, x, ...wn.map(() => 1)].join("\t"))];
  assert.throws(() => readText(lines.join("\n"), "p.txt"), /full/);
});

// ---------------------------------------------------------------- fitted-result tables
import { isResultsTable, readResultsTable, inferGrid, placeResults } from "../../web/js/engine/readers.js";

function smoothField(ny, nx) {
  // a blob plus a gentle slope: isotropic, like a particle on a substrate
  return Float64Array.from({ length: ny * nx }, (_, p) => {
    const i = Math.floor(p / nx), j = p % nx;
    return 100 * Math.exp(-((i - ny / 2) ** 2 + (j - nx / 3) ** 2) / 60) + 0.2 * i;
  });
}

test("Q-Carbon export: recognised, bands and ratio read, grid inferred", () => {
  const ny = 23, nx = 31, g = smoothField(ny, nx);
  const lines = ["\tD1 Intensity\tD1 Width\tD1 Area\tG Intensity\tG Width\tG Area\tRatio Intensity\tRatio Width\tRatio Area\t"];
  g.forEach((v, p) => lines.push([p + 1, v / 3, 90, v * 50, v, 20, v * 30, 1 / 3, 4.5, 1.6].join("\t")));
  const text = lines.join("\n");
  assert.ok(isResultsTable(text));
  const t = readResultsTable(text, "export.txt");
  assert.deepEqual(t.bands, ["D1", "G"]);
  assert.deepEqual(t.ratio, ["D1", "G"]);
  assert.equal(t.n, ny * nx);
  const best = inferGrid(t.n, t.cols.G.intensity)[0];
  assert.deepEqual([best.ny, best.nx], [ny, nx]);
  const placed = placeResults(t, best);
  close(placed.fit.intensity[(5 * nx + 7) * 2 + 1], g[5 * nx + 7], 1e-9);   // row-major, X fastest
});

test("this app's full CSV reopens on its own X / Y grid", () => {
  const rows = ["X (um),Y (um),Valid,G Intensity,G Position,G Width,G Area,D3 Intensity,D3 Position,D3 Width,D3 Area,Ratio Intensity D3/G,Std error"];
  for (const [x, y] of [[1, 0], [0, 0], [1, 1], [0, 1]]) rows.push([x, y, 1, 10 * y + x, 1580, 20, 300, 1, 1500, 120, 100, 0.1, 2].join(","));
  const text = rows.join("\n");
  assert.ok(isResultsTable(text));
  const t = readResultsTable(text, "full.csv");
  assert.deepEqual(t.ratio, ["D3", "G"]);
  const placed = placeResults(t, null);
  assert.equal(placed.nx, 2); assert.equal(placed.ny, 2);
  assert.deepEqual(Array.from({ length: 4 }, (_, p) => placed.fit.intensity[p * 2]), [0, 1, 10, 11]);
});

test("spectra are not mistaken for result tables", () => {
  assert.ok(!isResultsTable("\t\t1000\t1004\t1008\n0\t0\t1\t2\t3"));
  assert.ok(!isResultsTable("Raman shift (cm-1),Intensity\n1000,5"));
});

// ---------------------------------------------------------------- smoothing
import { smoothSpectra } from "../../web/js/engine/core.js";

test("Savitzky-Golay keeps polynomials up to its order, edges included", () => {
  const K = 60, cube = Float64Array.from({ length: K }, (_, k) => 3 + 0.5 * k - 0.02 * k * k + 1e-4 * k ** 3);
  const out = smoothSpectra(cube, 1, 1, K, { smooth_window: 9, smooth_order: 3, spatial_kernel: 0 });
  for (let k = 0; k < K; k++) close(out[k], cube[k], 1e-8);
});

test("spatial averaging keeps a flat map flat at the edges", () => {
  const ny = 5, nx = 7, K = 3, cube = new Float32Array(ny * nx * K).fill(4.5);
  const out = smoothSpectra(cube, ny, nx, K, { smooth_window: 0, spatial_kernel: 5 });
  for (const v of out) close(v, 4.5, 1e-12);
});
