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

test("a map stopped early keeps its grid; scattered points are refused", () => {
  const wn = Array.from({ length: 20 }, (_, k) => 1000 + 40 * k);
  const row = (y, x) => [y, x, ...wn.map(() => 1)].join("\t");
  // 3 of a 2 x 2 grid: the last point was never measured
  const m = readText(["\t\t" + wn.join("\t"), row(0, 0), row(0, 1), row(1, 0)].join("\n"), "p.txt");
  assert.equal(m.nx * m.ny, 4); assert.equal(m.missing, 1); assert.ok(Number.isNaN(m.cube[3 * 20]));
  // points that share no grid
  const scattered = [[0.1, 0.7], [3.3, 1.9], [2.2, 4.4], [5.1, 0.2]].map(([y, x]) => row(y, x));
  assert.throws(() => readText(["\t\t" + wn.join("\t"), ...scattered].join("\n"), "s.txt"), /point lists cannot be mapped/);
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

test("S Carbon export: recognised, bands and ratio read, grid inferred", () => {
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

import { pfAngles } from "../../web/js/engine/readers.js";
const near = (v, w, tol, msg) => assert.ok(Math.abs(v - w) <= tol, `${msg ?? ""} ${v} vs ${w}`);

test("ParticleFinder tilt: size from the box and the Feret axes, the way it leans from the mosaic", () => {
  // a 60 x 50 mosaic, X ascending and Y descending as LabSpec saves it, with four bright
  // particles on a dark ground: leaning +30 deg and -30 deg (from +X towards +Y), along X, and round
  const W = 60, H = 50, x = Float32Array.from({ length: W }, (_, q) => q), y = Float32Array.from({ length: H }, (_, r) => H - 1 - r);
  const parts = [[15, 12, 7, 3, Math.PI / 6], [45, 12, 7, 3, -Math.PI / 6], [15, 37, 7, 3, 0], [45, 37, 4, 4, 0]];
  const raw = new Uint8Array(W * H * 3).fill(20);
  for (let r = 0; r < H; r++) for (let q = 0; q < W; q++) for (const [cx, cy, a, b, t] of parts) {
    const ex = x[q] - cx, ey = y[r] - cy, u = (ex * Math.cos(t) + ey * Math.sin(t)) / a, v = (ey * Math.cos(t) - ex * Math.sin(t)) / b;
    if (u * u + v * v <= 1) raw.fill(200, (r * W + q) * 3, (r * W + q) * 3 + 3);
  }
  const pts = { x: [], y: [], feret_max: [], min_diam: [], xmin: [], xmax: [], ymin: [], ymax: [] };
  for (const [cx, cy, a, b, t] of parts) {
    const hw = Math.hypot(a * Math.cos(t), b * Math.sin(t)), hh = Math.hypot(a * Math.sin(t), b * Math.cos(t));
    Object.entries({ x: cx, y: cy, feret_max: 2 * a, min_diam: 2 * b, xmin: cx - hw, xmax: cx + hw, ymin: cy - hh, ymax: cy + hh }).forEach(([k, v]) => pts[k].push(v));
  }
  const ang = pfAngles(pts, { raw, off: 0, W, H, x, y });
  near(ang[0], Math.PI / 6, 1e-6, "leaning +30");
  near(ang[1], -Math.PI / 6, 1e-6, "leaning -30");
  near(ang[2], 0, 1e-6, "along X");
  assert.equal(ang[3], 0);                                 // round: no tilt
  // the same image upside down (Y ascending) still leans the same way on the stage
  const flip = new Uint8Array(raw.length);
  for (let r = 0; r < H; r++) flip.set(raw.subarray((H - 1 - r) * W * 3, (H - r) * W * 3), r * W * 3);
  const up = pfAngles(pts, { raw: flip, off: 0, W, H, x, y: Float32Array.from({ length: H }, (_, r) => r) });
  near(up[0], Math.PI / 6, 1e-6); near(up[1], -Math.PI / 6, 1e-6);
  // without a mosaic only the size of the tilt is known
  near(pfAngles(pts, null)[1], Math.PI / 6, 1e-6);
});

import { pfSeams } from "../../web/js/engine/readers.js";

test("ParticleFinder seams: objects on the dark border between mosaic tiles, and only those", () => {
  // a 120 x 80 mosaic, light with a little texture, a dark seam column at q = 60; X ascending, Y descending
  const W = 120, H = 80, x = Float32Array.from({ length: W }, (_, q) => q), y = Float32Array.from({ length: H }, (_, r) => H - 1 - r);
  const raw = new Uint8Array(W * H * 3);
  for (let r = 0; r < H; r++) for (let q = 0; q < W; q++) raw.fill(q === 60 ? 150 : 200 + ((q * 7 + r * 3) % 9), (r * W + q) * 3, (r * W + q) * 3 + 3);
  const pts = { x: [60, 60.6, 30, 62.6, 90], y: [40, 20, 40, 60, 30], feret_max: [30, 3, 6, 6, 30], min_diam: [2, 2, 5, 5, 2], angle: [Math.PI / 2, 0, 0, 0, Math.PI / 2] };
  // a long thin object along the seam, a speck centred on it, a round particle off it, a round particle
  // across the seam (widened to columns 59-61) but centred 1.6 px beyond it, and a long thin object far from any seam
  assert.deepEqual(Array.from(pfSeams(pts, { raw, off: 0, W, H, x, y })), [1, 1, 0, 0, 0]);
  assert.deepEqual(Array.from(pfSeams(pts, null)), [0, 0, 0, 0, 0]);
});

test("spectra in columns: names from the header, quoted cells, decimal commas", () => {
  const wn = Array.from({ length: 60 }, (_, k) => 1800 - 10 * k);                 // descending, as some programs write
  const s = (a, k) => 100 * a + k;
  const csv = ['"Raman shift","Sample A, spot 1","B"', ...wn.map((v, k) => `${v},${s(1, k)},${s(2, k)}`)].join("\n");
  const m = readText(csv, "set.csv");
  assert.equal(m.layout, "columns"); assert.equal(m.nx, 2); assert.equal(m.ny, 1);
  assert.deepEqual(m.names, ["Sample A, spot 1", "B"]);
  assert.ok(m.wn[0] < m.wn[59]);
  close(m.cube[0], s(1, 59), 1e-4); close(m.cube[60 + 59], s(2, 0), 1e-4);    // channels reversed with the axis
  const t = readText(["shift\tx\ty\tz", ...wn.map((v, k) => [v, 1, 2, 3].map(x => `${x + 0.5 * k}`.replace(".", ",")).join("\t"))].join("\n"), "t.txt");
  assert.equal(t.layout, "columns"); assert.equal(t.nx, 3); assert.deepEqual(t.names, ["x", "y", "z"]);
  close(t.wn[59], 1800, 1e-9); close(t.cube[58], 1.5, 1e-6); close(t.cube[3 * 60 - 2], 3.5, 1e-6);
  const sp = readText(wn.map((v, k) => `${(v + 0.25).toFixed(2).replace(".", ",")} ${(3.5 * k).toFixed(1).replace(".", ",")}`).join("\n"), "w.txt");
  assert.equal(sp.layout, "spectrum"); close(sp.wn[0], 1210.25, 1e-9);
});

test("spectra in columns vs a matrix of spectra in rows: the evener axis wins", () => {
  // 12 spectra in rows (no coordinates), first channel rising down the file by chance
  const wn = Array.from({ length: 30 }, (_, k) => 400 + 7 * k), rows = [wn.join("\t")];
  for (let p = 0; p < 12; p++) rows.push(wn.map((_, k) => (p + 1) ** 2 + k).join("\t"));
  const m = readText(rows.join("\n"), "rows.txt");
  assert.equal(m.layout, "matrix spectra"); assert.equal(m.nx, 12); assert.equal(m.k ?? m.wn.length, 30);
});

test("several single-spectrum files as one set, on the first file's channels", async () => {
  const { combineSpectra } = await import("../../web/js/engine/readers.js");
  const a = Array.from({ length: 50 }, (_, k) => 1000 + 10 * k), b = Array.from({ length: 80 }, (_, k) => 995 + 6.5 * k);
  const line = v => 3 + 0.01 * v;
  const A = readText(a.map(v => `${v},${line(v)}`).join("\n"), "first.csv"), B = readText(b.map(v => `${v}\t${line(v)}`).join("\n"), "second.txt");
  const C = combineSpectra([{ name: "first.csv", map: A }, { name: "second.txt", map: B }]);
  assert.equal(C.n, 2); assert.equal(C.resampled, 1);
  const set = readText(C.csv, "set.csv");
  assert.deepEqual(set.names, ["first", "second"]);
  assert.ok(set.wn[0] >= 1000 && set.wn[set.wn.length - 1] <= 995 + 6.5 * 79);
  for (let k = 0; k < set.wn.length; k++) close(set.cube[set.wn.length + k], line(set.wn[k]), 1e-4);  // a straight line interpolates exactly
});
