// The server, in the browser. Answers the same /api calls as scarbon/server.py
// so the interface runs unchanged without Python: files are read from the
// user's disk into the page, fits run in Web Workers, nothing is uploaded.
// Splat 3D needs the Python engine: in the web version it is a work in progress.

import { readMap, readL6v, readL6pf, placeSpectraList, PF_LABELS, TEXT_EXT } from "./readers.js";
import { prepareFit, finishFit, lmBatch, smoothSpectra } from "./core.js";
import { DEFAULT_CONFIG, toIni, fromIni } from "./config.js";

const DS = new Map();
let seq = 0;

// ---------------------------------------------------------------- workers
let pool = null;
function workers() {
  if (pool) return pool;
  const n = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));
  pool = [];
  try {
    for (let i = 0; i < n; i++) pool.push(new Worker(new URL("./fit-worker.js", import.meta.url), { type: "module" }));
  } catch { pool = []; }
  return pool;
}

async function parallelLM(st, iters, progress) {
  const { job, Y, TH, P, N, M } = st;
  const ws = workers(), chunk = 384;
  const chunks = [];
  for (let p0 = 0; p0 < P; p0 += chunk) chunks.push([p0, Math.min(P, p0 + chunk)]);
  let done = 0;
  if (!ws.length) {                                  // no workers: main thread, yielding between chunks
    for (const [p0, p1] of chunks) {
      lmBatch(job, Y, TH, iters, p0, p1);
      done += p1 - p0; progress(done / P);
      await new Promise(r => setTimeout(r, 0));
    }
    return;
  }
  let next = 0;
  await new Promise((resolve, reject) => {
    const feed = w => {
      if (next >= chunks.length) return;
      const id = next++, [p0, p1] = chunks[id];
      w.postMessage({ id, job, Y: Y.slice(p0 * N, p1 * N), TH: TH.slice(p0 * M, p1 * M) });
    };
    for (const w of ws) {
      w.onmessage = e => {
        const { id, TH: th, iters: it, error } = e.data;
        if (error) { reject(new Error(error)); return; }
        const [p0, p1] = chunks[id];
        TH.set(th, p0 * M); iters.set(it, p0);
        done += p1 - p0; progress(done / P);
        if (done >= P) resolve(); else feed(w);
      };
      w.onerror = e => reject(new Error(e.message || "fit worker failed"));
      feed(w);
    }
  });
}

// ---------------------------------------------------------------- datasets
function counts(cube) {
  const step = Math.max(1, Math.floor(cube.length / 400000)), s = [];
  let min = Infinity;
  for (let i = 0; i < cube.length; i++) if (cube[i] < min) min = cube[i];
  for (let i = 0; i < cube.length; i += step) if (Number.isFinite(cube[i])) s.push(cube[i]);
  s.sort((a, b) => a - b);
  return [min, s[Math.min(s.length - 1, Math.floor(0.999 * (s.length - 1)))]];
}

export async function videoUrl(v) {
  const c = document.createElement("canvas");
  c.width = v.w; c.height = v.h;
  const img = c.getContext("2d").createImageData(v.w, v.h);
  for (let i = 0, j = 0; i < v.rgb.length; i += 3, j += 4) {
    img.data[j] = v.rgb[i]; img.data[j + 1] = v.rgb[i + 1]; img.data[j + 2] = v.rgb[i + 2]; img.data[j + 3] = 255;
  }
  c.getContext("2d").putImageData(img, 0, 0);
  // a ParticleFinder mosaic is millions of pixels: JPEG keeps it quick
  const big = v.w * v.h > 4e6;
  const blob = await new Promise(r => c.toBlob(r, big ? "image/jpeg" : "image/png", 0.88));
  return URL.createObjectURL(blob);
}

function info(ds) {
  const m = ds.map;
  const d = { id: ds.id, name: m.name, path: m.name, nx: m.nx, ny: m.ny, k: m.wn.length,
              wn: m.wn, x: m.x, y: m.y, format: m.format, counts: counts(m.cube), missing: m.missing || 0,
              laser_nm: m.laser ?? null, settings: m.settings ?? null, scan: m.scan ?? null, axes: m.axes ?? null };
  if (ds.video) {
    const v = ds.video;
    d.video = { w: v.w, h: v.h, x: [v.x[0], v.x[v.x.length - 1]], y: [v.y[0], v.y[v.y.length - 1]] };
  }
  if (m.points) { d.kind = "particles"; d.points = m.points; d.point_labels = PF_LABELS; }
  return d;
}

async function upload(files) {
  const lower = f => f.name.toLowerCase();
  const projects = files.filter(f => lower(f).endsWith(".l6pf"));
  const main = files.find(f => lower(f).endsWith(".l6m")) || files.find(f => TEXT_EXT.some(e => lower(f).endsWith(e))) || projects[0];
  if (!main) throw new Error(`open a .l6m map (optionally with its .l6v video), a ParticleFinder .l6pf or a text export (${TEXT_EXT.join(", ")})`);
  let map, pfVideo = null;
  if (lower(main).endsWith(".l6pf")) ({ map, video: pfVideo } = readL6pf(await main.arrayBuffer(), main.name));
  else {
    try { map = readMap(await main.arrayBuffer(), main.name); }
    catch (e) {
      // a ParticleFinder spectra list: positions come from the project dropped with it
      if (!e.pointList) throw e;
      if (!projects.length) throw new Error(`${e.message}. For a ParticleFinder list, drop the .l6m together with its .l6pf, or the .l6pf alone.`);
      const pf = readL6pf(await projects[0].arrayBuffer(), projects[0].name);
      map = placeSpectraList(e, pf, main.name); pfVideo = pf.video;
    }
  }
  const ds = { id: `local-${++seq}`, map, video: null, videoUrl: null, fit: null, fitCfg: null, job: { state: "idle" } };
  if (pfVideo) { ds.video = pfVideo; ds.videoUrl = await videoUrl(pfVideo); }
  else if (lower(main).endsWith(".l6m")) {
    const stemOf = f => lower(f).replace(/\.[^.]+$/, "");
    const vids = files.filter(f => lower(f).endsWith(".l6v"));
    const vf = vids.find(f => stemOf(f) === stemOf(main)) || (vids.length === 1 ? vids[0] : null);
    if (vf) {
      try { ds.video = readL6v(await vf.arrayBuffer(), vf.name); ds.videoUrl = await videoUrl(ds.video); }
      catch { ds.video = null; }
    }
  }
  DS.set(ds.id, ds);
  return info(ds);
}

function getDs(id) {
  const ds = DS.get(id);
  if (!ds) throw new Error("dataset not open");
  return ds;
}

// ---------------------------------------------------------------- fitting
// One fit at a time: the worker pool is shared, so two fits running together
// would hand each other's chunks to the wrong job. A fit asked for while
// another runs waits its turn, reported as running at 0 %.
let fitChain = Promise.resolve();
function queueFit(ds, cfg) {
  ds.job = { state: "running", progress: 0, queued: true, started: Date.now() };
  fitChain = fitChain.then(() => runFit(ds, cfg));
}

async function runFit(ds, cfg) {
  const job = ds.job = { state: "running", progress: 0, started: Date.now() };
  try {
    await new Promise(r => setTimeout(r, 0));          // let the status paint
    const t0 = performance.now(), m = ds.map, P = m.nx * m.ny;
    // neighbouring particle numbers are not neighbouring places
    if (m.points && cfg.spatial_kernel) cfg = { ...cfg, spatial_kernel: 0 };
    const cube = cfg.smooth_window || cfg.spatial_kernel ? smoothSpectra(m.cube, m.ny, m.nx, m.wn.length, cfg) : m.cube;
    const st = prepareFit(cube, P, m.wn, cfg);
    job.progress = 0.05;
    const iters = new Int32Array(P);
    await parallelLM(st, iters, f => { job.progress = 0.05 + 0.95 * f; });
    ds.fit = finishFit(st, iters, (performance.now() - t0) / 1000);
    ds.fitCfg = cfg;
    Object.assign(job, { state: "done", progress: 1, seconds: ds.fit.seconds });
  } catch (e) {
    Object.assign(job, { state: "error", error: String(e.message || e) });
  }
}

function payload(ds) {
  return { ...ds.fit, config: ds.fitCfg };
}

// ---------------------------------------------------------------- results and export
const Q = ["intensity", "width", "area"];
const cap = s => s[0].toUpperCase() + s.slice(1);
const admits = (r, v) => (!r.use_lo || v >= r.lo) && (!r.use_hi || v <= r.hi);

function column(fit, name, q) {
  const j = fit.names.indexOf(name), m = fit.m, P = fit.std_error.length, out = new Float64Array(P);
  for (let p = 0; p < P; p++) out[p] = fit[q][p * m + j];
  return out;
}

function derived(fit, cfg) {
  const n = {}, d = {}, r = {};
  for (const q of Q) {
    n[q] = column(fit, cfg.ratio_num, q); d[q] = column(fit, cfg.ratio_den, q);
    r[q] = n[q].map((v, i) => (d[q][i] !== 0 ? v / d[q][i] : 0));
  }
  const valid = n.intensity.map((v, i) => (admits(cfg.intensity, Math.max(v, d.intensity[i])) && admits(cfg.ratio_i, r.intensity[i])
    && admits(cfg.ratio_a, r.area[i]) && Number.isFinite(fit.std_error[i]) ? 1 : 0));
  return { n, d, r, valid };
}

// Python's "%.6g"
export function g6(v) {
  if (Number.isNaN(v)) return "nan";
  if (!Number.isFinite(v)) return v > 0 ? "inf" : "-inf";
  if (v === 0) return "0";
  const [mant, ex] = v.toExponential(5).split("e"), e = +ex;
  const strip = s => (s.includes(".") ? s.replace(/\.?0+$/, "") : s);
  if (e < -4 || e >= 6) return `${strip(mant)}e${e < 0 ? "-" : "+"}${String(Math.abs(e)).padStart(2, "0")}`;
  return strip(v.toFixed(5 - e));
}

export function exportText(ds, body) {
  const cfg = { ...ds.fitCfg };
  for (const k of ["ratio_num", "ratio_den", "intensity", "ratio_i", "ratio_a"]) if (k in body) cfg[k] = body[k];
  const fit = ds.fit, m = ds.map, { n, d, r, valid } = derived(fit, cfg), P = valid.length;
  if (body.format === "full") {
    const pts = m.points;
    const cols = [["X (um)", pts ? Float64Array.from(pts.x) : Float64Array.from({ length: P }, (_, p) => m.x[p % m.nx])],
                  ["Y (um)", pts ? Float64Array.from(pts.y) : Float64Array.from({ length: P }, (_, p) => m.y[Math.floor(p / m.nx)])],
                  ["Valid", Float64Array.from(valid)]];
    if (pts) {
      // ASCII units, like "X (um)": Excel misreads a BOM-less UTF-8 "µ"
      const ascii = u => u.replace("µm²", "um2").replace("µm³", "um3").replace("µm", "um");
      for (const [k, [label, unit]] of Object.entries(PF_LABELS)) cols.push([unit ? `${label} (${ascii(unit)})` : label, Float64Array.from(pts[k])]);
      cols.push(["Class", Float64Array.from(pts.cls)]);
    }
    for (const name of fit.names)
      for (const q of ["intensity", "position", "width", "area"]) cols.push([`${name} ${cap(q)}`, column(fit, name, q)]);
    for (const q of Q) cols.push([`Ratio ${cap(q)} ${cfg.ratio_num}/${cfg.ratio_den}`, r[q]]);
    cols.push(["Std error", fit.std_error], ["Red chi2", fit.red_chi2]);
    const lines = [cols.map(c => c[0]).join(",")];
    for (let p = 0; p < P; p++) lines.push(cols.map(c => g6(c[1][p])).join(","));
    return lines.join("\n") + "\n";
  }
  // vendor layout: numerator's three columns, denominator's, then the ratios
  const cols = [...Q.map(q => [`${cfg.ratio_num} ${cap(q)}`, n[q]]), ...Q.map(q => [`${cfg.ratio_den} ${cap(q)}`, d[q]]),
                ...Q.map(q => [`Ratio ${cap(q)}`, r[q]])];
  const lines = ["\t" + cols.map(c => c[0]).join("\t") + "\t"];
  for (let p = 0; p < P; p++) lines.push(`${p + 1}\t` + cols.map(c => g6(c[1][p])).join("\t"));
  return lines.join("\n") + "\n";
}

// ---------------------------------------------------------------- routing
export function createLocalBackend() {
  async function call(path, body) {
    const url = new URL(path, "http://local");
    const p = url.pathname, q = url.searchParams;
    if (p === "/api/health") return { ok: true, browser: true, splat: false,
      splat_why: "Splat 3D is a work in progress in the web version." };
    if (p === "/api/config/default") return structuredClone(DEFAULT_CONFIG);
    if (p === "/api/config/ini") return toIni(body);
    if (p === "/api/config/parse") {
      try { return fromIni(body.text || ""); } catch (e) { throw new Error(`not a configuration file: ${e.message}`); }
    }
    if (p === "/api/library") return { items: [], open: [...DS.keys()] };
    if (p === "/api/open") throw new Error("the web version opens files from your computer: use Open file…");
    const m = p.match(/^\/api\/ds\/([^/]+)\/(\w+)/);
    if (!m) throw new Error(`unknown request ${p}`);
    const ds = getDs(m[1]);
    switch (m[2]) {
      case "cube": return ds.map.cube.buffer.slice(ds.map.cube.byteOffset, ds.map.cube.byteOffset + ds.map.cube.byteLength);
      case "fit":
        if (body !== undefined) {
          if (ds.job.state === "running") throw new Error("a fit is already running");
          queueFit(ds, structuredClone(body));
          return { ...ds.job };
        }
        return q.get("result") === "1" && ds.fit && ds.job.state === "done" ? { ...ds.job, result: payload(ds) } : { ...ds.job };
      case "export":
        if (!ds.fit) throw new Error("fit the map first");
        return exportText(ds, body);
      case "close":                                   // removed from the file list: free its memory
        if (ds.job.state === "running") throw new Error("a fit is running on this map");
        if (ds.videoUrl) URL.revokeObjectURL(ds.videoUrl);
        DS.delete(ds.id);
        return { ok: true };
      case "splat":
        if (ds.map.points) throw new Error("Splat 3D needs a gridded map; ParticleFinder spectra are one point per particle");
        throw new Error("Splat 3D is a work in progress in the web version");
      default: throw new Error(`unknown request ${p}`);
    }
  }
  return { call, upload, videoURL: id => getDs(id).videoUrl };
}
