// File readers for the browser: a port of qcarbon/labspec.py.
//
// LabSpec 6 binaries (.l6m maps, .l6v white-light images) and text exports
// from any program. Everything works on an ArrayBuffer in the page, so files
// never leave the user's computer. Spatial convention as LabSpec draws it: the
// fast (inner) stage index is X, rows follow Y, and Y is drawn downwards.

const TAGS = ["tam", "typm", "uni", "axi", "temx"].map(s => Uint8Array.from(s, c => c.charCodeAt(0)));
const MAT = TAGS[0];
const RECORD = 24;
// spectra recorded through the laser line start in the anti-Stokes region
// (e.g. -1130 to 1939 cm-1); a -300 floor made those maps unreadable
const SHIFT_MIN = -4000, SHIFT_MAX = 4500;
export const TEXT_EXT = [".txt", ".csv", ".tsv", ".dat"];
const TEXT_LAYOUTS = "a LabSpec matrix export (first row = Raman shift, then one row per point), " +
  "an x / y / shift / intensity table, or a two-column spectrum (shift, intensity)";

const stem = name => name.replace(/\.[^.]+$/, "");

function find(raw, pat, from) {
  const last = raw.length - pat.length;
  for (let i = raw.indexOf(pat[0], from); i !== -1 && i <= last; i = raw.indexOf(pat[0], i + 1)) {
    let k = 1;
    while (k < pat.length && raw[i + k] === pat[k]) k++;
    if (k === pat.length) return i;
  }
  return -1;
}

// per-file high half of the serialised pointers, from the bytes after known tags
function pointerWord(raw) {
  const seen = new Map();
  for (const tag of TAGS) {
    for (let t = find(raw, tag, 0); t !== -1; t = find(raw, tag, t + 1)) {
      if (t + 7 > raw.length) continue;
      const w = raw.subarray(t + 3, t + 7);
      if (w[2] === 0 && w[3] === 0 && (w[0] || w[1])) {
        const key = w[0] | (w[1] << 8);
        seen.set(key, (seen.get(key) || 0) + 1);
      }
    }
  }
  let best = null, n = 0;
  for (const [k, c] of seen) if (c > n) { best = k; n = c; }
  return best === null ? null : Uint8Array.of(best & 255, best >> 8, 0, 0);
}

function matBlocks(raw) {
  const word = pointerWord(raw);
  if (!word) return [];
  const recs = new Set();
  for (let i = find(raw, word, 0); i !== -1; i = find(raw, word, i + 1)) if (i >= 12) recs.add(i - 12);
  const sorted = [...recs].sort((a, b) => a - b), out = [];
  for (const p of sorted) {
    if (raw[p + 9] !== MAT[0] || raw[p + 10] !== MAT[1] || raw[p + 11] !== MAT[2]) continue;
    const data = p + RECORD;
    let lo = 0, hi = sorted.length;                        // bisect_right
    while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] <= data) lo = mid + 1; else hi = mid; }
    const end = lo < sorted.length ? sorted[lo] : raw.length;
    if (end - data >= 8) out.push([data, end - data]);
  }
  return out;
}

function f32(raw, off, nbytes) {
  const n = Math.floor(nbytes / 4);
  return new Float32Array(raw.buffer.slice(raw.byteOffset + off, raw.byteOffset + off + 4 * n));
}

function isAxis(a) {
  if (a.length < 2) return false;
  let up = true, down = true;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]) || Math.abs(a[i]) >= 1e6) return false;
    if (i) { const d = a[i] - a[i - 1]; if (!(d > 0)) up = false; if (!(d < 0)) down = false; }
  }
  return up || down;
}

function isStageAxis(a, tol = 0.05) {
  if (!isAxis(a)) return false;
  const n = a.length - 1;
  let mean = 0;
  for (let i = 1; i < a.length; i++) mean += (a[i] - a[i - 1]) / n;
  let v = 0;
  for (let i = 1; i < a.length; i++) v += (a[i] - a[i - 1] - mean) ** 2 / n;
  return Math.abs(mean) > 0 && Math.sqrt(v) <= tol * Math.abs(mean);
}

function magic(raw, name) {
  const m = "LabSpec6";
  for (let i = 0; i < 8; i++) if (raw[i] !== m.charCodeAt(i)) throw new Error(`${name}: not a LabSpec 6 file`);
}

export function readL6m(buf, name) {
  const raw = new Uint8Array(buf);
  magic(raw, name);
  const blocks = matBlocks(raw).map(([o, n]) => [o, f32(raw, o, n)]);
  if (!blocks.length) throw new Error(`${name}: no matrix records (older LabSpec file?)`);
  let [cubeOff, payload] = blocks[0];
  for (const b of blocks) if (b[1].length > payload.length) [cubeOff, payload] = b;
  for (const v of payload) if (!Number.isFinite(v)) throw new Error(`${name}: largest matrix is not spectral data`);

  const axes = blocks.filter(([o, a]) => o !== cubeOff && isAxis(a));
  const spec = axes.filter(([, a]) => a.length >= 16 && SHIFT_MIN < Math.min(a[0], a[a.length - 1])
    && Math.max(a[0], a[a.length - 1]) < SHIFT_MAX && payload.length % a.length === 0);
  if (!spec.length) throw new Error(`${name}: no Raman-shift axis fits the data`);

  const gridFor = (wnOff, K) => {
    const n = payload.length / K;
    // mean counts per spectrum, in file order, for the isotropy score
    const mean = new Float64Array(n);
    for (let p = 0; p < n; p++) { let s = 0; for (let k = 0; k < K; k++) s += payload[p * K + k]; mean[p] = s / K; }
    const aniso = (rows, cols) => {
      let dv = 0, dh = 0;
      for (let i = 0; i < rows - 1; i++) for (let j = 0; j < cols; j++) dv += Math.abs(mean[(i + 1) * cols + j] - mean[i * cols + j]);
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols - 1; j++) dh += Math.abs(mean[i * cols + j + 1] - mean[i * cols + j]);
      dv /= (rows - 1) * cols; dh /= rows * (cols - 1);
      return dh > 0 ? dv / dh : Infinity;
    };
    const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
    const less = (a, b) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i]; return false; };
    const stage = axes.filter(([o, a]) => o !== wnOff && a.length <= n && isStageAxis(a));
    let best = null;
    for (const [oS, aS] of stage) for (const [oF, aF] of stage) {
      if (oS === oF || aS.length * aF.length !== n) continue;
      // some files store the X axis twice: an identical pair is a last resort
      const score = Math.round(Math.abs(Math.log(Math.max(aniso(aS.length, aF.length), 1e-9))) * 1e6) / 1e6;
      const key = [same(aS, aF) ? 1 : 0, score, oF < oS ? 0 : 1];
      if (!best || less(key, best.key)) best = { key, x: aF, y: aS };
    }
    return best;
  };

  // the spectral axis is the one that leaves a consistent X x Y grid (a
  // 91-point X axis divides the payload too)
  let best = null, wn = null;
  for (const [wo, cand] of [...spec].sort((a, b) => b[1].length - a[1].length)) {
    best = gridFor(wo, cand.length);
    if (best) { wn = cand; break; }
  }
  if (!best) {
    wn = spec.reduce((a, b) => (b[1].length > a[1].length ? b : a))[1];
    const n = payload.length / wn.length;
    const err = new Error(`${name}: ${n} spectra with no stage grid. This is a ParticleFinder spectra list: ` +
      "open its .l6pf project, which holds the particle positions");
    Object.assign(err, { pointList: true, spectra: payload, wn, n });
    throw err;
  }
  const K = wn.length, n = payload.length / K;

  let cube = payload, wnOut = Float64Array.from(wn);
  if (wn[0] > wn[K - 1]) {
    cube = new Float32Array(payload.length);
    for (let p = 0; p < n; p++) for (let k = 0; k < K; k++) cube[p * K + k] = payload[p * K + K - 1 - k];
    wnOut.reverse();
  }
  return { cube, wn: wnOut, x: Float64Array.from(best.x), y: Float64Array.from(best.y), nx: best.x.length, ny: best.y.length,
           name: stem(name), format: "l6m" };
}

// white-light camera frame: uint8 RGB (H, W, 3), X axis listed before Y
export function readL6v(buf, name) {
  const raw = new Uint8Array(buf);
  magic(raw, name);
  const blocks = matBlocks(raw), axes = [];
  for (const [o, n] of blocks) { const a = f32(raw, o, n); if (isStageAxis(a, 0.02)) axes.push(Float64Array.from(a)); }
  if (axes.length < 2) throw new Error(`${name}: no stage axes for the video image`);
  const [x, y] = axes, need = x.length * y.length * 3;
  let big = blocks[0];
  for (const b of blocks) if (b[1] > big[1]) big = b;
  if (big[1] < need) throw new Error(`${name}: image block too small`);
  return { rgb: raw.slice(big[0], big[0] + need), w: x.length, h: y.length, x, y };
}

// --------------------------------------------------------------------------
// ParticleFinder projects (.l6pf): mosaic image, one spectrum per particle and
// a table of 24 float32 per particle (see qcarbon/labspec.py, where the field
// meanings were pinned against the instrument's own Results table).
// --------------------------------------------------------------------------
export const PF_FIELDS = { x: 0, y: 1, area: 2, ecd: 3, min_diam: 11, convexity: 15, elongation: 16, volume: 18,
                           xmin: 19, ymin: 20, xmax: 21, ymax: 22, cls: 23 };
export const PF_LABELS = { ecd: ["Diameter (circle equiv.)", "µm"], area: ["Area", "µm²"], min_diam: ["Min diameter", "µm"],
                           convexity: ["Convexity", ""], elongation: ["Elongation", "%"], volume: ["Volume", "µm³"] };
const PF_REC = 24;
const isIndex = a => a[0] === 1 && a.every((v, i) => !i || v - a[i - 1] === 1);

// P consecutive records, found by an identity that cannot hold by chance:
// field 3 (circle-equivalent diameter) = sqrt(4 * field 2 (area) / pi).
// Records are not 4-byte aligned, so every byte alignment is tried; one pass,
// counting the run of valid records separately for each of the 24 phases.
function pfTable(raw, P, lo) {
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength), c = 4 / Math.PI;
  for (let al = 0; al < 4; al++) {
    const start = lo + (((al - lo) % 4) + 4) % 4, nf = Math.floor((raw.length - start) / 4);
    const run = new Int32Array(PF_REC);
    for (let i = 0; i + 3 < nf; i++) {
      const a = dv.getFloat32(start + 4 * (i + 2), true), e = dv.getFloat32(start + 4 * (i + 3), true);
      const r = i % PF_REC;
      run[r] = a > 0 && a < 1e9 && Math.abs(e * e - c * a) <= 1e-3 * a ? run[r] + 1 : 0;
      if (run[r] === P) return start + 4 * (i - PF_REC * (P - 1));
    }
  }
  return -1;
}

export function readL6pf(buf, name) {
  const raw = new Uint8Array(buf);
  magic(raw, name);
  const blocks = matBlocks(raw);
  const arrays = blocks.filter(([, n]) => n < 64 << 20).map(([o, n]) => [o, f32(raw, o, n)]);
  const axes = arrays.filter(([, a]) => isAxis(a));
  const index = axes.filter(([, a]) => isIndex(a));
  if (!index.length) throw new Error(`${name}: no particle index in this ParticleFinder project`);
  const P = Math.max(...index.map(([, a]) => a.length));
  const specs = axes.filter(([, a]) => a.length >= 16 && !isIndex(a) && SHIFT_MIN < Math.min(a[0], a[a.length - 1])
    && Math.max(a[0], a[a.length - 1]) < SHIFT_MAX).sort((a, b) => b[1].length - a[1].length);
  let found = null;
  for (const [wo, wn] of specs) {
    const blk = blocks.find(([, n]) => Math.floor(n / 4) === P * wn.length);
    if (blk) { found = { off: blk[0], wn, wo }; break; }
  }
  if (!found) throw new Error(`${name}: ${P} particles but no matching block of spectra (the project may hold images only)`);
  const K = found.wn.length;
  let spectra = f32(raw, found.off, P * K * 4);

  // mosaic: two evenly stepped stage axes W, H and W*H*3 bytes of RGB, reduced
  // to at most ~4000 px across for the page
  let video = null, imageEnd = 0;
  const stage = axes.filter(([o, a]) => o !== found.wo && a.length > 16 && isStageAxis(a, 0.02) && !isIndex(a));
  for (let i = 0; i + 1 < stage.length && !video; i++) {
    const [xo, xa] = stage[i], [, ya] = stage[i + 1], W = xa.length, H = ya.length, need = W * H * 3;
    let off = (blocks.find(([, n]) => n === need) || [null])[0];
    if (off == null) {          // not tagged as a matrix in some versions: it ends just before the X axis
      const rec = xo - RECORD;
      let best = null;
      for (let st = Math.max(0, rec - need - 1024); st <= rec - need; st++) {
        let sm = 0, cnt = 0;
        for (let r = 0; r < H; r += 16) for (let q = 0; q + 16 < W; q += 16) {
          const o = st + (r * W + q) * 3;
          for (let ch = 0; ch < 3; ch++) sm += Math.abs(raw[o + ch] - raw[o + 3 + ch]);
          cnt++;
        }
        sm /= cnt || 1;
        if (!best || sm < best[0]) best = [sm, st];
      }
      off = best ? best[1] : null;
    }
    if (off == null) continue;
    const f = Math.max(1, Math.ceil(Math.max(W, H) / 4096)), w = Math.ceil(W / f), h = Math.ceil(H / f);
    const rgb = new Uint8Array(w * h * 3);
    for (let r = 0; r < h; r++) for (let q = 0; q < w; q++) {
      const s0 = off + ((r * f) * W + q * f) * 3, d0 = (r * w + q) * 3;
      rgb[d0] = raw[s0]; rgb[d0 + 1] = raw[s0 + 1]; rgb[d0 + 2] = raw[s0 + 2];
    }
    const xs = Float64Array.from({ length: w }, (_, q) => xa[q * f]), ys = Float64Array.from({ length: h }, (_, r) => ya[r * f]);
    video = { rgb, w, h, x: xs, y: ys };
    imageEnd = off + need;
  }

  const t = pfTable(raw, P, imageEnd);
  if (t < 0) throw new Error(`${name}: no particle table with ${P} records`);
  const dv = new DataView(raw.buffer, raw.byteOffset + t, P * PF_REC * 4), points = {};
  for (const [k, c] of Object.entries(PF_FIELDS)) {
    const a = new Float32Array(P);
    for (let p = 0; p < P; p++) a[p] = dv.getFloat32((p * PF_REC + c) * 4, true);
    points[k] = a;
  }
  let inside = 0;
  for (let p = 0; p < P; p++) if (points.x[p] >= points.xmin[p] - 1e-3 && points.x[p] <= points.xmax[p] + 1e-3
    && points.y[p] >= points.ymin[p] - 1e-3 && points.y[p] <= points.ymax[p] + 1e-3) inside++;
  if (inside < 0.95 * P) throw new Error(`${name}: particle table found but positions fall outside their boxes`);

  const wnOut = Float64Array.from(found.wn);
  if (wnOut[0] > wnOut[K - 1]) {
    const flipped = new Float32Array(spectra.length);
    for (let p = 0; p < P; p++) for (let k = 0; k < K; k++) flipped[p * K + k] = spectra[p * K + K - 1 - k];
    spectra = flipped; wnOut.reverse();
  }
  const map = { cube: spectra, wn: wnOut, x: Float64Array.from({ length: P }, (_, p) => p + 1), y: Float64Array.of(0),
                nx: P, ny: 1, name: stem(name), format: "l6pf", kind: "particles", points };
  return { map, video };
}

// a ParticleFinder spectra list (.l6m) placed with the project saved beside it
export function placeSpectraList(err, pf, name) {
  if (pf.map.nx !== err.n) throw new Error(`${name}: ${err.n} spectra but the project has ${pf.map.nx} particles`);
  const K = err.wn.length;
  let cube = err.spectra;
  const wn = Float64Array.from(err.wn);
  if (wn[0] > wn[K - 1]) {
    cube = new Float32Array(err.spectra.length);
    for (let p = 0; p < err.n; p++) for (let k = 0; k < K; k++) cube[p * K + k] = err.spectra[p * K + K - 1 - k];
    wn.reverse();
  }
  return { ...pf.map, cube, wn, name: stem(name), format: "l6m" };
}

// --------------------------------------------------------------------------
// text exports
// --------------------------------------------------------------------------
function grid(xs, ys, I, K, name) {
  const r4 = v => Math.round(v * 1e4) / 1e4;
  const ux = [...new Set(xs.map(r4))].sort((a, b) => a - b), uy = [...new Set(ys.map(r4))].sort((a, b) => a - b);
  if (ux.length * uy.length !== xs.length)
    throw new Error(`${name}: the ${xs.length} points do not form a full ${uy.length} x ${ux.length} grid (point lists cannot be mapped)`);
  const ix = new Map(ux.map((v, i) => [v, i])), iy = new Map(uy.map((v, i) => [v, i]));
  const cube = new Float32Array(ux.length * uy.length * K);
  for (let p = 0; p < xs.length; p++) cube.set(I.subarray(p * K, (p + 1) * K), (iy.get(r4(ys[p])) * ux.length + ix.get(r4(xs[p]))) * K);
  return { cube, x: Float64Array.from(ux), y: Float64Array.from(uy) };
}

function monotonic(a) {
  let up = true, down = true;
  for (let i = 1; i < a.length; i++) { if (!(a[i] > a[i - 1])) up = false; if (!(a[i] < a[i - 1])) down = false; }
  return up || down;
}

export function readText(text, name) {
  const lines = text.replace(/^﻿/, "").split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  const first = lines.find(l => !l.startsWith("#")) || "";
  const sep = first.includes("\t") ? "\t" : first.includes(";") ? ";" : first.includes(",") ? "," : null;
  const rows = [];
  let labels = [];
  for (const ln of lines) {
    let toks = (sep ? ln.split(sep) : ln.split(/\s+/)).map(t => t.trim());
    if (sep === ";") toks = toks.map(t => t.replace(",", "."));
    const vals = [];
    let ok = true;
    for (const t of toks) {
      if (t === "") continue;
      const v = Number(t);
      if (Number.isNaN(v) && !/^[+-]?nan$/i.test(t)) { ok = false; break; }
      vals.push(v);
    }
    if (ok && vals.length) rows.push(Float64Array.from(vals));
    else if (!ok) {
      if (rows.length) throw new Error(`${name}: non-numeric row after the data starts: '${ln.slice(0, 60)}'`);
      labels = toks.map(t => t.replace(/^#/, "").trim().toLowerCase());
    }
  }
  if (!rows.length) throw new Error(`${name}: no numeric data`);

  const done = (cube, wn, x, y, nx, ny, layout) => {
    const K = wn.length;
    if (K < 10) throw new Error(`${name}: only ${K} spectral channels`);
    if (wn[0] > wn[K - 1]) {
      wn = Float64Array.from(wn).reverse();
      const c2 = new Float32Array(cube.length), P = cube.length / K;
      for (let p = 0; p < P; p++) for (let k = 0; k < K; k++) c2[p * K + k] = cube[p * K + K - 1 - k];
      cube = c2;
    }
    for (let k = 1; k < K; k++) if (!(wn[k] > wn[k - 1])) throw new Error(`${name}: the Raman-shift axis is not monotonic`);
    return { cube, wn: Float64Array.from(wn), x: Float64Array.from(x), y: Float64Array.from(y), nx, ny,
             name: stem(name), format: "txt", layout };
  };

  const head = rows[0], widths = new Set(rows.slice(1).map(r => r.length));
  if (head.length >= 10 && widths.size === 1 && [0, 1, 2].includes([...widths][0] - head.length) && monotonic(head)) {
    const L = [...widths][0], nc = L - head.length, K = head.length, P = rows.length - 1;
    const I = new Float32Array(P * K), c0 = new Float64Array(P), c1 = new Float64Array(P);
    rows.slice(1).forEach((r, p) => { if (nc > 0) c0[p] = r[0]; if (nc > 1) c1[p] = r[1]; I.set(r.subarray(nc), p * K); });
    if (nc === 2) {
      // the column that changes more often down the file is X (LabSpec writes Y first)
      let d0 = 0, d1 = 0;
      for (let p = 1; p < P; p++) { if (c0[p] !== c0[p - 1]) d0++; if (c1[p] !== c1[p - 1]) d1++; }
      const [xs, ys] = d1 > d0 ? [c1, c0] : [c0, c1];
      const g = grid([...xs], [...ys], I, K, name);
      return done(g.cube, head, g.x, g.y, g.x.length, g.y.length, "matrix");
    }
    const x = nc === 1 ? c0 : Float64Array.from({ length: P }, (_, i) => i);
    return done(I, head, x, [0], P, 1, nc === 1 ? "matrix line" : "matrix spectra");
  }

  const all = new Set(rows.map(r => r.length));
  if (all.size === 1 && all.has(4)) {
    let col = { x: 0, y: 1, w: 2, i: 3 };
    if (labels.length === 4) {
      const fi = keys => labels.findIndex(s => keys.some(k => s.includes(k)));
      const got = { x: fi(["x"]), y: fi(["y"]), w: fi(["wave", "shift", "raman", "cm"]), i: fi(["intens", "count", "signal"]) };
      if (Object.values(got).every(v => v >= 0) && new Set(Object.values(got)).size === 4) col = got;
    }
    const R = rows.length, key = r => `${Math.round(r[col.x] * 1e4)}|${Math.round(r[col.y] * 1e4)}`;
    const starts = [], seen = new Set();
    for (let i = 0; i < R; i++) { const k = key(rows[i]); if (!seen.has(k)) { seen.add(k); starts.push(i); } }
    const K = R / starts.length;
    if (!Number.isInteger(K) || starts.some((s, i) => s !== i * K))
      throw new Error(`${name}: each point's channels must be on consecutive rows, the same number for every point`);
    const wn = Float64Array.from({ length: K }, (_, k) => rows[k][col.w]);
    const I = new Float32Array(R);
    for (let i = 0; i < R; i++) {
      if (Math.abs(rows[i][col.w] - wn[i % K]) > 1e-3 + 1e-5 * Math.abs(wn[i % K])) throw new Error(`${name}: points do not share one Raman-shift axis`);
      I[i] = rows[i][col.i];
    }
    const g = grid(starts.map(s => rows[s][col.x]), starts.map(s => rows[s][col.y]), I, K, name);
    return done(g.cube, wn, g.x, g.y, g.x.length, g.y.length, "xy table");
  }
  if (all.size === 1 && all.has(2)) {
    return done(Float32Array.from(rows, r => r[1]), Float64Array.from(rows, r => r[0]), [0], [0], 1, 1, "spectrum");
  }
  throw new Error(`${name}: not a layout this reader knows. Expected ${TEXT_LAYOUTS}.`);
}

// --------------------------------------------------------------------------
// fitted-result tables: Q-Carbon's "Export data" (.txt, one row per point,
// "<band> Intensity / Width / Area" columns, no coordinates) and this app's own
// full CSV export (with X, Y and every band)
// --------------------------------------------------------------------------
const QCOL = /^(.*?)\s+(intensity|width|area|position)$/i;

function headerTokens(text) {
  const first = text.replace(/^﻿/, "").split(/\r?\n/).find(l => l.trim()) || "";
  const sep = first.includes("\t") ? "\t" : first.includes(";") ? ";" : ",";
  return { sep, toks: first.split(sep).map(t => t.trim()) };
}

export function isResultsTable(text) {
  const { toks } = headerTokens(text);
  return toks.some(t => { const m = t.match(QCOL); return m && m[2].toLowerCase() === "intensity" && !/^ratio\b/i.test(m[1]); })
    && toks.every(t => t === "" || Number.isNaN(Number(t)));
}

export function readResultsTable(text, name) {
  const { sep, toks } = headerTokens(text);
  const bands = [], cols = {}, idx = {};
  let xi = -1, yi = -1, ratio = null;
  toks.forEach((t, i) => {
    const m = t.match(QCOL);
    if (m && !/^ratio\b/i.test(m[1])) {
      const b = m[1].trim(), q = m[2].toLowerCase();
      if (!idx[b]) { idx[b] = {}; bands.push(b); }
      idx[b][q] = i;
    } else if (/^x\b/i.test(t)) xi = i;
    else if (/^y\b/i.test(t)) yi = i;
    const r = t.match(/^ratio\s+\w+\s+(.+)\/(.+)$/i);
    if (r) ratio = [r[1].trim(), r[2].trim()];
  });
  const usable = bands.filter(b => idx[b].intensity !== undefined);
  if (!usable.length) throw new Error(`${name}: no "<band> Intensity" columns`);
  const rows = text.replace(/^﻿/, "").split(/\r?\n/).filter(l => l.trim()).slice(1)
    .map(l => l.split(sep).map(t => (t.trim() === "" ? NaN : Number(sep === ";" ? t.replace(",", ".") : t))));
  const n = rows.length;
  if (!n) throw new Error(`${name}: no data rows`);
  const col = i => Float64Array.from(rows, r => (i >= 0 && i < r.length ? r[i] : NaN));
  for (const b of usable) cols[b] = Object.fromEntries(["intensity", "width", "area", "position"].map(q => [q, col(idx[b][q] ?? -1)]));
  if (!ratio || !usable.includes(ratio[0]) || !usable.includes(ratio[1])) ratio = usable.length > 1 ? [usable[0], usable[1]] : [usable[0], usable[0]];
  return { name: stem(name), n, bands: usable, cols, ratio,
           x: xi >= 0 && yi >= 0 ? col(xi) : null, y: xi >= 0 && yi >= 0 ? col(yi) : null };
}

// The vendor export has no coordinates: candidate (rows, columns) with
// rows x columns = n, best first. A map read at its true row length is an
// isotropic field; at the wrong one, vertical neighbours are unrelated. Small
// noisy maps make that test weak, so long thin shapes are mildly penalised
// (weight 0.3 on |log aspect|): on 34 exports with a same-size .l6m on disk the
// first choice was right 33 times and the true grid always in the top 3.
export function inferGrid(n, v) {
  const out = [];
  for (let nx = 2; nx <= n / 2; nx++) {
    if (n % nx) continue;
    const ny = n / nx;
    let dv = 0, dh = 0;
    for (let i = 0; i < ny - 1; i++) for (let j = 0; j < nx; j++) dv += Math.abs(v[(i + 1) * nx + j] - v[i * nx + j]);
    for (let i = 0; i < ny; i++) for (let j = 0; j < nx - 1; j++) dh += Math.abs(v[i * nx + j + 1] - v[i * nx + j]);
    dv /= (ny - 1) * nx; dh /= ny * (nx - 1);
    const iso = dh > 0 && dv > 0 ? Math.abs(Math.log(dv / dh)) : Infinity;
    out.push({ ny, nx, score: iso + 0.3 * Math.abs(Math.log(ny / nx)) });
  }
  const wide = out.filter(g => Math.min(g.nx, g.ny) >= 3);
  const ranked = (wide.length ? wide : out).sort((a, b) => a.score - b.score);
  return ranked.length ? ranked : [{ ny: 1, nx: n, score: 0 }];
}

// lay a table out on its grid: from its own X / Y columns when it has them
export function placeResults(t, grid) {
  let order = null, x, y, nx, ny;
  if (t.x) {
    const r4 = v => Math.round(v * 1e4) / 1e4;
    const ux = [...new Set(Array.from(t.x, r4))].sort((a, b) => a - b), uy = [...new Set(Array.from(t.y, r4))].sort((a, b) => a - b);
    if (ux.length * uy.length !== t.n) throw new Error(`${t.name}: the ${t.n} points do not form a full grid`);
    const ix = new Map(ux.map((v, i) => [v, i])), iy = new Map(uy.map((v, i) => [v, i]));
    order = Int32Array.from({ length: t.n }, (_, p) => iy.get(r4(t.y[p])) * ux.length + ix.get(r4(t.x[p])));
    x = Float64Array.from(ux); y = Float64Array.from(uy); nx = ux.length; ny = uy.length;
  } else {
    ({ nx, ny } = grid);
    x = grid.x || Float64Array.from({ length: nx }, (_, i) => i);
    y = grid.y || Float64Array.from({ length: ny }, (_, i) => i);
  }
  const m = t.bands.length, P = t.n;
  const fit = { intensity: new Float64Array(P * m), width: new Float64Array(P * m), area: new Float64Array(P * m), position: new Float64Array(P * m) };
  t.bands.forEach((b, j) => {
    for (const q of Object.keys(fit)) {
      const src = t.cols[b][q];
      for (let p = 0; p < P; p++) fit[q][(order ? order[p] : p) * m + j] = src[p];
    }
  });
  return { nx, ny, x, y, fit };
}

export function readMap(buf, name) {
  const ext = (name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
  if (ext === ".l6m") return readL6m(buf, name);
  if (ext === ".l6pf") return readL6pf(buf, name).map;
  if (TEXT_EXT.includes(ext)) return readText(new TextDecoder("utf-8").decode(buf), name);
  throw new Error(`unsupported file type ${ext || "(none)"}: open a .l6m map, a ParticleFinder .l6pf or a text export (${TEXT_EXT.join(", ")})`);
}
