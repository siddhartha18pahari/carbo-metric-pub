// File readers for the browser: a port of qcarbon/labspec.py.
//
// LabSpec 6 binaries (.l6m maps, .l6v white-light images) and text exports
// from any program. Everything works on an ArrayBuffer in the page, so files
// never leave the user's computer. Spatial convention as LabSpec draws it: the
// fast (inner) stage index is X, rows follow Y, and Y is drawn downwards.

const TAGS = ["tam", "typm", "uni", "axi", "temx"].map(s => Uint8Array.from(s, c => c.charCodeAt(0)));
const MAT = TAGS[0];
const RECORD = 24;
const SHIFT_MIN = -300, SHIFT_MAX = 4500;
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
  let [wnOff, wn] = spec[0];
  for (const s of spec) if (s[1].length > wn.length) [wnOff, wn] = s;
  const K = wn.length, n = payload.length / K;
  if (n < 4) throw new Error(`${name}: ${n} spectra, not a map`);

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
  const stage = axes.filter(([o, a]) => o !== wnOff && a.length <= n && isStageAxis(a));
  let best = null;
  for (const [oS, aS] of stage) for (const [oF, aF] of stage) {
    if (oS === oF || aS.length * aF.length !== n) continue;
    const score = Math.round(Math.abs(Math.log(Math.max(aniso(aS.length, aF.length), 1e-9))) * 1e6) / 1e6;
    const key = [score, oF < oS ? 0 : 1];
    if (!best || key[0] < best.key[0] || (key[0] === best.key[0] && key[1] < best.key[1])) best = { key, x: aF, y: aS };
  }
  if (!best) throw new Error(`${name}: ${n} spectra but no evenly stepped stage-axis pair multiplies to ${n} (point lists and ParticleFinder maps are not grids)`);

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

export function readMap(buf, name) {
  const ext = (name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
  if (ext === ".l6m") return readL6m(buf, name);
  if (TEXT_EXT.includes(ext)) return readText(new TextDecoder("utf-8").decode(buf), name);
  throw new Error(`unsupported file type ${ext || "(none)"}: open a .l6m map or a text export (${TEXT_EXT.join(", ")})`);
}
