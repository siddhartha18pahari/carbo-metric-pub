// Small shared helpers: fetch, base64 arrays, number formatting, statistics.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "style") e.style.cssText = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, "");
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(k));
  return e;
}

// Without a Python server (the static web build) the same calls are answered
// in the page by engine/local-api.js.
let LOCAL = null;
export function setLocalBackend(b) { LOCAL = b; }
export const isBrowserOnly = () => LOCAL !== null;

export async function api(path, body, opts = {}) {
  if (LOCAL) return LOCAL.call(path, body);
  const init = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const r = await fetch(path, { ...init, ...opts });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch { /* not json */ }
    throw new Error(msg);
  }
  const ct = r.headers.get("content-type") || "";
  if (ct.includes("json")) return r.json();
  if (ct.includes("octet-stream")) return r.arrayBuffer();
  return r.text();
}

export async function apiUpload(files) {
  if (LOCAL) return LOCAL.upload(files);
  const fd = new FormData(); files.forEach(f => fd.append("files", f));
  const r = await fetch("/api/upload", { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json()).detail);
  return r.json();
}

export const videoSrc = id => (LOCAL ? LOCAL.videoURL(id) : `/api/ds/${id}/video.png`);

export function f32(b64) {
  if (ArrayBuffer.isView(b64)) return b64 instanceof Float32Array ? b64 : Float32Array.from(b64);
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

export function download(name, data, type = "text/plain") {
  const blob = data instanceof Blob ? data : new Blob([data], { type });
  const a = el("a", { href: URL.createObjectURL(blob), download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

export function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

// LabSpec writes thousands with a thin space ("2 600"); keep that look.
export function fmt(v, digits = 4) {
  if (v == null || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  let s;
  if (Number.isInteger(v) && a < 1e7) s = String(v);
  else if (a !== 0 && (a < 1e-3 || a >= 1e7)) s = v.toExponential(2);
  else if (a >= 1000) s = v.toFixed(a >= 1e5 ? 0 : 1);
  else s = Number(v.toPrecision(digits)).toString();
  const [i, d] = s.split(".");
  if (s.includes("e")) return s;
  return i.replace(/\B(?=(\d{3})+(?!\d))/g, " ") + (d ? "." + d : "");
}

export function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const span = hi - lo;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n + 0.5) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toPrecision(12));
  return out;
}

export function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const p = (sorted.length - 1) * q, i = Math.floor(p), f = p - i;
  return i + 1 < sorted.length ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[i];
}

// mean, SD (n-1), median, quartiles: same definitions as qcarbon/results.py
export function summary(values) {
  const v = Float64Array.from(values).filter(Number.isFinite).sort();
  const n = v.length;
  if (!n) return { n: 0, mean: NaN, sd: NaN, median: NaN, q1: NaN, q3: NaN };
  let s = 0; for (const x of v) s += x;
  const mean = s / n;
  let ss = 0; for (const x of v) ss += (x - mean) ** 2;
  return { n, mean, sd: n > 1 ? Math.sqrt(ss / (n - 1)) : 0, median: quantile(v, 0.5),
           q1: quantile(v, 0.25), q3: quantile(v, 0.75), min: v[0], max: v[n - 1] };
}

export function otsu(values, bins = 256) {
  let lo = Infinity, hi = -Infinity;
  for (const x of values) if (Number.isFinite(x)) { if (x < lo) lo = x; if (x > hi) hi = x; }
  if (!(hi > lo)) return lo;
  const h = new Float64Array(bins);
  for (const x of values) if (Number.isFinite(x)) h[Math.min(bins - 1, Math.floor((x - lo) / (hi - lo) * bins))]++;
  let total = 0, sum = 0;
  for (let i = 0; i < bins; i++) { total += h[i]; sum += i * h[i]; }
  let w0 = 0, s0 = 0, best = -1, bi = 0;
  for (let i = 0; i < bins; i++) {
    w0 += h[i]; s0 += i * h[i];
    const w1 = total - w0; if (!w0 || !w1) continue;
    const m0 = s0 / w0, m1 = (sum - s0) / w1, v = w0 * w1 * (m0 - m1) ** 2;
    if (v > best) { best = v; bi = i; }
  }
  return lo + (bi + 0.5) / bins * (hi - lo);
}

export function toast(msg, kind = "info", ms = 4200) {
  const host = document.getElementById("toasts");
  const t = el("div", { class: `toast ${kind}` }, msg);
  host.append(t);
  setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 400); }, ms);
}

export function hidpi(canvas, w, h) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(w * dpr));
  canvas.height = Math.max(1, Math.round(h * dpr));
  canvas.style.width = w + "px"; canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}
