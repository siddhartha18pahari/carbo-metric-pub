// Carbo Metrics: browser side. Holds the open map, drives the fit (on the
// Python server, or in the page when there is none), and derives everything
// the instrument shows (maps, ratios, valid points, quality, statistics) plus
// the splat and particle views.

import { smoothSpectra } from "./engine/core.js";
import { $, $$, el, api, apiUpload, videoSrc, setLocalBackend, isBrowserOnly, f32, download, debounce, fmt, summary, otsu, toast, hidpi, quantile, niceTicks } from "./util.js";
import { MapView } from "./mapview.js";
import { SpectrumPlot, HistogramPlot, DensityPlot, PEAK_COLOURS } from "./plots.js";
import { SplatView, sliceField } from "./splat3d.js";
import { segment, outlineOverlay, PARTICLE_COLOURS } from "./particles.js";
import { buildReport } from "./report.js";
import { wilson, weightedQuantile, laCancado, robustScale, bestGmm, bootstrapShares, median, quantileSorted,
         heightCovariances, ratioVariance, heterogeneity, moranI, spearmanCI } from "./pf-analysis.js";
import { T, font, setTheme, currentTheme } from "./theme.js";

const QUANT = { intensity: "Intensity", width: "Width", area: "Area", position: "Position" };
const UNIT = { intensity: "counts", width: "cm⁻¹", area: "cm⁻¹ × counts", position: "cm⁻¹" };
const ERRQ = { std_error: ["Std. error", "counts"], red_chi2: ["Red. χ²", ""], n_iter: ["Iterations", ""], counts: ["Mean counts", "counts"] };

const S = {
  ds: null, cube: null, wn: null, x: null, y: null, nx: 0, ny: 0, k: 0,
  cfg: null, defaultCfg: null, fit: null, ratio: null, valid: null, counts: null,
  specs: [{ kind: "num", q: "intensity" }, { kind: "den", q: "intensity" }, { kind: "ratio", q: "intensity" }, { kind: "error", q: "std_error" }],
  pspec: { kind: "ratio", q: "intensity" },
  active: 0, hover: null, pinned: null, video: null, vidFull: false,
  seg: null, splat: {}, health: {}, fitSeq: 0, fitRunning: false, pendingFit: false, library: [],
};
const P = () => S.nx * S.ny;

// =========================================================================
// derived quantities
// =========================================================================
function col(arr, j) {
  const m = S.fit.m, n = P(), out = new Float32Array(n);
  for (let p = 0; p < n; p++) out[p] = arr[p * m + j];
  return out;
}
function peakCol(name, q) {
  const f = S.fit; if (!f) return null;
  const j = f.names.indexOf(name); if (j < 0) return null;
  f.cols[name] ||= {};
  return (f.cols[name][q] ||= col(f[q], j));
}
function safeRatio(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = b[i] !== 0 ? a[i] / b[i] : 0;
  return out;
}
function admits(r, v) { return (!r.use_lo || v >= r.lo) && (!r.use_hi || v <= r.hi); }

function recomputeDerived() {
  const f = S.fit, c = S.cfg;
  const num = c.ratio_num, den = c.ratio_den;
  if (!f || !f.names.includes(num) || !f.names.includes(den)) { S.ratio = null; S.valid = null; return false; }
  S.ratio = {};
  for (const q of ["intensity", "width", "area"]) S.ratio[q] = safeRatio(peakCol(num, q), peakCol(den, q));
  const In = peakCol(num, "intensity"), Id = peakCol(den, "intensity");
  const v = new Uint8Array(P());
  for (let p = 0; p < v.length; p++) {
    v[p] = admits(c.intensity, Math.max(In[p], Id[p])) && admits(c.ratio_i, S.ratio.intensity[p])
      && admits(c.ratio_a, S.ratio.area[p]) && (f.resultsOnly || Number.isFinite(f.std_error[p])) ? 1 : 0;
  }
  S.valid = v;
  return true;
}

function meanCounts() {
  if (!S.cube) return null;
  if (S.counts) return S.counts;
  const [lo, hi] = S.fit ? S.fit.range : [S.cfg.range_lo, S.cfg.range_hi];
  const ks = []; for (let c = 0; c < S.k; c++) if (S.wn[c] >= lo && S.wn[c] <= hi) ks.push(c);
  const out = new Float32Array(P());
  for (let p = 0; p < out.length; p++) { let s = 0; for (const c of ks) s += S.cube[p * S.k + c]; out[p] = s / ks.length; }
  return (S.counts = out);
}

function peakOf(spec) {
  if (spec.kind === "num") return S.cfg.ratio_num;
  if (spec.kind === "den") return S.cfg.ratio_den;
  return spec.peak;
}

// values + labels for a map spec
function resolve(spec) {
  if (spec.kind === "error") {
    const [label, unit] = ERRQ[spec.q];
    const values = spec.q === "counts" ? meanCounts() : S.fit ? S.fit[spec.q] : null;
    return { values, label, unit, title: `Error : <u>${label}</u>`, name: label, useValid: spec.q !== "counts" };
  }
  if (spec.kind === "ratio") {
    const r = `${S.cfg.ratio_num}/${S.cfg.ratio_den}`;
    return { values: S.ratio ? S.ratio[spec.q] : null, label: `Ratio ${QUANT[spec.q]} ${r}`, unit: "",
             title: `Ratio ${r} : <u>${QUANT[spec.q]}</u>`, name: `Ratio ${QUANT[spec.q]}`, useValid: true };
  }
  const pk = peakOf(spec);
  return { values: peakCol(pk, spec.q), label: spec.q === "intensity" ? "Peak amplitude" : spec.q === "area" ? "Peak area" : `Peak ${spec.q}`,
           unit: UNIT[spec.q], title: `Map ${pk} : <u>${QUANT[spec.q]}</u>`, name: `${pk} ${QUANT[spec.q]}`, useValid: true };
}

// =========================================================================
// maps
// =========================================================================
const maps = [0, 1, 2, 3].map(i => new MapView($(`#map-${i}`), {
  onTitle: (mv, e) => specMenu(e, S.specs[i], () => { renderMap(i); if (S.active === i) { renderHist(); renderVideo(); } }),
  onHover: p => setHover(p),
  onPick: p => { setPinned(p); setActive(i); },
  onChange: () => { if (S.active === i) { renderHist(); renderVideo(); } },
  onView: (v, src) => { for (const m of maps) if (m !== src) m.setView(v ? { ...v } : null); renderVideo(); },
  onSmoothChange: rememberSmoothing,
  onProfile: on => { for (const m of maps) m.setProfile(on); },      // all four maps, as Q-Carbon shows them
}));
maps.forEach((mv, i) => mv.host.addEventListener("mousedown", () => setActive(i)));

// zoom every map onto one particle (or point) and pin its spectrum
function zoomToPoint(p) {
  const [xu, yu] = maps[0].idxXY(p);
  const hw = S.points ? Math.max(8 * S.points.ecd[p], 15) : Math.max(6 * maps[0].dx, 3);
  maps[0].zoomTo(xu, yu, hw);
  setPinned(p);
}
maps[3].cmap = "Inferno";

// D1-D4 are weak bands, noisy pixel to pixel: drawn with a light 3 x 3 filter
// and interpolated pixels by default (display only); G, ratios and the error
// as measured. A map's Smooth/Raw chip or its display menu overrides this,
// remembered per map and band.
const WEAK_BAND = /^D\d/i;
S.smoothPick = {};
function applySmoothing(mv, key, band) {
  mv.smoothKey = key;
  mv.setSmoothing(S.smoothPick[key] || (band && WEAK_BAND.test(band) ? { filter: "gauss3", smooth: true } : { filter: "none", smooth: false }));
}
function rememberSmoothing(mv) { if (mv.smoothKey) S.smoothPick[mv.smoothKey] = { filter: mv.filter, smooth: mv.smooth }; }

function renderMap(i) {
  const mv = maps[i], r = resolve(S.specs[i]), spec = S.specs[i];
  const band = ["num", "den", "peak"].includes(spec.kind) ? peakOf(spec) : null;
  applySmoothing(mv, `${i}:${band || spec.kind}`, band);
  mv.setTitle(r.title);
  mv.emptyText = S.fit && S.fit.resultsOnly && S.specs[i].kind === "error"
    ? "Not in this file: Q-Carbon's export has no error. Fit the map (.l6m) to get it." : null;
  if (r.values) mv.setData(r.values, { valid: r.useValid ? S.valid : null, label: r.label, unit: r.unit });
  else mv.render();
}
function renderMaps() { maps.forEach((_, i) => renderMap(i)); }

function setActive(i) {
  S.active = i;
  maps.forEach((m, k) => m.host.classList.toggle("active", k === i));
  renderHist(); renderVideo(); renderStats();
}

function specMenu(e, spec, done) {
  e.stopPropagation();
  $$(".popover").forEach(p => p.remove());
  const pop = el("div", { class: "popover", onclick: ev => ev.stopPropagation() });
  const close = () => { pop.remove(); done(); };
  if (["num", "den", "peak"].includes(spec.kind) && S.fit) {
    pop.append(el("div", { class: "pop-h" }, "Peak"),
      el("div", { class: "menu-grid" }, ...S.fit.names.map(n => el("button", {
        class: n === peakOf(spec) ? "on" : "",
        onclick: () => {
          if (n === S.cfg.ratio_num) spec.kind = "num"; else if (n === S.cfg.ratio_den) spec.kind = "den"; else { spec.kind = "peak"; spec.peak = n; }
          close();
        },
      }, n))));
  }
  const qs = spec.kind === "error" ? Object.keys(ERRQ) : spec.kind === "ratio" ? ["intensity", "width", "area"] : ["intensity", "width", "area", "position"];
  pop.append(el("div", { class: "pop-h" }, "Show"),
    el("div", { class: "menu-list" }, ...qs.map(q => el("button", {
      class: q === spec.q ? "on" : "", onclick: () => { spec.q = q; close(); },
    }, spec.kind === "error" ? ERRQ[q][0] : QUANT[q]))));
  document.body.append(pop);
  const r = e.currentTarget.getBoundingClientRect();
  pop.style.left = Math.max(8, Math.min(window.innerWidth - 250, r.left)) + "px";
  pop.style.top = r.bottom + 6 + "px";
  setTimeout(() => document.addEventListener("click", () => pop.remove(), { once: true }));
}

// =========================================================================
// cursor spectrum
// =========================================================================
const cursorPlot = new SpectrumPlot($("#plot-cursor"));

function evalFit(p) {
  const f = S.fit, wn = S.wn, n = wn.length;
  const [lo, hi] = f.range, mid = (lo + hi) / 2, half = (hi - lo) / 2;
  const base = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    const t = (wn[c] - mid) / half; let b = 0, tk = 1;
    for (let i = 0; i < f.nb; i++) { b += f.baseline[p * f.nb + i] * tk; tk *= t; }
    base[c] = b;
  }
  const comps = f.names.map((name, j) => {
    const A = f.intensity[p * f.m + j], w = f.width[p * f.m + j];
    let c0 = f.position[p * f.m + j];
    if (!Number.isFinite(c0) && f.nominalPos && f.nominalPos[j] != null) c0 = f.nominalPos[j];   // the file has no positions
    const y = new Float32Array(n);
    for (let c = 0; c < n; c++) {
      if (f.shapes[j] === "Lorentz") { const u = 2 * (wn[c] - c0) / w; y[c] = A / (1 + u * u); }
      else { const v = (wn[c] - c0) / w; y[c] = A * Math.exp(-2.772588722239781 * v * v); }
    }
    return { name, y, A, pos: c0, width: w };
  });
  const total = base.map((b, c) => b + comps.reduce((s, q) => s + q.y[c], 0));
  return { baseline: base, comps, total };
}

// the spectra the last fit actually saw (smoothed when smoothing was on)
function fitCube() {
  const c = S.fit && S.fit.config;
  if (!S.cube || !c || !(c.smooth_window || c.spatial_kernel)) return null;
  const key = `${S.ds.id}/${c.smooth_window}/${c.smooth_order}/${c.spatial_kernel}`;
  if (S.cubeFitKey !== key) { S.cubeFit = Float32Array.from(smoothSpectra(S.cube, S.ny, S.nx, S.k, c)); S.cubeFitKey = key; }
  return S.cubeFit;
}

function renderCursor() {
  const p = S.hover ?? S.pinned;
  cursorPlot.emptyText = S.fit && S.fit.resultsOnly ? "Hover or click a map to see the bands its fitted values describe" : null;
  if (p != null && !S.cube && S.fit && S.fit.resultsOnly) {
    cursorPlot.setData(modelSpectrum(p));
    const i = Math.floor(p / S.nx), j = p % S.nx;
    $("#cursor-info").textContent = `· point ${p + 1} · column ${j + 1}, row ${i + 1}${S.hover != null && S.hover !== S.pinned ? "" : " · pinned"}${S.valid && !S.valid[p] ? " · not valid" : ""}`;
    const tb = $("#cursor-table"); tb.textContent = "";
    // every configured band, so the ones Q-Carbon did not export are visibly absent
    const all = [...new Set([...S.cfg.peaks.map(q => q.name), ...S.fit.names])].sort();
    all.forEach(name => {
      const j2 = S.fit.names.indexOf(name);
      tb.append(j2 < 0 ? el("span", { class: "muted" }, el("b", {}, name), " not in this file")
        : el("span", {}, el("b", { style: `color:${PEAK_COLOURS[j2 % PEAK_COLOURS.length]}` }, name),
            ` I ${fmt(S.fit.intensity[p * S.fit.m + j2], 3)} · w ${fmt(S.fit.width[p * S.fit.m + j2], 3)} · A ${fmt(S.fit.area[p * S.fit.m + j2], 3)}`));
    });
    if (S.ratio) tb.append(el("span", {}, el("b", { style: "color:var(--text)" }, `${S.cfg.ratio_num}/${S.cfg.ratio_den}`), ` I ${fmt(S.ratio.intensity[p], 3)} · A ${fmt(S.ratio.area[p], 3)}`));
    return;
  }
  if (p == null || !S.cube) { cursorPlot.setData(null); $("#cursor-table").textContent = ""; $("#cursor-info").textContent = ""; return; }
  if (!Number.isFinite(S.cube[p * S.k])) {
    cursorPlot.emptyText = "Not measured: the acquisition stopped before this point";
    cursorPlot.setData(null); $("#cursor-table").textContent = "";
    $("#cursor-info").textContent = `· point ${p + 1} · not measured`;
    cursorPlot.emptyText = null;
    return;
  }
  const raw = S.cube.subarray(p * S.k, (p + 1) * S.k), sm = fitCube();
  const d = { wn: S.wn, raw: sm ? sm.subarray(p * S.k, (p + 1) * S.k) : raw, raw0: sm ? raw : null,
              range: S.fit ? S.fit.range : [S.cfg.range_lo, S.cfg.range_hi] };
  if (S.fit) d.fit = evalFit(p);
  cursorPlot.setData(d);
  const i = Math.floor(p / S.nx), j = p % S.nx;
  const where = S.points
    ? `· particle ${p + 1} · X ${fmt(S.points.x[p], 4)} µm, Y ${fmt(S.points.y[p], 4)} µm · ⌀ ${fmt(S.points.ecd[p], 3)} µm`
    : `· point ${p + 1} · X ${fmt(S.x[j], 4)} µm, Y ${fmt(S.y[i], 4)} µm`;
  $("#cursor-info").textContent = `${where}${S.hover != null && S.hover !== S.pinned ? "" : " · pinned"}${S.valid && !S.valid[p] ? " · not valid" : ""}`;
  const tb = $("#cursor-table"); tb.textContent = "";
  if (S.fit) {
    d.fit.comps.forEach((c, j2) => tb.append(el("span", {}, el("b", { style: `color:${PEAK_COLOURS[j2 % PEAK_COLOURS.length]}` }, c.name), ` ${fmt(c.A, 3)} @ ${c.pos.toFixed(1)} w ${c.width.toFixed(1)}`)));
    tb.append(el("span", {}, `σ ${fmt(S.fit.std_error[p], 3)} · χ²ᵣ ${fmt(S.fit.red_chi2[p], 3)}`));
    if (S.ratio) tb.append(el("span", {}, el("b", { style: "color:var(--text)" }, `${S.cfg.ratio_num}/${S.cfg.ratio_den}`), ` I ${fmt(S.ratio.intensity[p], 3)} · A ${fmt(S.ratio.area[p], 3)}`));
  }
}

function setHover(p) {
  S.hover = p;
  maps.forEach(m => m.setCursor(p, S.pinned));
  renderCursor(); renderVideo();
}
function setPinned(p) {
  S.pinned = p; S.hover = null;
  maps.forEach(m => m.setCursor(null, p));
  renderCursor(); renderVideo();
}
document.addEventListener("keydown", e => {
  if (!S.ds || S.pinned == null || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (!$("#tab-qc").classList.contains("on")) return;
  const i = Math.floor(S.pinned / S.nx), j = S.pinned % S.nx;
  const mv = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[e.key];
  if (!mv) return;
  e.preventDefault();
  if (S.points) { setPinned(Math.max(0, Math.min(S.nx - 1, S.pinned + mv[1] + 10 * mv[0]))); return; }
  const di = maps[0].yAsc ? mv[0] : -mv[0], dj = maps[0].xAsc ? mv[1] : -mv[1];
  const ni = Math.max(0, Math.min(S.ny - 1, i + di)), nj = Math.max(0, Math.min(S.nx - 1, j + dj));
  setPinned(ni * S.nx + nj);
});

// =========================================================================
// histogram, quality, statistics
// =========================================================================
const hist = new HistogramPlot($("#plot-hist"));
$("#hist-full").addEventListener("change", () => renderHist());
$("#hist-title").addEventListener("click", e => specMenu(e, S.specs[S.active], () => { renderMap(S.active); renderHist(); renderVideo(); }));

function validValues(r) {
  const out = [];
  if (!r.values) return out;
  for (let p = 0; p < r.values.length; p++) if ((!r.useValid || !S.valid || S.valid[p]) && Number.isFinite(r.values[p])) out.push(r.values[p]);
  return out;
}

function renderHist() {
  if (!S.fit) return;
  const spec = S.specs[S.active], r = resolve(spec);
  $("#hist-title").innerHTML = `Histogram : <u>${r.name}</u> ▾`;
  const vals = validValues(r);
  if (!vals.length) { hist.setData(null); return; }
  const st = summary(vals), sorted = Float64Array.from(vals).sort();
  const range = $("#hist-full").checked ? [st.min, st.max] : [quantile(sorted, 0.005), quantile(sorted, 0.995)];
  if (!(range[1] > range[0])) range[1] = range[0] + 1;
  hist.setData({ values: vals, range, mean: st.mean, median: st.median, label: `${r.name}${r.unit ? ` (${r.unit})` : ""} · n = ${fmt(vals.length)}`, cmap: maps[S.active].cmap });
}

function renderQuality() {
  const q = $("#quality"); q.textContent = "";
  if (!S.fit || !S.valid) return;
  let nv = 0, se = 0, chi = 0;
  for (let p = 0; p < P(); p++) if (S.valid[p]) { nv++; se += S.fit.std_error[p]; chi += S.fit.red_chi2[p]; }
  // points measured: cells of a map stopped early were never acquired
  const total = P() - (S.ds && S.ds.missing || 0);
  const cell = (k, v, cls = "") => el("div", { class: "qcell" }, el("div", { class: "k" }, k), el("div", { class: "v " + cls }, v));
  q.append(cell("Total points", fmt(total)), cell("Valid points", fmt(nv), nv === total ? "" : "warn"),
    cell("Std. error", fmt(nv ? se / nv : NaN, 4)), cell("Red. χ²", fmt(nv ? chi / nv : NaN, 4)));
}

function statRows() {
  if (!S.fit || !S.ratio) return [];
  const rows = [];
  for (const [kind, name] of [["num", S.cfg.ratio_num], ["den", S.cfg.ratio_den]])
    for (const q of ["intensity", "width", "area"]) rows.push({ spec: { kind, q }, label: `${name} ${QUANT[q]}` });
  for (const q of ["intensity", "width", "area"]) rows.push({ spec: { kind: "ratio", q }, label: `Ratio ${QUANT[q]}` });
  return rows.map(r => ({ ...r, st: summary(validValues(resolve(r.spec))) }));
}

function renderStats() {
  const t = $("#stats"); t.textContent = "";
  const rows = statRows(); if (!rows.length) return;
  t.append(el("tr", {}, el("th", {}, "Parameter"), el("th", {}, "Mean"), el("th", {}, "St Dev"), el("th", {}, "Median")));
  const a = S.specs[S.active];
  for (const r of rows) {
    const on = r.spec.kind === a.kind && r.spec.q === a.q;
    t.append(el("tr", {
      class: "click" + (on ? " on" : ""), title: `IQR ${fmt(r.st.q1)} – ${fmt(r.st.q3)} · n = ${r.st.n}`,
      onclick: () => { const slot = r.spec.kind === "ratio" ? 2 : r.spec.kind === "num" ? 0 : 1; S.specs[slot] = { ...r.spec }; renderMap(slot); setActive(slot); },
    }, el("td", {}, r.label), el("td", {}, fmt(r.st.mean)), el("td", {}, fmt(r.st.sd)), el("td", { class: "sub" }, fmt(r.st.median))));
  }
}

// =========================================================================
// optical image
// =========================================================================
const vidHost = $("#video"), vidCanvas = vidHost.querySelector("canvas");
new ResizeObserver(() => renderVideo()).observe(vidHost);
$("#vid-overlay").addEventListener("change", () => renderVideo());
$("#vid-alpha").addEventListener("input", () => renderVideo());
$("#vid-zoom").addEventListener("click", () => { S.vidFull = !S.vidFull; renderVideo(); });

function renderVideo() {
  drawVideo(vidCanvas, vidHost.clientWidth, vidHost.clientHeight, maps[S.active]);
}

function drawVideo(canvas, W, H, mv) {
  if (W < 20 || H < 20) return;
  const ctx = hidpi(canvas, W, H);
  ctx.fillStyle = T.bg; ctx.fillRect(0, 0, W, H);
  if (!S.ds) return;
  const V = S.ds.video;
  if (!V || !S.video) {
    ctx.fillStyle = T.label; ctx.font = font(14); ctx.textAlign = "center";
    ctx.fillText("No white-light image (.l6v) saved with this map", W / 2, H / 2); return;
  }
  const vx0 = Math.min(...V.x), vx1 = Math.max(...V.x), vy0 = Math.min(...V.y), vy1 = Math.max(...V.y);
  let rx0, rx1, ry0, ry1;
  if (S.vidFull) { [rx0, rx1, ry0, ry1] = [vx0, vx1, vy0, vy1]; }
  else if (mv.zoomed()) {
    // the region the maps are zoomed to, with a little context around it
    const mx = (mv.x1 - mv.x0) * 0.15, my = (mv.y1 - mv.y0) * 0.15;
    rx0 = Math.max(vx0, mv.x0 - mx); rx1 = Math.min(vx1, mv.x1 + mx); ry0 = Math.max(vy0, mv.y0 - my); ry1 = Math.min(vy1, mv.y1 + my);
  } else {
    const E = mv.ext, mx = (E.x1 - E.x0) * 0.45, my = (E.y1 - E.y0) * 0.45;
    rx0 = Math.max(vx0, E.x0 - mx); rx1 = Math.min(vx1, E.x1 + mx); ry0 = Math.max(vy0, E.y0 - my); ry1 = Math.min(vy1, E.y1 + my);
  }
  const k = Math.min((W - 8) / (rx1 - rx0), (H - 8) / (ry1 - ry0));
  const ox = (W - (rx1 - rx0) * k) / 2, oy = (H - (ry1 - ry0) * k) / 2;
  const sx = x => ox + (x - rx0) * k, sy = y => oy + (y - ry0) * k;
  ctx.save(); ctx.beginPath(); ctx.rect(ox, oy, (rx1 - rx0) * k, (ry1 - ry0) * k); ctx.clip();
  const dxv = (V.x[1] - V.x[0]) / (V.w - 1), dyv = (V.y[1] - V.y[0]) / (V.h - 1);
  ctx.save();
  ctx.translate(sx(V.x[0] - dxv / 2), sy(V.y[0] - dyv / 2)); ctx.scale(dxv * k, dyv * k);
  ctx.imageSmoothingEnabled = true; ctx.drawImage(S.video, 0, 0);
  ctx.restore();
  if ($("#vid-overlay").checked && mv.values) {
    ctx.globalAlpha = +$("#vid-alpha").value; ctx.imageSmoothingEnabled = mv.smooth;
    const E = mv.ext;
    ctx.drawImage(mv.off, sx(E.x0), sy(E.y0), (E.x1 - E.x0) * k, (E.y1 - E.y0) * k);
    ctx.globalAlpha = 1;
  }
  ctx.setLineDash([5, 4]); ctx.strokeStyle = "#3fe0ff"; ctx.lineWidth = 1.2;
  const F = mv.zoomed() ? mv.view() : mv.ext;           // the zoom window, or the map footprint
  ctx.strokeRect(sx(F.x0), sy(F.y0), (F.x1 - F.x0) * k, (F.y1 - F.y0) * k); ctx.setLineDash([]);
  const p = S.hover ?? S.pinned;
  if (p != null) {
    const [xu, yu] = mv.idxXY(p);
    ctx.strokeStyle = "#7dff6a"; ctx.lineWidth = 1; ctx.beginPath();
    ctx.moveTo(0, sy(yu) + .5); ctx.lineTo(W, sy(yu) + .5); ctx.moveTo(sx(xu) + .5, 0); ctx.lineTo(sx(xu) + .5, H); ctx.stroke();
  }
  ctx.restore();
  ctx.fillStyle = T.tagBg; ctx.fillRect(6, H - 22, 160, 16);
  ctx.fillStyle = T.tagInk; ctx.font = font(11); ctx.textAlign = "left"; ctx.textBaseline = "middle";
  ctx.fillText(`${fmt(rx1 - rx0, 3)} × ${fmt(ry1 - ry0, 3)} µm field`, 10, H - 14);
}

// =========================================================================
// spectral data (all spectra density)
// =========================================================================
const density = new DensityPlot($("#plot-density"));
function renderDensity() {
  if (!S.cube) return;
  density.setData({ cube: S.cube, k: S.k, wn: S.wn, range: [S.cfg.range_lo, S.cfg.range_hi], peaks: S.cfg.peaks });
}

// =========================================================================
// side panel: configuration
// =========================================================================
const NEW_PEAKS = [
  { name: "D5", pos: 1450, pos_min: 1430, pos_max: 1470, width: 60, width_min: 20, width_max: 120, shape: "Gauss", fit: true },
  { name: "P7", pos: 1100, pos_min: 1080, pos_max: 1120, width: 40, width_min: 10, width_max: 100, shape: "Lorentz", fit: true },
  { name: "P8", pos: 1700, pos_min: 1680, pos_max: 1720, width: 40, width_min: 10, width_max: 100, shape: "Lorentz", fit: true },
];

function buildConstraints() {
  const host = $("#constraints"); host.textContent = "";
  for (const [key, label] of [["intensity", "Intensity (cnt)"], ["ratio_i", "Ratio(I)"], ["ratio_a", "Ratio(Ar)"]]) {
    const r = S.cfg[key];
    const mk = (side) => {
      const chk = el("input", { type: "checkbox", checked: r[`use_${side}`], title: `Use the ${side === "lo" ? "lower" : "upper"} limit` });
      const inp = el("input", { type: "number", step: "any", value: r[side] });
      chk.onchange = () => { r[`use_${side}`] = chk.checked; onView(); };
      inp.onchange = () => { const v = parseFloat(inp.value); if (Number.isFinite(v)) { r[side] = v; onView(); } };
      return [chk, inp];
    };
    const [c1, i1] = mk("lo"), [c2, i2] = mk("hi");
    host.append(el("div", { class: "crow" }, el("label", {}, label), c1, i1, c2, i2));
  }
  host.append(el("div", { class: "btnrow", style: "margin-top:6px" },
    el("button", { title: "Lower intensity limit at the Otsu split of log(1 + I) of the stronger ratio band: keeps the particles, drops bare substrate", onclick: excludeSubstrate }, "Exclude substrate (Otsu)"),
    el("button", { title: "All points valid", onclick: () => { for (const k of ["intensity", "ratio_i", "ratio_a"]) S.cfg[k].use_lo = false; S.cfg.ratio_i.use_hi = S.cfg.ratio_a.use_hi = false; buildConstraints(); onView(); } }, "Clear limits")));
}

function excludeSubstrate() {
  const In = peakCol(S.cfg.ratio_num, "intensity"), Id = peakCol(S.cfg.ratio_den, "intensity");
  if (!In || !Id) return toast("Fit the map first", "error");
  const v = new Float32Array(In.length);
  for (let p = 0; p < v.length; p++) v[p] = Math.log1p(Math.max(0, In[p], Id[p]));
  const thr = Math.expm1(otsu(v));
  S.cfg.intensity.lo = +thr.toPrecision(3); S.cfg.intensity.use_lo = true;
  buildConstraints(); onView();
  toast(`Valid points need a ${S.cfg.ratio_num} or ${S.cfg.ratio_den} band of at least ${fmt(S.cfg.intensity.lo)} counts`, "good");
}

function buildPeaks() {
  const t = $("#peaks"); t.textContent = "";
  const names = $("#cfg-names").checked;
  t.append(el("colgroup", {}, el("col", { style: "width:34px" }), ...Array(6).fill(0).map(() => el("col")), el("col", { style: "width:66px" }), el("col", { style: "width:22px" })));
  t.append(el("tr", {}, el("th"), el("th", { class: "grp", colspan: 3 }, "Position"), el("th", { class: "grp", colspan: 3 }, "Width (FWHM)"), el("th"), el("th")));
  t.append(el("tr", {}, el("th"), ...["Pos.", "Min", "Max", "Width", "Min", "Max", "Type", "Fit"].map(h => el("th", {}, h))));
  S.cfg.peaks.forEach((pk, j) => {
    const num = key => {
      const i = el("input", { type: "number", step: "any", value: pk[key] });
      i.onchange = () => { const v = parseFloat(i.value); pk[key] = v; onProcessing(); };
      return el("td", {}, i);
    };
    const shape = el("select", {}, ...["Lorentz", "Gauss"].map(s => el("option", { value: s, selected: s === pk.shape }, s)));
    shape.onchange = () => { pk.shape = shape.value; onProcessing(); };
    const fit = el("input", { type: "checkbox", checked: pk.fit, title: "Unticked: position and width held at Pos./Width" });
    fit.onchange = () => { pk.fit = fit.checked; onProcessing(); };
    let nm;
    if (names) {
      const i = el("input", { class: "name", value: pk.name });
      i.onchange = () => {
        const old = pk.name, v = i.value.trim(); if (!v) { i.value = old; return; }
        pk.name = v;
        if (S.cfg.ratio_num === old) S.cfg.ratio_num = v;
        if (S.cfg.ratio_den === old) S.cfg.ratio_den = v;
        buildRatio(); onProcessing();
      };
      nm = el("td", {}, i);
    } else nm = el("td", { class: "nm" }, el("span", { class: "sw", style: `background:${PEAK_COLOURS[j % PEAK_COLOURS.length]}` }), pk.name);
    t.append(el("tr", {}, nm, num("pos"), num("pos_min"), num("pos_max"), num("width"), num("width_min"), num("width_max"), el("td", {}, shape), el("td", { class: "fit" }, fit)));
  });
}

function buildRatio() {
  for (const id of ["#ratio-num", "#ratio-den"]) {
    const s = $(id), cur = id === "#ratio-num" ? S.cfg.ratio_num : S.cfg.ratio_den;
    s.textContent = "";
    const names = S.fit && S.fit.resultsOnly ? S.fit.names : S.cfg.peaks.map(p => p.name);
    names.forEach(n => s.append(el("option", { value: n, selected: n === cur }, n)));
  }
}

function fillSide() {
  const c = S.cfg;
  $("#cfg-baseline").value = String(c.baseline_order);
  $("#cfg-separate").checked = c.fit_separately;
  $("#cfg-lo").value = c.range_lo; $("#cfg-hi").value = c.range_hi;
  $("#cfg-skip").checked = c.skip_fitting;
  $("#cfg-auto").checked = c.auto_shapes !== false;
  $("#cfg-sgw").value = String(c.smooth_window || 0); $("#cfg-sgo").value = String(c.smooth_order ?? 2);
  $("#cfg-sgo").disabled = !c.smooth_window;
  $("#cfg-spatial").value = String(c.spatial_kernel || 0);
  const np = $("#cfg-npeaks"); np.textContent = "";
  for (let n = 1; n <= 8; n++) np.append(el("option", { value: n, selected: n === c.peaks.length }, n));
  buildConstraints(); buildPeaks(); buildRatio();
}

function validate() {
  const c = S.cfg, errs = [];
  if (!(c.range_hi > c.range_lo)) errs.push("Spectral range: high must exceed low");
  const names = new Set();
  c.peaks.forEach(p => {
    if (names.has(p.name)) errs.push(`Duplicate peak name ${p.name}`); names.add(p.name);
    if (![p.pos, p.pos_min, p.pos_max, p.width, p.width_min, p.width_max].every(Number.isFinite)) errs.push(`${p.name}: empty field`);
    else if (!(p.pos_min <= p.pos && p.pos <= p.pos_max)) errs.push(`${p.name}: Pos. outside its Min–Max`);
    else if (!(0 < p.width_min && p.width_min <= p.width && p.width <= p.width_max)) errs.push(`${p.name}: Width outside its Min–Max`);
  });
  $$("#peaks tr").slice(2).forEach((tr, j) => {
    const p = c.peaks[j]; if (!p) return;
    const ins = tr.querySelectorAll("input[type=number]");
    ins[0].classList.toggle("bad", !(p.pos_min <= p.pos && p.pos <= p.pos_max));
    ins[3].classList.toggle("bad", !(p.width_min <= p.width && p.width <= p.width_max && p.width_min > 0));
  });
  $("#cfg-hi").classList.toggle("bad", !(c.range_hi > c.range_lo));
  return errs;
}

const fitSoon = debounce(() => startFit(), 650);
function onProcessing() {
  const errs = validate();
  if (errs.length) { setStatus(errs[0], "err"); return; }
  setStatus("Settings changed, refitting…", "busy");
  S.counts = null;
  renderDensity();
  fitSoon();
}
function onView() {
  if (!recomputeDerived()) { refreshAll(); return; }
  refreshAll();
}

$("#cfg-baseline").onchange = e => { S.cfg.baseline_order = +e.target.value; onProcessing(); };
$("#cfg-separate").onchange = e => { S.cfg.fit_separately = e.target.checked; onProcessing(); };
$("#cfg-lo").onchange = e => { S.cfg.range_lo = parseFloat(e.target.value); onProcessing(); };
$("#cfg-hi").onchange = e => { S.cfg.range_hi = parseFloat(e.target.value); onProcessing(); };
$("#cfg-skip").onchange = e => { S.cfg.skip_fitting = e.target.checked; onProcessing(); };
$("#cfg-auto").onchange = e => { S.cfg.auto_shapes = e.target.checked; onProcessing(); };
$("#cfg-sgw").onchange = e => { S.cfg.smooth_window = +e.target.value; $("#cfg-sgo").disabled = !S.cfg.smooth_window; onProcessing(); };
$("#cfg-sgo").onchange = e => { S.cfg.smooth_order = +e.target.value; if (S.cfg.smooth_window) onProcessing(); };
$("#cfg-spatial").onchange = e => { S.cfg.spatial_kernel = +e.target.value; onProcessing(); };
$("#cfg-names").onchange = () => buildPeaks();
$("#cfg-npeaks").onchange = e => {
  const n = +e.target.value, c = S.cfg;
  while (c.peaks.length > n) {
    const gone = c.peaks.pop();
    if (gone.name === c.ratio_num) c.ratio_num = c.peaks[0].name;
    if (gone.name === c.ratio_den) c.ratio_den = c.peaks[0].name;
  }
  const used = new Set(c.peaks.map(p => p.name));
  for (const tmpl of [...S.defaultCfg.peaks, ...NEW_PEAKS]) { if (c.peaks.length >= n) break; if (!used.has(tmpl.name)) { c.peaks.push({ ...tmpl }); used.add(tmpl.name); } }
  buildPeaks(); buildRatio(); onProcessing();
};
$("#ratio-num").onchange = e => { S.cfg.ratio_num = e.target.value; onView(); };
$("#ratio-den").onchange = e => { S.cfg.ratio_den = e.target.value; onView(); };
$("#ratio-swap").onclick = () => { [S.cfg.ratio_num, S.cfg.ratio_den] = [S.cfg.ratio_den, S.cfg.ratio_num]; buildRatio(); onView(); };
$("#cfg-save").onclick = async () => {
  try { download("qcarbon_config.ini", await api("/api/config/ini", S.cfg)); } catch (e) { toast(e.message, "error"); }
};
$("#cfg-open").onclick = () => $("#cfg-input").click();
$("#cfg-input").onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  try {
    S.cfg = await api("/api/config/parse", { text: await f.text() });
    $("#cfg-file").value = f.name; fillSide(); onProcessing();
    toast(`Loaded ${f.name}`, "good");
  } catch (err) { toast(err.message, "error"); }
  e.target.value = "";
};
$("#cfg-reset").onclick = () => { S.cfg = structuredClone(S.defaultCfg); $("#cfg-file").value = ""; fillSide(); onProcessing(); };
// folded sections stay folded, per browser
const CLOSED_KEY = "carbo-closed";
let closedSecs = [];
try { closedSecs = JSON.parse(localStorage.getItem(CLOSED_KEY) || "[]"); } catch { closedSecs = []; }
$$(".sec-h.toggle").forEach(h => {
  const sec = h.parentElement, id = sec.dataset.sec;
  if (id && closedSecs.includes(id)) sec.classList.add("closed");
  h.addEventListener("click", () => {
    sec.classList.toggle("closed");
    if (!id) return;
    try { localStorage.setItem(CLOSED_KEY, JSON.stringify($$(".sec[data-sec].closed").map(s => s.dataset.sec))); } catch { /* storage blocked */ }
  });
});

// =========================================================================
// fitting
// =========================================================================
function setStatus(text, kind = "") {
  $("#status-text").textContent = text;
  $("#status").className = "status " + kind;
}
function setProgress(f) {
  const b = $("#fitbar div");
  b.style.opacity = f >= 1 ? 0 : 1; b.style.width = (f * 100).toFixed(1) + "%";
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function startFit() {
  if (!S.ds) return;
  if (S.ds.resultsOnly) {
    // no spectra to refit: spatial smoothing acts on the fitted maps; the
    // spectral settings cannot apply
    if (!S.results || !S.results.raw) return;
    decodeFit(smoothFitted(S.results.raw)); buildRatio(); modelDensity();
    const k = S.cfg.spatial_kernel;
    setStatus(k ? `Fitted maps smoothed over ${k} × ${k} neighbours (this file has no spectra)`
                : "Fitted values from the file (no spectra: spectral settings do not apply)");
    return;
  }
  if (S.fitRunning) { S.pendingFit = true; return; }
  if (validate().length) return;
  // the batch is fitting this map: refit it with these settings when it is done
  if (F.busy === S.ds.id) { S.pendingFit = true; setStatus("Fitting in the batch; refits with these settings after it", "busy"); return; }
  S.fitRunning = true; S.pendingFit = false;
  const id = S.ds.id, f = fileOf(id);
  if (f) { f.state = "fitting"; f.progress = 0; }
  document.body.classList.add("fitting");
  setStatus(`Fitting ${fmt(P())} spectra…`, "busy"); setProgress(0.02);
  try {
    await api(`/api/ds/${id}/fit`, S.cfg);
    let st;
    for (;;) {
      await sleep(140);
      st = await api(`/api/ds/${id}/fit`);
      setProgress(Math.max(0.03, st.progress || 0));
      if (f) { f.progress = st.progress || 0; renderFiles({ compare: false }); }
      if (st.state !== "running") break;
    }
    if (st.state === "error") throw new Error(st.error);
    const r = await api(`/api/ds/${id}/fit?result=1`);
    rememberFit(id, r.result);
    if (!S.ds || S.ds.id !== id) return;
    decodeFit(r.result);
    setStatus(`Fitted ${fmt(P() - (S.ds.missing || 0))} spectra in ${r.result.seconds.toFixed(1)} s`);
  } catch (e) {
    if (f) { f.state = "error"; f.error = e.message; renderFiles(); }
    setStatus("Fit failed", "err"); toast(e.message, "error", 8000);
  } finally {
    setProgress(1); S.fitRunning = false; document.body.classList.remove("fitting");
    if (S.pendingFit) startFit();
  }
}

function decodeFit(r) {
  S.fit = {
    names: r.names, shapes: r.shapes, m: r.m, nb: r.nb, range: r.range, seconds: r.seconds, config: r.config,
    intensity: f32(r.intensity), position: f32(r.position), width: f32(r.width), area: f32(r.area),
    baseline: f32(r.baseline), std_error: f32(r.std_error), red_chi2: f32(r.red_chi2), n_iter: f32(r.n_iter), cols: {},
    held: r.held || [], resultsOnly: !!r.results_only, nominalPos: r.nominal_pos || null,
  };
  $("#held-note").textContent = S.fit.held.length
    ? `Shape held at the mean-spectrum value (one pixel cannot determine it at this noise level): ${S.fit.held.join(", ")}. Only the intensity is fitted per pixel.`
    : "";
  S.counts = null;
  recomputeDerived();
  if (S.pinned == null) S.pinned = brightest();
  refreshAll();
  S.seg = null;
  if ($("#tab-particles").classList.contains("on")) particlesInit();
}

function brightest() {
  const g = peakCol(S.cfg.ratio_den, "intensity") || peakCol(S.fit.names[0], "intensity");
  let best = 0; for (let p = 1; p < g.length; p++) if (g[p] > g[best]) best = p;
  return best;
}

function refreshAll() {
  renderMaps(); renderQuality(); renderStats(); renderHist(); renderCursor(); renderVideo(); renderDensity();
  maps.forEach(m => m.setCursor(S.hover, S.pinned));
  if ($("#tab-particles").classList.contains("on")) renderParticles();
  renderFiles();
}

// =========================================================================
// opening maps
// =========================================================================
async function loadDataset(info) {
  addFile(info);
  S.results = null;
  document.body.classList.remove("results-only");
  $("#grid-row").classList.add("hidden"); $("#results-note").textContent = ""; $("#results-bands").textContent = "";
  S.ds = info; S.nx = info.nx; S.ny = info.ny; S.k = info.k;
  S.wn = f32(info.wn); S.x = f32(info.x); S.y = f32(info.y);
  S.points = info.points ? Object.fromEntries(Object.entries(info.points).map(([k, v]) => [k, f32(v)])) : null;
  S.pointLabels = info.point_labels || null;
  document.body.classList.toggle("particles-mode", !!S.points);
  // neighbouring particle numbers are not neighbouring places
  $("#cfg-spatial").disabled = !!S.points;
  $("#cfg-spatial").closest(".frow").title = S.points ? "Not for ParticleFinder data: each particle has one spectrum, and its neighbours in the list are not its neighbours on the sample" : "";
  S.fit = null; $("#held-note").textContent = ""; S.ratio = null; S.valid = null; S.counts = null; S.seg = null; S.pinned = null; S.hover = null;
  $("#file-name").textContent = info.name;
  $("#spec-name").textContent = S.points ? `${fmt(info.nx)} particles · ${info.k} ch`
    : `${info.nx} × ${info.ny} · ${info.k} ch${info.missing ? ` · ${info.missing} not measured` : ""}`;
  document.title = `${info.name} · Carbo Metrics`;
  $("#welcome").classList.add("hidden"); document.body.classList.remove("no-data");
  setStatus("Loading spectra…", "busy");
  S.cube = new Float32Array(await api(`/api/ds/${info.id}/cube`));
  S.video = null;
  if (info.video) {
    // onload, not img.decode(): decode() can stall while the window is in the background
    const img = new Image();
    const ok = await new Promise(res => { img.onload = () => res(true); img.onerror = () => res(false); img.src = videoSrc(info.id); });
    if (ok) S.video = img;
  }
  for (const m of [...maps, pmap]) { m.setGeometry({ nx: S.nx, ny: S.ny, x: S.x, y: S.y, points: S.points }); m.values = null; }
  fitMapRow();
  S.vidFull = false;
  setActive(S.active);
  renderDensity(); renderVideo();
  splatReset();
  // reopening a map the server already fitted with these settings: reuse it
  const prev = await api(`/api/ds/${info.id}/fit?result=1`).catch(() => null);
  if (prev && prev.state === "done" && prev.result && JSON.stringify(prev.result.config) === JSON.stringify(S.cfg)) {
    rememberFit(info.id, prev.result);
    decodeFit(prev.result);
    setStatus(`Fitted ${fmt(P() - (S.ds.missing || 0))} spectra (kept from earlier)`);
  } else startFit();
}

async function openPath(path) {
  try {
    setStatus("Opening…", "busy");
    await loadDataset(await api("/api/open", { path }));
    $("#library").classList.add("hidden");
  } catch (e) { setStatus("Could not open", "err"); toast(e.message, "error", 8000); }
}

// library
async function showLibrary(scan = false) {
  $("#library").classList.remove("hidden");
  $("#lib-meta").textContent = scan ? "Reading every map once (cached afterwards)…" : "Loading…";
  try {
    const r = await api(`/api/library?scan=${scan ? 1 : 0}`);
    S.library = r.items; S.openIds = new Set(r.open);
    renderLibrary();
  } catch (e) { $("#lib-meta").textContent = e.message; }
  $("#lib-search").focus();
}
function renderLibrary() {
  const q = $("#lib-search").value.toLowerCase().trim();
  const only = $("#lib-ok").checked;
  const items = S.library.filter(it => (!q || (it.name + " " + it.folder).toLowerCase().includes(q)) && !(only && it.ok === false));
  const why = it => /no matrix records/.test(it.why) ? "older file format"
    : /ParticleFinder spectra list/.test(it.why) ? "particle list: open its .l6pf"
    : /stage-axis|no stage grid/.test(it.why) ? "point list, not a grid"
    : /images only/.test(it.why) ? "image only, no spectra" : "not a map";
  const t = $("#lib-table"); t.textContent = "";
  t.append(el("tr", {}, ...["Map", "Folder", "Grid", "Points", "Range (cm⁻¹)", "Video", "Saved"].map(h => el("th", {}, h))));
  for (const it of items) {
    const known = it.ok !== undefined;
    const tr = el("tr", {
      class: "row" + (it.ok === false ? " bad" : "") + (S.ds && S.ds.path === it.path ? " open" : ""),
      title: it.ok === false ? it.why : it.path, onclick: () => it.ok !== false && openPath(it.path),
    },
      el("td", {}, el("b", {}, it.name)),
      el("td", { class: "folder" }, `${it.root} › ${it.folder}`),
      el("td", { class: "num" }, it.ok ? (it.particles ? "particles" : `${it.nx} × ${it.ny}`) : it.ok === false ? el("span", { class: "tag no" }, why(it)) : "–"),
      el("td", { class: "num" }, it.ok ? fmt(it.nx * it.ny) : "–"),
      el("td", { class: "num" }, it.ok ? `${Math.round(it.wn[0])}–${Math.round(it.wn[1])}` : "–"),
      el("td", {}, known && it.ok ? (it.video ? el("span", { class: "tag ok" }, "yes") : el("span", { class: "tag" }, "no")) : ""),
      el("td", { class: "num" }, new Date(it.mtime * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })));
    t.append(tr);
  }
  const nOk = S.library.filter(i => i.ok).length, nUnknown = S.library.filter(i => i.ok === undefined).length;
  $("#lib-meta").textContent = `${items.length} of ${S.library.length} files · ${nOk} readable maps` + (nUnknown ? ` · ${nUnknown} not read yet (Scan details)` : "");
}
$("#btn-library").onclick = e => (F.files.length > 1 ? fileMenu(e) : isBrowserOnly() ? $("#file-input").click() : showLibrary());
$("#hero-open").onclick = () => showLibrary();
$("#lib-close").onclick = () => $("#library").classList.add("hidden");
$("#library").addEventListener("click", e => { if (e.target.id === "library") $("#library").classList.add("hidden"); });
$("#lib-search").addEventListener("input", renderLibrary);
$("#lib-ok").addEventListener("change", renderLibrary);
$("#lib-scan").onclick = () => showLibrary(true);
document.addEventListener("keydown", e => { if (e.key === "Escape") { $("#library").classList.add("hidden"); $$(".maxed").forEach(m => m.classList.remove("maxed")); document.body.classList.remove("has-max"); maps.forEach(m => m.render()); } });

async function renderRecent() {
  try {
    const r = await api("/api/library");
    S.library = r.items;
    const ok = r.items.filter(i => i.ok !== false).slice(0, 6);
    $("#recent").replaceChildren(...ok.map(it => el("button", { onclick: () => openPath(it.path) },
      el("b", {}, it.name), el("span", {}, `${it.root} › ${it.folder}`),
      el("span", {}, it.ok ? `${it.particles ? `${fmt(it.particles)} particles` : `${it.nx} × ${it.ny} points`} · ${new Date(it.mtime * 1000).toLocaleDateString()}` : new Date(it.mtime * 1000).toLocaleDateString()))));
  } catch { /* library optional */ }
}

// fitted-result tables (Q-Carbon export, or our full CSV): no spectra, no server
async function openResultsTable(R, r) {
  S.results = { R, r };
  const row = $("#grid-row"), sel = $("#grid-select");
  sel.textContent = "";
  if (r.choices) {
    r.choices.slice(0, 12).forEach((g, i) => sel.append(el("option", { value: i }, `${g.nx} × ${g.ny}${i === 0 ? "  (most likely)" : ""}`)));
    row.classList.remove("hidden");
  } else row.classList.add("hidden");
  await loadResults(r.grid);
}
async function loadResults(grid) {
  const { R, r } = S.results;
  const { info, fit } = R.resultsDataset(r, grid, S.cfg);
  S.ds = info; S.nx = info.nx; S.ny = info.ny; S.k = 0;
  S.wn = new Float32Array(0); S.x = f32(info.x); S.y = f32(info.y);
  S.points = null; S.pointLabels = null; document.body.classList.remove("particles-mode");
  $("#cfg-spatial").disabled = false; $("#cfg-spatial").closest(".frow").title = "";
  S.cube = null; S.fit = null; S.ratio = null; S.valid = null; S.counts = null; S.seg = null; S.pinned = null; S.hover = null;
  S.cfg.ratio_num = fit.config.ratio_num; S.cfg.ratio_den = fit.config.ratio_den;
  document.body.classList.add("results-only");
  $("#held-note").textContent = "";
  $("#results-note").textContent = r.note;
  // Q-Carbon's export carries only the two bands of its ratio, no error: the
  // rest needs the spectra, so offer the map itself
  const absent = S.cfg.peaks.map(q => q.name).filter(n => !fit.names.includes(n));
  $("#results-bands").replaceChildren(
    el("div", { class: "muted small" }, `This file holds ${fit.names.join(" and ")} only (Q-Carbon exports the two bands of its ratio) and no error.` +
      (absent.length ? ` ${absent.join(", ")} and the error need the spectra.` : " The error needs the spectra.")),
    r.map ? el("button", { onclick: () => uploadFiles(r.map) }, `Fit all bands from ${r.map[0].name}`)
          : el("button", { onclick: () => pickFiles() }, "Open the map (.l6m) to fit all bands…"));
  $("#file-name").textContent = info.name;
  $("#spec-name").textContent = `${info.nx} × ${info.ny} · fitted values, no spectra`;
  document.title = `${info.name} · Carbo Metrics`;
  $("#welcome").classList.add("hidden"); document.body.classList.remove("no-data");
  S.video = null;
  if (r.video) {
    const img = new Image();
    if (await new Promise(res => { img.onload = () => res(true); img.onerror = () => res(false); img.src = r.video.url; })) S.video = img;
  }
  for (const m of [...maps, pmap]) { m.setGeometry({ nx: S.nx, ny: S.ny, x: S.x, y: S.y }); m.values = null; }
  fitMapRow();
  // no spectra in the file: the graphs show the bands its fitted values describe
  S.wn = Float32Array.from({ length: 321 }, (_, c) => S.cfg.range_lo + (S.cfg.range_hi - S.cfg.range_lo) * c / 320);
  setActive(S.active);
  splatReset();
  S.results.raw = fit;
  decodeFit(smoothFitted(fit));
  buildRatio();
  modelDensity();
  $("#spec-name").textContent = `${info.nx} × ${info.ny} · bands rebuilt from fitted values`;
  setStatus(`Opened ${fmt(P())} fitted points (${S.fit.names.join(", ")})` +
    (S.cfg.spatial_kernel ? `, smoothed over ${S.cfg.spatial_kernel} × ${S.cfg.spatial_kernel} neighbours` : ""));
}

// A file of fitted values has no spectra. Its graphs draw what the values
// describe: each band at its exported height and width (line shape read off
// the file's areas, position nominal where the file has none), no baseline.
function modelCaption() {
  const nom = (S.ds && S.ds.nominal) || [];
  return "Rebuilt from the fitted values: this file holds no spectra" +
    (nom.length ? ` (${nom.join(", ")} at the configured position: the file has none)` : "");
}
function modelSpectrum(p) {
  const fit = evalFit(p);
  if (!fit.total.every(Number.isFinite)) return null;          // no fitted values at this point
  return { wn: S.wn, raw: fit.total, range: S.fit.range, fit, modelOnly: true, caption: modelCaption() };
}
function modelDensity() {
  const K = S.wn.length, n = P(), cube = new Float32Array(n * K), top = [];
  for (let p = 0; p < n; p++) {
    const t = evalFit(p).total; cube.set(t, p * K);
    let m = -Infinity; for (const v of t) if (v > m) m = v;
    if (Number.isFinite(m)) top.push(m);
  }
  // no baseline under rebuilt bands, so scale to the peak heights rather
  // than to all channels (which would clip the tallest bands)
  top.sort((a, b) => a - b);
  const hi = top.length ? top[Math.floor((top.length - 1) * 0.995)] * 1.1 : undefined;
  density.setData({ cube, k: K, wn: S.wn, range: S.fit.range, peaks: S.cfg.peaks, hi: hi > 0 ? hi : undefined });
}

// A file of fitted values has no spectra, so spatial smoothing is applied to
// the fitted maps themselves (each band's intensity, position, width, area),
// with the same Gaussian weights over measured neighbours as the pre-fit
// smoothing; ratios are then recomputed from the smoothed bands.
function smoothFitted(fit) {
  const k = S.cfg.spatial_kernel || 0;
  if (!k) return fit;
  const r = (k - 1) / 2, g = Array.from({ length: k }, (_, i) => Math.exp(-0.5 * ((i - r) / (k / 4)) ** 2));
  const { nx, ny } = S, m = fit.m, out = { ...fit };
  for (const q of ["intensity", "position", "width", "area"]) {
    const a = f32(fit[q]), b = new Float64Array(a.length);
    for (let j = 0; j < m; j++) for (let i = 0; i < ny; i++) for (let c = 0; c < nx; c++) {
      let s = 0, w = 0;
      for (let di = -r; di <= r; di++) for (let dc = -r; dc <= r; dc++) {
        const ii = i + di, cc = c + dc;
        if (ii < 0 || ii >= ny || cc < 0 || cc >= nx) continue;
        const v = a[(ii * nx + cc) * m + j];
        if (!Number.isFinite(v)) continue;
        const wt = g[di + r] * g[dc + r]; s += wt * v; w += wt;
      }
      const p = i * nx + c;
      b[p * m + j] = Number.isFinite(a[p * m + j]) && w ? s / w : a[p * m + j];
    }
    out[q] = b;
  }
  return out;
}
$("#grid-select").onchange = e => { if (S.results) loadResults(S.results.r.choices[+e.target.value]); };

// open files: drag & drop or the file picker
async function uploadFiles(files) {
  if (!files.length) return;
  try {
    const { groups, tables } = await mapGroups(files);
    if (groups.length > 1) { await openMany(groups, tables); $("#library").classList.add("hidden"); return; }
    const R = await import("./engine/results-file.js");
    const table = await R.findResultsFile(files);
    if (table) {
      setStatus("Reading…", "busy");
      await openResultsTable(R, await R.openResults(table, files));
      $("#library").classList.add("hidden");
      return;
    }
    setStatus(isBrowserOnly() ? "Reading…" : "Uploading…", "busy");
    await loadDataset(await apiUpload(files));
    $("#library").classList.add("hidden");
  } catch (err) { setStatus("Could not open", "err"); toast(err.message, "error", 9000); }
}
const pickFiles = () => $("#file-input").click();
$("#hero-file").onclick = pickFiles;
$("#lib-file").onclick = pickFiles;
$("#file-input").onchange = e => { const f = [...e.target.files]; e.target.value = ""; uploadFiles(f); };
let dragDepth = 0;
window.addEventListener("dragenter", e => { e.preventDefault(); dragDepth++; $("#drop").classList.remove("hidden"); });
window.addEventListener("dragleave", e => { e.preventDefault(); if (--dragDepth <= 0) { dragDepth = 0; $("#drop").classList.add("hidden"); } });
window.addEventListener("dragover", e => e.preventDefault());
window.addEventListener("drop", async e => {
  e.preventDefault(); dragDepth = 0; $("#drop").classList.add("hidden");
  uploadFiles([...e.dataTransfer.files]);
});

// exports
async function exportData(format) {
  if (!S.fit) return toast("Fit a map first", "error");
  try {
    const body = { format, ratio_num: S.cfg.ratio_num, ratio_den: S.cfg.ratio_den, intensity: S.cfg.intensity, ratio_i: S.cfg.ratio_i, ratio_a: S.cfg.ratio_a };
    const text = S.ds.resultsOnly
      ? (await import("./engine/local-api.js")).exportText({ fit: S.fit, fitCfg: S.cfg, map: { x: S.x, y: S.y, nx: S.nx } }, body)
      : await api(`/api/ds/${S.ds.id}/export`, body);
    download(`${S.ds.name} CarboMetrics ${S.cfg.ratio_num}-${S.cfg.ratio_den}.${format === "full" ? "csv" : "txt"}`, text);
  } catch (e) { toast(e.message, "error"); }
}
$("#exp-txt").onclick = () => exportData("qcarbon");
$("#exp-csv").onclick = () => exportData("full");
$("#exp-png").onclick = () => maps.forEach((m, i) => setTimeout(() => m.save(), i * 250));
$("#exp-report").onclick = () => switchTab("report");

// =========================================================================
// tabs
// =========================================================================
function switchTab(name) {
  $$("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === name));
  $$(".tab").forEach(t => t.classList.toggle("on", t.id === `tab-${name}`));
  moveInk();
  if (name === "splat") splatShow();
  if (name === "particles") particlesInit();
  if (name === "report") makeReport();
  if (name === "files") setTimeout(() => renderFiles(), 0);
  if (name === "qc") setTimeout(() => { fitMapRow(); maps.forEach(m => m.render()); renderVideo(); cursorPlot.render(); hist.render(); density.render(); }, 0);
}
function moveInk() {
  const on = $("#tabs button.on"), ink = $(".tab-ink");
  ink.style.left = on.offsetLeft + "px"; ink.style.width = on.offsetWidth + "px";
}
$$("#tabs button").forEach(b => b.addEventListener("click", () => switchTab(b.dataset.tab)));
window.addEventListener("resize", () => { moveInk(); fitMapRow(); });

// Map cards as tall as the map needs, so a wide map leaves no empty band.
// Side by side, four in a row or 2 x 2, whichever draws each map at least
// 1.5x bigger (wide maps get tiny four across); the row below takes the rest.
// Stacked (narrow screens), the columns are the stylesheet's.
function fitMapRow() {
  const chart = $(".chart"), grid = $(".maps"), mv = maps.find(m => !m.host.classList.contains("maxed")) || maps[0];
  const clear = () => { chart.style.removeProperty("--maps-h"); grid.style.removeProperty("--map-cols"); chart.style.removeProperty("--map-card-h"); };
  if (!S.ds || !mv.ext || !grid.clientWidth) { clear(); return; }
  const head = mv.head.offsetHeight + 2, gap = 12;
  if (matchMedia("(max-width: 1280px)").matches) {
    clear();
    chart.style.setProperty("--map-card-h", Math.round(Math.max(220, Math.min(innerHeight * 0.8, mv.fitBody(mv.body.clientWidth) + head))) + "px");
    return;
  }
  // the maps may take 62% of the height, leaving the row below at least 280 px
  const W = grid.clientWidth, cap = Math.min(chart.clientHeight * 0.62, chart.clientHeight - 280 - gap);
  const layout = cols => {
    const rows = 4 / cols, cw = (W - gap * (cols - 1)) / cols;
    const rowH = Math.max(170, Math.min(mv.fitBody(cw) + head, (cap - gap * (rows - 1)) / rows));
    const s = mv.plotSize(cw, rowH - head);
    return { cols, h: rows * rowH + gap * (rows - 1), area: s.w * s.h };
  };
  const four = layout(4), two = layout(2), L = two.area >= 1.5 * four.area ? two : four;
  grid.style.setProperty("--map-cols", L.cols);
  chart.style.setProperty("--maps-h", Math.round(L.h) + "px");
}

// =========================================================================
// several files: open together, fit as a batch, switch between, compare
// =========================================================================
// Every map opened this session is listed. A batch fits them one after
// another with the Carbon fit tab's settings; the summary and the compare
// maps use each file's own fit, with the current ratio and constraints.
const F = { files: [], busy: null, running: false, dirty: false };
const VIEW_KEYS = ["ratio_num", "ratio_den", "intensity", "ratio_i", "ratio_a"];
function stableJSON(v) {
  if (Array.isArray(v)) return `[${v.map(stableJSON).join(",")}]`;
  if (v && typeof v === "object") return `{${Object.keys(v).sort().map(k => JSON.stringify(k) + ":" + stableJSON(v[k])).join(",")}}`;
  return JSON.stringify(v);
}
// the settings a fit depends on: the ratio and constraints are applied after it
function fitKey(cfg, info) {
  const c = { ...cfg };
  for (const k of VIEW_KEYS) delete c[k];
  if (info && info.points) c.spatial_kernel = 0;        // never applied to particles
  return stableJSON(c);
}
const fileOf = id => F.files.find(f => f.id === id);
const escHTML = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function addFile(info) {
  if (info.resultsOnly || fileOf(info.id)) return;
  F.files.push({ id: info.id, name: info.name, info, result: null, key: null, state: "idle", progress: 0, error: "",
                 compare: F.files.filter(f => f.compare).length < 4 });
  renderFiles();
}
function rememberFit(id, result) {
  const f = fileOf(id); if (!f) return;
  Object.assign(f, { result, key: fitKey(result.config, f.info), state: "done", progress: 1, error: "", dec: null, sum: null });
  renderFiles();
}
function statusOf(f) {
  if (f.state === "fitting") return ["st-run", `fitting ${Math.round(f.progress * 100)} %`];
  if (f.state === "queued") return ["st-run", "queued"];
  if (f.state === "error") return ["st-err", "failed"];
  if (!f.result) return ["", "not fitted"];
  return f.key === fitKey(S.cfg, f.info) ? ["", "fitted"] : ["st-old", "settings changed"];
}

// a file's chosen map and its valid points, from its own fit
function fileValues(f, q) {
  const r = f.result, c = S.cfg, m = r.m;
  f.dec ||= { I: f32(r.intensity), A: f32(r.area), se: f32(r.std_error) };
  const { I, A, se } = f.dec, P = se.length;
  const jn = r.names.indexOf(c.ratio_num), jd = r.names.indexOf(c.ratio_den);
  if (jn < 0 || jd < 0) return null;
  const col = (X, j) => { const o = new Float32Array(P); for (let p = 0; p < P; p++) o[p] = X[p * m + j]; return o; };
  const In = col(I, jn), Id = col(I, jd), rI = safeRatio(In, Id), rA = safeRatio(col(A, jn), col(A, jd));
  const valid = new Uint8Array(P);
  for (let p = 0; p < P; p++) valid[p] = admits(c.intensity, Math.max(In[p], Id[p])) && admits(c.ratio_i, rI[p])
    && admits(c.ratio_a, rA[p]) && Number.isFinite(se[p]) ? 1 : 0;
  return { values: { "ratio-i": rI, "ratio-a": rA, "num-i": In, "den-i": Id, err: se }[q], valid, rI, rA, In, Id, se };
}
function fileSummary(f) {
  if (!f.result) return null;
  const vk = stableJSON(VIEW_KEYS.map(k => S.cfg[k]));
  if (f.sum && f.sum.vk === vk) return f.sum;
  const v = fileValues(f, "ratio-i");
  if (!v) return (f.sum = { vk, missing: true });
  const pick = a => summary(a.filter((_, p) => v.valid[p]));
  let measured = 0, nValid = 0;
  for (let p = 0; p < v.se.length; p++) { if (Number.isFinite(v.se[p])) measured++; nValid += v.valid[p]; }
  return (f.sum = { vk, measured, valid: nValid, rI: pick(v.rI), rA: pick(v.rA), In: pick(v.In), Id: pick(v.Id), se: pick(v.se) });
}

function renderFiles(opts = {}) {
  filePill();
  if (!$("#tab-files").classList.contains("on")) { F.dirty = true; return; }
  F.dirty = false;
  $("#files-count").textContent = F.files.length ? `${F.files.length} open` : "";
  const t = $("#files-table"); t.textContent = "";
  const lab = S.cfg ? `${S.cfg.ratio_num}/${S.cfg.ratio_den}` : "";
  const nCmp = F.files.filter(f => f.compare).length;
  if (F.files.length) t.append(el("tr", {},
    el("th", { title: "Show in Compare (up to four)" }, ""), el("th", {}, "File"), el("th", {}, "Status"),
    el("th", { title: "valid points / points fitted" }, "Valid"), el("th", {}, `${lab} I median`), el("th", {}, "mean ± SD"),
    el("th", {}, `${lab} A median`), el("th", {}, "mean ± SD"), el("th", {}, "")));
  for (const f of F.files) {
    const [cls, st] = statusOf(f), s = fileSummary(f);
    const cb = el("input", { type: "checkbox", title: "Show in Compare" });
    cb.checked = f.compare; cb.disabled = !f.compare && nCmp >= 4;
    cb.onchange = () => { f.compare = cb.checked; renderFiles(); };
    const cells = !s ? ["", "", "", "", ""] : s.missing ? ["", `no ${lab}`, "", "", ""]
      : [`${fmt(s.valid)} / ${fmt(s.measured)}`, fmt(s.rI.median), `${fmt(s.rI.mean)} ± ${fmt(s.rI.sd)}`, fmt(s.rA.median), `${fmt(s.rA.mean)} ± ${fmt(s.rA.sd)}`];
    t.append(el("tr", { class: S.ds && S.ds.id === f.id ? "open" : "" },
      el("td", {}, cb),
      el("td", { class: "name", title: `${f.name}: open on the Carbon fit tab` }, el("button", { class: "link", onclick: () => openFile(f) }, f.name)),
      el("td", { class: cls, title: f.error || "" }, st),
      ...cells.map(c => el("td", {}, c)),
      el("td", {}, el("button", { class: "icon", title: "Remove from the list", onclick: () => removeFile(f) }, "✕"))));
  }
  $("#files-run").disabled = F.running || !F.files.length;
  $("#files-run").textContent = F.running ? "Fitting…" : "Fit all";
  $("#files-csv").disabled = !F.files.some(f => f.result);
  if (opts.compare !== false) renderCompare();
}

async function openFile(f) {
  switchTab("qc");
  if (S.ds && S.ds.id === f.id && !S.ds.resultsOnly) return;
  try { await loadDataset(f.info); } catch (e) { setStatus("Could not open", "err"); toast(e.message, "error"); }
}
function removeFile(f) {
  if (F.busy === f.id) { toast("This file is being fitted: remove it when it has finished", "error"); return; }
  F.files.splice(F.files.indexOf(f), 1);
  // the web build keeps every file in memory: let it go (unless it is the one open)
  if (isBrowserOnly() && !(S.ds && S.ds.id === f.id)) api(`/api/ds/${f.id}/close`, {}).catch(() => {});
  renderFiles();
}

// fit every file whose fit is missing or used other settings, one at a time
async function fitAll() {
  if (F.running || !S.cfg) return;
  const errs = validate();
  if (errs.length) { toast(errs[0], "error"); return; }
  const cfg = structuredClone(S.cfg), key = f => fitKey(cfg, f.info);
  const todo = F.files.filter(f => !(f.result && f.key === key(f)));
  if (!todo.length) { toast("Every file is already fitted with these settings", "good"); return; }
  F.running = true;
  todo.forEach(f => { f.state = "queued"; });
  renderFiles({ compare: false });
  try {
    for (const [i, f] of todo.entries()) {
      if (!fileOf(f.id)) continue;                       // removed meanwhile
      while (S.fitRunning) await sleep(200);             // the open map's own fit goes first
      if (f.result && f.key === key(f)) continue;        // fitted meanwhile, as the open map
      setStatus(`Fitting ${f.name} (${i + 1} of ${todo.length})…`, "busy");
      await fitOne(f, cfg);
      // the open map was waiting on this fit: refit only if its settings changed meanwhile
      if (S.pendingFit && S.ds && !S.fitRunning) {
        const g = fileOf(S.ds.id);
        if (g && g.result && JSON.stringify(g.result.config) === JSON.stringify(S.cfg)) S.pendingFit = false; else startFit();
      }
    }
  } finally {
    F.running = false;
    const bad = todo.filter(f => f.state === "error").length;
    setStatus(`Fitted ${todo.length - bad} of ${todo.length} files` + (bad ? `: ${bad} failed (hover its status)` : ""), bad ? "err" : "");
    renderFiles();
  }
}
async function fitOne(f, cfg) {
  F.busy = f.id; f.state = "fitting"; f.progress = 0;
  renderFiles({ compare: false });
  try {
    await api(`/api/ds/${f.id}/fit`, cfg);
    let st;
    do {
      await sleep(250);
      st = await api(`/api/ds/${f.id}/fit`);
      f.progress = st.progress || 0; renderFiles({ compare: false });
    } while (st.state === "running");
    if (st.state === "error") throw new Error(st.error);
    const r = await api(`/api/ds/${f.id}/fit?result=1`);
    F.busy = null;
    rememberFit(f.id, r.result);
    if (S.ds && S.ds.id === f.id && !S.ds.resultsOnly) decodeFit(r.result);     // the open map: show it
  } catch (e) {
    f.state = "error"; f.error = e.message;
  } finally { F.busy = null; }
}

// several maps dropped together: one group per map, each with its white-light
// image (matched by name) or its ParticleFinder project; result tables left out
async function mapGroups(files) {
  const { isResultsTable } = await import("./engine/readers.js");
  const stem = f => f.name.toLowerCase().replace(/\.[^.]+$/, "");
  const maps_ = [], pfs = [], vids = [], tables = [];
  for (const f of files) {
    if (/\.l6v$/i.test(f.name)) vids.push(f);
    else if (/\.l6pf$/i.test(f.name)) pfs.push(f);
    else if (/\.l6m$/i.test(f.name)) maps_.push(f);
    else if (/\.(txt|csv|tsv|dat)$/i.test(f.name)) (isResultsTable(await f.slice(0, 65536).text()) ? tables : maps_).push(f);
  }
  const groups = maps_.map(f => [f]);
  for (const pf of pfs) { const g = groups.find(g => stem(g[0]).startsWith(stem(pf))); if (g) g.push(pf); else groups.push([pf]); }
  const common = (a, b) => { let i = 0; while (i < a.length && a[i] === b[i]) i++; return i; };
  for (const v of vids) {
    let best = null, n = 3;
    for (const g of groups) { const c = common(stem(v), stem(g[0])); if (c > n) { n = c; best = g; } }
    if (best) best.push(v);
  }
  return { groups, tables };
}
async function openMany(groups, tables) {
  if (tables.length) toast(`Fitted-result tables open one at a time: left out ${tables.map(f => f.name).join(", ")}`, "info", 8000);
  const infos = [];
  for (const [i, g] of groups.entries()) {
    setStatus(`Reading ${i + 1} of ${groups.length}: ${g[0].name}…`, "busy");
    try { const info = await apiUpload(g); addFile(info); infos.push(info); }
    catch (e) { toast(`${g[0].name}: ${e.message}`, "error", 9000); }
  }
  if (!infos.length) throw new Error("none of these files could be opened");
  switchTab("files");
  await loadDataset(infos[0]);
  fitAll();
}

// the same map of up to four files, side by side
const cmpMaps = [0, 1, 2, 3].map(i => new MapView($(`#cmp-${i}`), {
  onTitle: mv => { const f = fileOf(mv.fileId); if (f) openFile(f); },
  onSmoothChange: mv => { rememberSmoothing(mv); renderCompare(); },    // compared maps look alike
}));
cmpMaps.forEach(mv => { mv.titleEl.title = "Open this file on the Carbon fit tab"; });
function fillCmpQ() {
  const s = $("#cmp-q"), cur = s.value || "ratio-i", n = S.cfg.ratio_num, d = S.cfg.ratio_den;
  s.replaceChildren(...[["ratio-i", `Ratio ${n}/${d} · Intensity`], ["ratio-a", `Ratio ${n}/${d} · Area`],
    ["num-i", `${n} · Intensity`], ["den-i", `${d} · Intensity`], ["err", "Error · Std. error"]]
    .map(([v, t]) => el("option", { value: v, selected: v === cur }, t)));
}
function renderCompare() {
  if (!S.cfg) return;
  fillCmpQ();
  const sel = F.files.filter(f => f.compare && f.result), q = $("#cmp-q").value;
  const grid = $("#cmp-grid"); grid.dataset.n = sel.length;
  $("#cmp-empty").classList.toggle("hidden", sel.length > 0);
  $("#cmp-empty").innerHTML = F.files.length
    ? (F.files.some(f => f.compare) ? "Fitting… the maps appear here when their fits finish." : "Tick files in the summary to compare them here (up to four).")
    : "Open two or more maps to compare them here: drop them on the window together, or use <b>Add files…</b>.";
  $("#cmp-info").textContent = sel.length ? `${$("#cmp-q").selectedOptions[0].textContent} · ${$("#cmp-shared").checked ? "one colour scale" : "each map its own scale"}` : "";
  const vals = sel.map(f => fileValues(f, q));
  let range = null;
  if ($("#cmp-shared").checked && vals.some(Boolean)) {
    const pool = [];
    for (const v of vals) if (v) for (let p = 0; p < v.values.length; p += Math.max(1, Math.floor(v.values.length / 20000))) if (v.valid[p] && Number.isFinite(v.values[p])) pool.push(v.values[p]);
    pool.sort((a, b) => a - b);
    if (pool.length) range = [quantile(pool, 0.01), quantile(pool, 0.99)];
  }
  const unit = q === "ratio-i" || q === "ratio-a" ? "" : "counts", label = $("#cmp-q").selectedOptions[0].textContent;
  cmpMaps.forEach((mv, i) => {
    const f = sel[i], v = vals[i];
    mv.host.classList.toggle("hidden", !f);
    if (!f) { mv.fileId = null; return; }
    if (mv.fileId !== f.id) {
      const inf = f.info;
      mv.setGeometry({ nx: inf.nx, ny: inf.ny, x: f32(inf.x), y: f32(inf.y),
        points: inf.points ? Object.fromEntries(Object.entries(inf.points).map(([k, a]) => [k, f32(a)])) : null });
      mv.fileId = f.id;
    }
    mv.setTitle(escHTML(f.name));
    const band = q === "num-i" ? S.cfg.ratio_num : q === "den-i" ? S.cfg.ratio_den : null;
    applySmoothing(mv, `cmp:${band || q}`, band);
    mv.emptyText = v ? null : `This fit has no ${S.cfg.ratio_num} or ${S.cfg.ratio_den} band`;
    if (v) mv.setData(v.values, { valid: v.valid, label, unit, fixedRange: range });
    else mv.render();
  });
}
$("#cmp-q").onchange = () => renderCompare();
$("#cmp-shared").onchange = () => renderCompare();
$("#files-add").onclick = () => $("#file-input").click();
$("#files-run").onclick = () => fitAll();
$("#files-csv").onclick = () => {
  const n = S.cfg.ratio_num, d = S.cfg.ratio_den, num = v => (Number.isFinite(v) ? String(+v.toPrecision(6)) : "");
  const csv = v => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
  const rows = [["File", "Status", "Points fitted", "Valid points",
    `Ratio ${n}/${d} intensity median`, "mean", "SD", `Ratio ${n}/${d} area median`, "mean", "SD",
    `${n} intensity median`, `${d} intensity median`, "Std error median", "Fit seconds"]];
  for (const f of F.files) {
    const s = fileSummary(f), st = statusOf(f)[1];
    rows.push(!s || s.missing ? [f.name, s ? `no ${n}/${d}` : st] : [f.name, st, String(s.measured), String(s.valid),
      num(s.rI.median), num(s.rI.mean), num(s.rI.sd), num(s.rA.median), num(s.rA.mean), num(s.rA.sd),
      num(s.In.median), num(s.Id.median), num(s.se.median), num(f.result.seconds)]);
  }
  download("carbo_metrics_summary.csv", rows.map(r => r.map(csv).join(",")).join("\n") + "\n", "text/csv");
};

// the file pill: with several files open, a menu to switch between them
function filePill() {
  const n = F.files.length;
  $("#file-n").textContent = n > 1 ? `${n} ▾` : "";
  $("#btn-library").title = n > 1 ? "Switch between the open files" : "Open a map from the library (or drop a .l6m or text file anywhere)";
}
function fileMenu(e) {
  e.stopPropagation();
  $$(".popover").forEach(p => p.remove());
  const pop = el("div", { class: "popover file-menu", onclick: ev => ev.stopPropagation() });
  const close = () => pop.remove();
  pop.append(el("div", { class: "pop-h" }, "Open files"),
    el("div", { class: "menu-list" }, ...F.files.map(f => el("button", {
      class: S.ds && S.ds.id === f.id ? "on" : "", title: f.name, onclick: () => { close(); openFile(f); },
    }, el("span", {}, f.name), el("span", { class: "s" }, statusOf(f)[1])))),
    el("div", { class: "menu-list" },
      el("button", { onclick: () => { close(); $("#file-input").click(); } }, "Add files…"),
      ...(isBrowserOnly() ? [] : [el("button", { onclick: () => { close(); showLibrary(); } }, "Open from the library…")]),
      el("button", { onclick: () => { close(); switchTab("files"); } }, "Fit all and compare (Files tab)")));
  document.body.append(pop);
  const r = e.currentTarget.getBoundingClientRect();
  pop.style.width = "320px";
  pop.style.left = Math.max(8, Math.min(window.innerWidth - 328, r.left)) + "px";
  pop.style.top = r.bottom + 6 + "px";
  setTimeout(() => document.addEventListener("click", () => pop.remove(), { once: true }));
}

// =========================================================================
// splat 3-D
// =========================================================================
let sview = null;
const sliceData = new MapView($("#slice-data"), {}), sliceFieldView = new MapView($("#slice-field"), {});
sliceData.cmap = sliceFieldView.cmap = "Magma";

function splatState() { return S.ds ? (S.splat[S.ds.id] ||= {}) : {}; }
function splatReset() {
  $("#gl-empty").classList.remove("hidden");
  $("#sp-metrics").replaceChildren(el("div", { class: "muted small" }, "Appears when the fit finishes."));
  $("#sp-prog div").style.width = "0";
  const st = splatState();
  if (st.field) { $("#gl-empty").classList.add("hidden"); }
}

function splatShow() {
  if (!sview) {
    sview = new SplatView($("#gl"));
    $("#sp-exposure").oninput = e => { sview.exposure = Math.exp(+e.target.value); sview.draw(); };
    $("#sp-depth").oninput = e => sview.setDepth(+e.target.value);
    $("#sp-colour").onchange = e => { sview.colorBy = e.target.value; if (sview.scene) sview.setField(sview.scene.field); sview.draw(); };
    $("#sp-floor").onchange = e => { sview.showFloor = e.target.checked; sview.draw(); };
    $("#sp-eq").onchange = e => { sview.equalise = e.target.checked; if (sview.scene) sview.setField(sview.scene.field); sview.draw(); };
    $("#sp-rotate").onchange = e => { sview.autoRotate = e.target.checked; sview.draw(); };
    $("#sp-reset").onclick = () => sview.resetView();
  }
  if (S.points) {
    $("#gl-empty").classList.remove("hidden");
    $("#sp-status").textContent = "Splat 3D needs a gridded map: ParticleFinder spectra are one point per particle.";
    $("#sp-run").disabled = $("#sp-run2").disabled = true;
    return;
  }
  $("#sp-run").disabled = $("#sp-run2").disabled = !!splatState().running || !S.health.splat;
  const st = splatState();
  buildChips();
  if (st.field) { showField(st.field); if (st.result) showMetrics(st.result); drawLoss(st.history || []); }
  else { $("#gl-empty").classList.remove("hidden"); if (sview.scene) { sview.n = 0; sview.draw(); } }
  if (st.running) pollSplat();
  else if (!st.field && S.ds) {
    // a job started before a page reload, or still running on the server
    api(`/api/ds/${S.ds.id}/splat`).then(r => {
      if (r.state === "running" || r.state === "done") { st.running = r.state === "running"; st.snapIter = -1; pollSplat(); }
    }).catch(() => {});
  }
}

function sceneFor(field) {
  const mv = maps[0];
  return {
    field, peaks: S.fit ? S.fit.names.map((n, j) => ({ name: n, pos: S.cfg.peaks[j] ? S.cfg.peaks[j].pos : 0, width: S.cfg.peaks[j] ? S.cfg.peaks[j].width : 40 })) : [],
    wnRange: field.wn_range, mapRect: [mv.x0, mv.x1, mv.y0, mv.y1],
    video: S.video && S.ds.video ? { img: S.video, x: S.ds.video.x, y: S.ds.video.y } : null,
  };
}

function showField(field) {
  $("#gl-empty").classList.add("hidden");
  if (!sview.scene || sview.scene.mapRect[0] !== maps[0].x0 || sview._dsid !== S.ds.id) { sview.setScene(sceneFor(field)); sview._dsid = S.ds.id; sview.resetView(); }
  else { sview.setField(field); sview.draw(); }
  sview.setSlice(+$("#sp-nu").value);
  renderSlices();
}

function buildChips() {
  const host = $("#sp-chips"); host.textContent = "";
  (S.cfg.peaks || []).forEach((p, j) => host.append(el("button", {
    style: `border-color:${PEAK_COLOURS[j % PEAK_COLOURS.length]}66`,
    onclick: () => { $("#sp-nu").value = p.pos; onNu(); },
  }, `${p.name} ${p.pos}`)));
}

function onNu() {
  const nu = +$("#sp-nu").value;
  $("#sp-nu-label").textContent = nu;
  if (sview) sview.setSlice(nu);
  renderSlicesSoon();
}
$("#sp-nu").addEventListener("input", onNu);
const renderSlicesSoon = debounce(() => renderSlices(), 40);

function renderSlices() {
  const st = splatState(); if (!st.field || !S.fit) return;
  const nu = +$("#sp-nu").value;
  // measured, baseline-corrected, +/- 4 cm-1
  const ks = []; for (let c = 0; c < S.k; c++) if (Math.abs(S.wn[c] - nu) <= 4) ks.push(c);
  if (!ks.length) { let b = 0; for (let c = 1; c < S.k; c++) if (Math.abs(S.wn[c] - nu) < Math.abs(S.wn[b] - nu)) b = c; ks.push(b); }
  const f = S.fit, [lo, hi] = f.range, mid = (lo + hi) / 2, half = (hi - lo) / 2;
  const data = new Float32Array(P());
  for (let p = 0; p < P(); p++) {
    let s = 0;
    for (const c of ks) {
      const t = (S.wn[c] - mid) / half; let b = 0, tk = 1;
      for (let i = 0; i < f.nb; i++) { b += f.baseline[p * f.nb + i] * tk; tk *= t; }
      s += S.cube[p * S.k + c] - b;
    }
    data[p] = s / ks.length;
  }
  const up = 3, xs = new Float32Array(S.nx * up), ys = new Float32Array(S.ny * up);
  const xa = Math.min(S.x[0], S.x[S.nx - 1]), xb = Math.max(S.x[0], S.x[S.nx - 1]);
  const ya = Math.min(S.y[0], S.y[S.ny - 1]), yb = Math.max(S.y[0], S.y[S.ny - 1]);
  for (let j = 0; j < xs.length; j++) xs[j] = xa + (xb - xa) * j / (xs.length - 1);
  for (let i = 0; i < ys.length; i++) ys[i] = ya + (yb - ya) * i / (ys.length - 1);
  const field = sliceField(st.field, nu, xs, ys);
  const sorted = Float64Array.from(data).sort();
  const range = [Math.min(0, quantile(sorted, 0.01)), quantile(sorted, 0.995)];
  sliceData.setGeometry({ nx: S.nx, ny: S.ny, x: S.x, y: S.y });
  sliceData.setTitle(`Measured · ${nu} cm⁻¹`); sliceData.setData(data, { label: "counts", fixedRange: range });
  sliceFieldView.setGeometry({ nx: xs.length, ny: ys.length, x: xs, y: ys });
  sliceFieldView.setTitle(`Splat field · ${nu} cm⁻¹`); sliceFieldView.setData(field, { label: "counts", fixedRange: range });
}

async function runSplat() {
  if (!S.fit) return toast("Open and fit a map first", "error");
  if (S.ds.resultsOnly) return toast("Splat 3D needs spectra: open the map itself, not its fitted values", "error");
  const st = splatState();
  try {
    await api(`/api/ds/${S.ds.id}/splat`, { n_gaussians: +$("#sp-n").value, iters: +$("#sp-iters").value, g_name: S.cfg.ratio_den });
    st.running = true; st.snapIter = -1; st.history = [];
    $("#sp-run").disabled = $("#sp-run2").disabled = true;
    pollSplat();
  } catch (e) { toast(e.message, "error", 8000); }
}
$("#sp-run").onclick = runSplat; $("#sp-run2").onclick = runSplat;

async function pollSplat() {
  const id = S.ds.id, st = splatState();
  if (st.polling) return; st.polling = true;
  try {
    for (;;) {
      const r = await api(`/api/ds/${id}/splat?since=${st.snapIter ?? -1}`);
      if (!S.ds || S.ds.id !== id) return;
      st.history = r.history || st.history;
      $("#sp-prog div").style.width = ((r.progress || 0) * 100).toFixed(1) + "%";
      if (r.last) $("#sp-status").textContent = `Iteration ${r.last.iter} / ${r.iters} · loss ${fmt(r.last.loss, 4)} · ${r.last.n} Gaussians · ${Math.round(Date.now() / 1000 - r.started)} s`;
      drawLoss(st.history);
      if (r.snapshot && $("#tab-splat").classList.contains("on")) { st.snapIter = r.snap_iter; st.field = r.snapshot; showField(r.snapshot); }
      if (r.state === "done") {
        st.field = r.result; st.result = r.result; st.running = false;
        if ($("#tab-splat").classList.contains("on")) { showField(r.result); }
        showMetrics(r.result);
        $("#sp-status").textContent = `Done in ${r.result.seconds.toFixed(0)} s · ${r.result.n} Gaussians on ${fmt(r.result.voxels)} voxels (particles = ${(r.result.mask_fraction * 100).toFixed(0)} % of the map)`;
        toast("Splat field fitted", "good");
        break;
      }
      if (r.state === "error") { st.running = false; toast(r.error, "error", 9000); $("#sp-status").textContent = "Failed: " + r.error; break; }
      if (r.state === "idle") { st.running = false; break; }
      await sleep(800);
    }
  } finally { st.polling = false; $("#sp-run").disabled = $("#sp-run2").disabled = false; }
}

function drawLoss(h) {
  const c = $("#sp-loss"), W = c.clientWidth, H = c.clientHeight; if (!W) return;
  const ctx = hidpi(c, W, H);
  if (!h.length) return;
  const ls = h.map(p => Math.log10(p.loss)), lo = Math.min(...ls), hi = Math.max(...ls);
  const n = h[h.length - 1].iter, X = i => 4 + (i - 1) / Math.max(1, n - 1) * (W - 8), Y = v => H - 14 - (v - lo) / ((hi - lo) || 1) * (H - 22);
  const g = ctx.createLinearGradient(0, 0, W, 0); g.addColorStop(0, T.accent2); g.addColorStop(1, T.accent);
  ctx.strokeStyle = g; ctx.lineWidth = 1.6; ctx.beginPath();
  h.forEach((p, k) => { const x = X(p.iter), y = Y(Math.log10(p.loss)); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  ctx.fillStyle = T.label; ctx.font = font(11); ctx.fillText("loss (log) vs iteration", 6, H - 2);
}

function showMetrics(r) {
  const R = r.report || {}, m = (k, v, s, cls = "") => el("div", { class: "metric" }, el("div", { class: "k" }, k), el("div", { class: "v " + cls }, v), s ? el("div", { class: "s" }, s) : null);
  const ron = R.resid_over_noise;
  $("#sp-metrics").replaceChildren(
    m("Held-out error ÷ noise", fmt(ron, 3), "1.0 = at the map's own noise floor", ron < 1.15 ? "good" : ""),
    m("Gaussians", fmt(r.n), `${fmt(R.n_params)} parameters`),
    m("Compression", `${fmt(R.compression_ratio_full, 3)}×`, "full cube values per parameter"),
    m("PSNR held out", `${fmt(R.psnr_eval_dB, 3)} dB`, `SSIM ${fmt(R.ssim_train, 3)} (train)`),
  );
}

// =========================================================================
// particles
// =========================================================================
const pmap = new MapView($("#map-p"), {
  onTitle: (mv, e) => specMenu(e, S.pspec, () => renderParticles()),
  noProfile: true,
});
pmap.cmap = "Viridis";
const stripHost = $("#pt-strip"), stripCanvas = el("canvas"); stripHost.append(stripCanvas);
new ResizeObserver(() => (S.points ? S.pf && drawScatter() : S.seg && drawStrip())).observe(stripHost);

function segSource() {
  const v = $("#pt-src").value || `log:${S.cfg.ratio_den}`;
  if (v === "counts") return meanCounts();
  const [mode, name] = v.split(":"), I = peakCol(name, "intensity");
  if (!I) return null;
  return mode === "log" ? I.map(x => Math.log1p(Math.max(0, x))) : I;
}

function particlesInit() {
  if (!S.fit) return;
  if (S.points) { pfInit(); return; }
  const src = $("#pt-src"), cur = src.value;
  src.textContent = "";
  for (const n of S.fit.names) src.append(el("option", { value: `log:${n}` }, `log(1 + ${n} intensity)`));
  src.append(el("option", { value: "counts" }, "Mean counts"));
  src.value = cur && [...src.options].some(o => o.value === cur) ? cur : `log:${S.cfg.ratio_den}`;
  if (!S.seg) autoThreshold();
  else renderParticles();
}
function autoThreshold() {
  const v = segSource(); if (!v) return;
  let lo = Infinity, hi = -Infinity; for (const x of v) { if (x < lo) lo = x; if (x > hi) hi = x; }
  const t = otsu(v);
  const sl = $("#pt-thr"); sl.min = lo; sl.max = hi; sl.step = (hi - lo) / 500; sl.value = t;
  $("#pt-thr-n").value = +t.toPrecision(4);
  doSegment();
}
function doSegment() {
  const v = segSource(); if (!v) return;
  S.seg = segment(v, S.nx, S.ny, +$("#pt-thr").value, +$("#pt-min").value || 1, $("#pt-split").checked);
  renderParticles();
}
$("#pt-split").onchange = doSegment;
$("#pt-src").onchange = autoThreshold;
$("#pt-otsu").onclick = autoThreshold;
$("#pt-thr").oninput = e => { $("#pt-thr-n").value = +(+e.target.value).toPrecision(4); doSegmentSoon(); };
$("#pt-thr-n").onchange = e => { $("#pt-thr").value = e.target.value; doSegment(); };
$("#pt-min").onchange = doSegment;
const doSegmentSoon = debounce(doSegment, 60);

function particleStats() {
  const r = resolve(S.pspec), seg = S.seg; if (!r.values || !seg) return null;
  const area = Math.abs((S.x[1] - S.x[0]) * (S.y[1] - S.y[0]));
  const gW = peakCol(S.cfg.ratio_den, "width");
  const parts = seg.parts.map(pt => {
    const vals = pt.pixels.filter(p => !r.useValid || !S.valid || S.valid[p]).map(p => r.values[p]);
    let sx = 0, sy = 0, sw = 0; for (const p of pt.pixels) { const [x, y] = pmap.idxXY(p); sx += x; sy += y; sw += gW ? gW[p] : 0; }
    return { id: pt.id, n: pt.pixels.length, area: pt.pixels.length * area, st: summary(vals), vals, cx: sx / pt.pixels.length, cy: sy / pt.pixels.length, gw: sw / pt.pixels.length };
  });
  const inside = parts.flatMap(p => p.vals);
  return { r, parts, pixel: summary(inside), particle: summary(parts.filter(p => p.st.n).map(p => p.st.mean)), cover: parts.reduce((s, p) => s + p.n, 0) / P() };
}

function renderParticles() {
  if (!S.fit) return;
  if (S.points) { renderPF(); return; }
  $("#pt-cards-h").textContent = "Pixels vs particles"; $("#pt-strip-h").textContent = "Ratio by particle";
  pmap.categories = null;
  const r = resolve(S.pspec);
  pmap.setTitle(r.title);
  pmap.overlays = [outlineOverlay(S.seg)];
  if (r.values) pmap.setData(r.values, { valid: r.useValid ? S.valid : null, label: r.label, unit: r.unit });
  const ps = particleStats(); if (!ps) return;
  S.pstats = ps;
  const card = (k, v, s) => el("div", { class: "card" }, el("div", { class: "k" }, k), el("div", { class: "v" }, v), el("div", { class: "s" }, s));
  $("#pt-cards").replaceChildren(
    card("Particles", fmt(ps.parts.length), `${(ps.cover * 100).toFixed(1)} % of the map`),
    card("Pixels inside", fmt(ps.pixel.n), `of ${fmt(P())}`),
    card(`Pixel level · ${r.name}`, `${fmt(ps.pixel.mean)} ± ${fmt(ps.pixel.sd, 3)}`, `n = ${fmt(ps.pixel.n)} pixels`),
    card(`Particle level · ${r.name}`, `${fmt(ps.particle.mean)} ± ${fmt(ps.particle.sd, 3)}`, `n = ${ps.particle.n} particles`));
  const t = $("#pt-table"); t.textContent = "";
  t.append(el("tr", {}, ...["#", "Pixels", "Area µm²", "Median", "IQR", `${S.cfg.ratio_den} width`, "Centre (µm)"].map(h => el("th", {}, h))));
  for (const p of ps.parts) t.append(el("tr", {},
    el("td", {}, el("b", { style: `color:${PARTICLE_COLOURS[(p.id - 1) % PARTICLE_COLOURS.length]}` }, `● ${p.id}`)),
    el("td", {}, fmt(p.n)), el("td", {}, fmt(p.area, 3)), el("td", {}, fmt(p.st.median)), el("td", { class: "sub" }, `${fmt(p.st.q1, 3)}–${fmt(p.st.q3, 3)}`),
    el("td", {}, fmt(p.gw, 3)), el("td", { class: "sub" }, `${fmt(p.cx, 3)}, ${fmt(p.cy, 3)}`)));
  drawStrip();
}

function drawStrip() {
  const ps = S.pstats; if (!ps) return;
  const W = stripHost.clientWidth, H = stripHost.clientHeight; if (W < 20) return;
  const ctx = hidpi(stripCanvas, W, H);
  const parts = ps.parts.slice(0, 24);
  const all = parts.flatMap(p => p.vals).sort((a, b) => a - b); if (!all.length) return;
  const lo = quantile(all, 0.02), hi = quantile(all, 0.98) || lo + 1;
  const L = 44, B = 18, Y = v => 8 + (1 - (v - lo) / (hi - lo)) * (H - 8 - B);
  ctx.font = font(11); ctx.fillStyle = T.ink; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const t of niceTicks(lo, hi, 4)) { ctx.fillStyle = T.grid; ctx.fillRect(L, Y(t), W - L, 1); ctx.fillStyle = T.ink; ctx.fillText(fmt(t, 3), L - 4, Y(t)); }
  const bw = (W - L - 6) / Math.max(1, parts.length);
  parts.forEach((p, k) => {
    const cx = L + bw * (k + 0.5), col = PARTICLE_COLOURS[(p.id - 1) % PARTICLE_COLOURS.length];
    const s = Float64Array.from(p.vals).sort(); if (!s.length) return;
    ctx.globalAlpha = 0.35; ctx.fillStyle = col;
    const step = Math.max(1, Math.floor(s.length / 160));
    for (let i = 0; i < s.length; i += step) { const v = s[i]; if (v < lo || v > hi) continue; ctx.fillRect(cx - bw * 0.3 + Math.random() * bw * 0.6, Y(v), 1.5, 1.5); }
    ctx.globalAlpha = 1;
    const q1 = quantile(s, 0.25), q3 = quantile(s, 0.75), md = quantile(s, 0.5);
    ctx.strokeStyle = col; ctx.lineWidth = 1.4;
    ctx.strokeRect(cx - bw * 0.18, Y(Math.min(hi, q3)), bw * 0.36, Y(Math.max(lo, q1)) - Y(Math.min(hi, q3)));
    ctx.fillStyle = T.strong; ctx.fillRect(cx - bw * 0.2, Y(md), bw * 0.4, 2);
    ctx.fillStyle = col; ctx.textAlign = "center"; ctx.fillText(p.id, cx, H - 8);
  });
}

// =========================================================================
// ParticleFinder: one spectrum per particle, with the instrument's morphology
// =========================================================================
const LOGX = new Set(["ecd", "area", "min_diam", "volume"]);

function ranks(a) {
  const idx = a.map((v, i) => i).sort((p, q) => a[p] - a[q]), r = new Float64Array(a.length);
  for (let i = 0; i < idx.length;) {
    let j = i; while (j + 1 < idx.length && a[idx[j + 1]] === a[idx[i]]) j++;
    for (let k = i; k <= j; k++) r[idx[k]] = (i + j) / 2;
    i = j + 1;
  }
  return r;
}
function spearman(x, y) {
  if (x.length < 3) return NaN;
  const rx = ranks(x), ry = ranks(y), n = x.length, mx = (n - 1) / 2;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { sxy += (rx[i] - mx) * (ry[i] - mx); sxx += (rx[i] - mx) ** 2; syy += (ry[i] - mx) ** 2; }
  return sxy / Math.sqrt(sxx * syy);
}

function pfInit() {
  if (S.pfLaserFor !== S.ds.id) { S.pfLaserFor = S.ds.id; S.pfa = null; }
  const sel = $("#pf-x");
  if (!sel.options.length) {
    for (const [k, [lab, unit]] of Object.entries(S.pointLabels || {})) sel.append(el("option", { value: k }, unit ? `${lab} (${unit})` : lab));
    sel.value = "ecd";
    sel.onchange = () => renderPF();
  }
  renderPF();
}

function renderPF() {
  const r = resolve(S.pspec);
  pmap.setTitle(r.title);
  pmap.overlays = [];
  if (r.values) pmap.setData(r.values, { valid: r.useValid ? S.valid : null, label: r.label, unit: r.unit });
  pmap.setCursor(null, S.pinned);
  renderPfAnalysis();
  const xk = $("#pf-x").value || "ecd", X = S.points[xk], [xlab, xunit] = S.pointLabels[xk];
  const idx = [];
  for (let p = 0; p < S.nx; p++) if ((!r.useValid || !S.valid || S.valid[p]) && Number.isFinite(r.values[p]) && Number.isFinite(X[p])) idx.push(p);
  const st = summary(idx.map(p => r.values[p])), rho = spearman(idx.map(p => X[p]), idx.map(p => r.values[p]));
  const dia = summary(Array.from(S.points.ecd));
  S.pf = { r, idx, X, xk, xlab, xunit, st, rho };
  S.pstats = null;
  $("#pt-cards-h").textContent = "Particles"; $("#pt-strip-h").textContent = `${r.name} against ${xlab.toLowerCase()}`;
  const card = (k, v, s2) => el("div", { class: "card" }, el("div", { class: "k" }, k), el("div", { class: "v" }, v), el("div", { class: "s" }, s2));
  $("#pt-cards").replaceChildren(
    card("Particles", fmt(S.nx), `${fmt(idx.length)} valid`),
    card("Median diameter", `${fmt(dia.median, 3)} µm`, `IQR ${fmt(dia.q1, 3)}–${fmt(dia.q3, 3)} µm`),
    card(r.name, `${fmt(st.mean)} ± ${fmt(st.sd, 3)}`, `median ${fmt(st.median)} · n = ${fmt(st.n)} particles`),
    card(`Rank correlation with ${xlab.toLowerCase()}`, Number.isFinite(rho) ? rho.toFixed(3) : "–", "Spearman ρ over valid particles"));
  const t = $("#pt-table"); t.textContent = "";
  t.append(el("tr", {}, ...["#", "X, Y (µm)", "⌀ µm", "Convexity", "Elong. %", "Pop.", r.name].map(h => el("th", {}, h))));
  const popOf = p => (S.pfa && S.pfa.k && S.pfa.label[p] >= 0 ? S.pfa.pops[S.pfa.label[p]].name : "–");
  const top = [...idx].sort((a, b) => r.values[b] - r.values[a]).slice(0, 150);
  for (const p of top) t.append(el("tr", { class: "click" + (p === S.pinned ? " on" : ""), title: "Show this particle's spectrum and zoom onto it",
    onclick: () => { setPinned(p); pmap.zoomTo(S.points.x[p], S.points.y[p], Math.max(8 * S.points.ecd[p], 15)); renderPF(); } },
    el("td", {}, el("button", { class: "link pnum", title: "Open the D, G, Ratio and Error maps zoomed onto this particle",
      onclick: e => { e.stopPropagation(); switchTab("qc"); setTimeout(() => zoomToPoint(p), 40); } }, `#${p + 1}`)),
    el("td", { class: "sub" }, `${fmt(S.points.x[p], 4)}, ${fmt(S.points.y[p], 4)}`),
    el("td", {}, fmt(S.points.ecd[p], 3)), el("td", {}, fmt(S.points.convexity[p], 3)), el("td", {}, fmt(S.points.elongation[p], 3)),
    el("td", {}, popOf(p)), el("td", {}, fmt(r.values[p]))));
  if (idx.length > top.length) t.append(el("tr", {}, el("td", { colspan: 7, class: "sub" }, `${fmt(top.length)} highest of ${fmt(idx.length)} shown; Export particle analysis (CSV) has every particle`)));
  drawScatter();
}

function drawScatter() {
  const F = S.pf; if (!F) return;
  const W = stripHost.clientWidth, H = stripHost.clientHeight; if (W < 20) return;
  const ctx = hidpi(stripCanvas, W, H);
  const { idx, X, r } = F; if (!idx.length) return;
  const logx = LOGX.has(F.xk) && idx.every(p => X[p] > 0);
  const xs = idx.map(p => (logx ? Math.log10(X[p]) : X[p])), ys = idx.map(p => r.values[p]);
  const sx = Float64Array.from(xs).sort(), sy = Float64Array.from(ys).sort();
  const xlo = quantile(sx, 0.005), xhi = quantile(sx, 0.995) || xlo + 1, ylo = quantile(sy, 0.01), yhi = quantile(sy, 0.99) || ylo + 1;
  const L = 46, B = 30, Rr = 8, Tt = 8;
  const PX = v => L + (v - xlo) / ((xhi - xlo) || 1) * (W - L - Rr), PY = v => Tt + (1 - (v - ylo) / ((yhi - ylo) || 1)) * (H - Tt - B);
  ctx.font = font(11); ctx.textBaseline = "middle";
  for (const t of niceTicks(ylo, yhi, 4)) { ctx.fillStyle = T.grid; ctx.fillRect(L, PY(t), W - L - Rr, 1); ctx.fillStyle = T.ink; ctx.textAlign = "right"; ctx.fillText(fmt(t, 3), L - 4, PY(t)); }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const xt = logx ? [-2, -1, 0, 1, 2, 3, 4, 5].filter(v => v >= xlo && v <= xhi) : niceTicks(xlo, xhi, 5);
  for (const t of xt) { ctx.fillStyle = T.grid; ctx.fillRect(PX(t), Tt, 1, H - Tt - B); ctx.fillStyle = T.ink; ctx.fillText(logx ? fmt(10 ** t, 3) : fmt(t, 3), PX(t), H - B + 4); }
  ctx.fillStyle = T.label; ctx.fillText(`${F.xlab}${F.xunit ? ` (${F.xunit})` : ""}${logx ? ", log scale" : ""}`, (L + W - Rr) / 2, H - 13);
  const cls = S.points.cls, cols = ["#3fe0ff", "#ff9f40", "#b18cff", "#7dff6a"];
  const A = S.pfa, by = $("#pf-colour").value, byPop = A && A.k && by === "pop", byCoat = A && A.coat && by === "coat", CC = COAT_COLOURS();
  ctx.globalAlpha = Math.max(0.15, Math.min(0.8, 300 / idx.length));
  idx.forEach((p, i) => {
    if (ys[i] < ylo || ys[i] > yhi || xs[i] < xlo || xs[i] > xhi) return;
    ctx.fillStyle = byCoat ? (A.coat.cat[p] < 0 ? T.hatchLine : CC[A.coat.cat[p]])
      : byPop ? (A.label[p] < 0 ? T.hatchLine : A.pops[A.label[p]].nd ? T.label : PARTICLE_COLOURS[A.label[p] % PARTICLE_COLOURS.length]) : cols[Math.max(0, cls[p]) % cols.length];
    ctx.fillRect(PX(xs[i]) - 1.2, PY(ys[i]) - 1.2, 2.4, 2.4);
  });
  ctx.globalAlpha = 1;
  // binned medians: the trend without assuming a shape
  const nb = 10, bins = Array.from({ length: nb }, () => []);
  xs.forEach((v, i) => { const b = Math.floor((v - xlo) / ((xhi - xlo) || 1) * nb); if (b >= 0 && b < nb) bins[b].push(ys[i]); });
  ctx.strokeStyle = T.strong; ctx.lineWidth = 2; ctx.beginPath();
  let started = false;
  bins.forEach((b, k) => {
    if (b.length < 8) return;
    b.sort((u, v) => u - v);
    const x = PX(xlo + (k + 0.5) / nb * (xhi - xlo)), y = PY(quantile(b, 0.5));
    if (started) ctx.lineTo(x, y); else { ctx.moveTo(x, y); started = true; }
  });
  ctx.stroke();
  if (S.pinned != null && idx.includes(S.pinned)) {
    const i = idx.indexOf(S.pinned);
    ctx.strokeStyle = "#7dff6a"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(PX(xs[i]), PY(ys[i]), 5, 0, 7); ctx.stroke();
  }
}

// =========================================================================
// ParticleFinder: semi-quantitative analysis
// =========================================================================
// Detection and quality from each particle's own fit, populations (a Gaussian
// mixture, or cut-offs) with shares by number and by volume and their 95 %
// intervals, the crystallite size La, and size-weighted results.
const POP_NAMES = ["A", "B", "C", "D"];
const pct = v => (Number.isFinite(v) ? `${(100 * v).toFixed(v < 0.1 ? 1 : 0)} %` : "–");

function pfAnalysis() {
  const f = S.fit, c = S.cfg, pts = S.points, n = S.nx, m = f.m;
  const kd = Math.max(0.5, +$("#pf-kd").value || 3), kq = Math.max(kd, +$("#pf-kq").value || 10);
  const mode = $("#pf-pop-mode").value, cuts = [parseFloat($("#pf-cut1").value), parseFloat($("#pf-cut2").value)].filter(Number.isFinite).sort((a, b) => a - b);
  const lambda = parseFloat($("#pf-laser").value);
  // the band the populations are found on, over G (D1 unless chosen otherwise)
  const bandSel = $("#pf-pop-ratio"), others = f.names.filter(b => b.toLowerCase() !== "g");
  if ([...bandSel.options].map(o => o.value).join() !== others.join()) {
    const keep = bandSel.value;
    bandSel.replaceChildren(...others.map(b => el("option", { value: b }, `${b}/G`)));
    bandSel.value = others.includes(keep) ? keep : (others.find(b => b.toLowerCase() === "d1") || others[0] || "");
  }
  const popBand = bandSel.value;
  const coatSel = $("#pf-coat-band");
  if ([...coatSel.options].map(o => o.value).join() !== others.join()) {
    const keep = coatSel.value;
    coatSel.replaceChildren(...others.map(b => el("option", { value: b }, `${b}/G`)));
    coatSel.value = others.includes(keep) ? keep : (others.find(b => b.toLowerCase() === "d3") || others[0] || "");
  }
  const coatBand = coatSel.value;
  const key = [S.ds.id, f.seconds, f.names.join(), stableJSON(VIEW_KEYS.map(k => c[k])), kd, kq, mode, cuts.join(), lambda, popBand, coatBand].join("|");
  if (S.pfa && S.pfa.key === key) return S.pfa;
  const band = name => f.names.findIndex(b => b.toLowerCase() === name);
  const jG = band("g") >= 0 ? band("g") : f.names.indexOf(c.ratio_den), jD = band("d1");
  const col = (q, j) => Float64Array.from({ length: n }, (_, p) => (j < 0 ? NaN : f[q][p * m + j]));
  const IG = col("intensity", jG), WG = col("width", jG), XG = col("position", jG), AG = col("area", jG);
  const ID = col("intensity", jD), AD = col("area", jD), se = f.std_error;
  const fc = f.config || c, gp = fc.peaks.find(q => q.name === f.names[jG]);
  // the uncertainty of every band height, from each particle's own fit (once per fit)
  const ckey = `${S.ds.id}|${f.seconds}|${f.names.join()}`;
  if (!S.pfCov || S.pfCov.key !== ckey) S.pfCov = { key: ckey, cov: heightCovariances(f, fc, S.wn) };
  const cov = S.pfCov.cov, zOf = (p, j) => (j < 0 ? NaN : f.intensity[p * m + j] / Math.sqrt(cov[p * m * m + j * m + j]));
  const zG = new Float64Array(n).fill(NaN);
  const near = (v, b, span) => Math.abs(v - b) < 1e-3 * span;
  const snr = new Float64Array(n).fill(NaN), atBound = new Uint8Array(n), det = new Uint8Array(n), quant = new Uint8Array(n);
  let fitted = 0;
  for (let p = 0; p < n; p++) {
    if (!(se[p] > 0)) continue;
    fitted++;
    snr[p] = IG[p] / se[p]; zG[p] = zOf(p, jG);
    atBound[p] = gp && (near(XG[p], gp.pos_min, gp.pos_max - gp.pos_min) || near(XG[p], gp.pos_max, gp.pos_max - gp.pos_min)
      || near(WG[p], gp.width_min, gp.width_max - gp.width_min) || near(WG[p], gp.width_max, gp.width_max - gp.width_min)) ? 1 : 0;
    det[p] = zG[p] >= kd && !atBound[p] ? 1 : 0;
    quant[p] = snr[p] >= kq && !atBound[p] && (!S.valid || S.valid[p]) ? 1 : 0;
  }
  const Q = []; for (let p = 0; p < n; p++) if (quant[p]) Q.push(p);
  // the grouping ratio, and whether its band is detected in each particle: a band
  // below detection gives a ratio of noise (often exactly 0), not a population
  const jR = f.names.indexOf(popBand), IR = col("intensity", jR);
  const ratio = jR >= 0 ? Float64Array.from({ length: n }, (_, p) => (IG[p] > 0 ? IR[p] / IG[p] : NaN)) : null;
  const rDet = new Uint8Array(n); for (const p of Q) rDet[p] = jR >= 0 && zOf(p, jR) >= kd ? 1 : 0;
  const Qd = Q.filter(p => rDet[p]), nNd = Q.length - Qd.length, rlab = `${popBand}/G`;

  // populations, then renamed A, B, ... by their median ratio, low to high
  let label = new Int16Array(n).fill(-1), k = 0, how = "";
  if (ratio && Qd.length && mode === "auto") {
    if (Qd.length >= 20) {
      const feats = robustScale([Qd.map(p => Math.log10(ratio[p])), Qd.map(p => WG[p]), Qd.map(p => XG[p])]);
      const g = bestGmm(feats, 4, 1);
      Qd.forEach((p, i) => { label[p] = g.labels[i]; });
      k = g.k;
      how = `Gaussian mixture on log ${rlab}, G width and G position (each scaled by its median and MAD): ${g.components} component${g.components > 1 ? "s" : ""} (the lowest ICL from 1 to ${g.tried.length}), ` +
        (g.components > k ? `merged where their density has no dip between them (ridge ratio above 0.2; Hennig 2010) into ${k} population${k > 1 ? "s" : ""}.` : `${k > 1 ? "each" : "a"} separate population.`) +
        (k === 1 ? " One continuous population: use cut-offs to split it into classes." : "");
    } else { Qd.forEach(p => { label[p] = 0; }); k = 1; how = `Fewer than 20 particles with ${popBand} detected: one population.`; }
  } else if (ratio && Qd.length && mode === "cut") {
    Qd.forEach(p => { let l = 0; while (l < cuts.length && ratio[p] >= cuts[l]) l++; label[p] = l; });
    k = cuts.length + 1;
    how = cuts.length ? `Cut-offs on ${rlab} at ${cuts.map(v => fmt(v)).join(" and ")}.` : "Enter a cut-off to split the particles.";
  }
  if (k) {
    const med = Array.from({ length: k }, (_, l) => median(Qd.filter(p => label[p] === l).map(p => ratio[p])));
    const order = med.map((v, l) => [Number.isFinite(v) ? v : Infinity, l]).sort((a, b) => a[0] - b[0]).map(x => x[1]);
    const rename = new Int16Array(k); order.forEach((l, i) => { rename[l] = i; });
    label = label.map(l => (l < 0 ? -1 : rename[l]));
    // a cut-off range holding no particle is not a population
    const used = new Set(Qd.map(p => label[p])); if (used.size < k) { const keep = [...used].sort(); label = label.map(l => (l < 0 ? -1 : keep.indexOf(l))); k = keep.length; }
  }
  // the quantified particles whose band is below detection: a category of their own
  const ND = k; if (k && nNd) for (const p of Q) if (!rDet[p]) label[p] = ND;
  const cats = k + (k && nNd ? 1 : 0);

  // La per quantified particle with a detected D band (Cancado et al. 2006)
  const la = new Float64Array(n).fill(NaN);
  let noD = 0;
  for (const p of Q) {
    if (jD < 0 || !(zOf(p, jD) >= kd)) { noD++; continue; }
    la[p] = laCancado(AD[p], AG[p], lambda);
  }
  const laQ = Q.map(p => la[p]).filter(Number.isFinite);

  // shares by number (Wilson) and by volume (bootstrap over particles)
  const vol = pts.volume, nQ = Q.length;
  const boot = cats ? bootstrapShares(Int16Array.from(Q, p => label[p]), Float64Array.from(Q, p => (vol[p] > 0 ? vol[p] : 0)), cats) : null;
  const pops = Array.from({ length: cats }, (_, l) => {
    const idx = Q.filter(p => label[p] === l);
    return { name: l === ND ? `${popBand} n.d.` : POP_NAMES[l] || String(l + 1), nd: l === ND, n: idx.length, share: idx.length / nQ, ci: wilson(idx.length, nQ), vol: boot ? boot[l] : null,
             ratio: median(idx.map(p => ratio[p])), gw: median(idx.map(p => WG[p])), gx: median(idx.map(p => XG[p])),
             la: median(idx.map(p => la[p])), ecd: median(idx.map(p => pts.ecd[p])) };
  });

  // size: every particle (morphology does not need a spectrum)
  const ecd = Array.from(pts.ecd), vw = Array.from(vol, v => (v > 0 ? v : 0)), es = Float64Array.from(ecd).filter(Number.isFinite).sort();
  const size = { num: [0.1, 0.5, 0.9].map(q => quantileSorted(es, q)), vol: [0.1, 0.5, 0.9].map(q => weightedQuantile(ecd, vw, q)) };
  // the ratio by size class: quartiles of diameter among the quantified particles
  const eq = Float64Array.from(Qd.map(p => pts.ecd[p])).sort(), edges = [0, 0.25, 0.5, 0.75, 1].map(q => quantileSorted(eq, q));
  const classes = ratio && Qd.length >= 8 ? [0, 1, 2, 3].map(i => {
    const idx = Qd.filter(p => pts.ecd[p] >= edges[i] && (i === 3 ? pts.ecd[p] <= edges[4] : pts.ecd[p] < edges[i + 1]));
    const r = Float64Array.from(idx.map(p => ratio[p])).sort();
    return { lo: edges[i], hi: edges[i + 1], n: idx.length, ratio: quantileSorted(r, 0.5), q1: quantileSorted(r, 0.25), q3: quantileSorted(r, 0.75), la: median(idx.map(p => la[p])) };
  }) : [];
  let sw = 0, swr = 0, sr = 0;
  if (ratio) for (const p of Qd) { const w = vol[p] > 0 ? vol[p] : 0; sw += w; swr += w * ratio[p]; sr += ratio[p]; }
  const rho = ratio && Qd.length >= 3 ? spearman(Qd.map(p => pts.ecd[p]), Qd.map(p => ratio[p])) : NaN;

  // mean spectra of groups of particles: baseline removed, each scaled to its G height
  const [lo, hi] = f.range, ks = []; for (let q = 0; q < S.k; q++) if (S.wn[q] >= lo && S.wn[q] <= hi) ks.push(q);
  const K2 = ks.length, NS = new Float32Array(Q.length * K2);
  Q.forEach((p, qi) => { const base = evalFit(p).baseline, o = p * S.k; ks.forEach((q, i) => { NS[qi * K2 + i] = (S.cube[o + q] - base[q]) / IG[p]; }); });
  const groupSpectra = (groupOf, groups) => {
    const sum = Array.from({ length: groups }, () => new Float64Array(K2)), sq = sum.map(() => new Float64Array(K2)), cnt = new Float64Array(groups);
    Q.forEach((p, qi) => {
      const g = groupOf(p); if (g < 0) return;
      for (let i = 0; i < K2; i++) { const y = NS[qi * K2 + i]; sum[g][i] += y; sq[g][i] += y * y; }
      cnt[g]++;
    });
    return { wn: ks.map(q => S.wn[q]), mean: sum.map((s, g) => Array.from(s, v => v / Math.max(1, cnt[g]))),
             sd: sq.map((s, g) => Array.from(s, (v, i) => Math.sqrt(Math.max(0, v / Math.max(1, cnt[g]) - (sum[g][i] / Math.max(1, cnt[g])) ** 2)))), n: Array.from(cnt) };
  };
  const spectra = groupSpectra(p => (cats ? label[p] : 0), cats || 1);

  // Coating uniformity: is the coating band over G the same in every particle
  // once each particle's own measurement error is allowed for? A slow change
  // along the particle order is told apart first (a running median), since
  // ParticleFinder measures the particles in that order and a drift of laser,
  // focus or background would otherwise read as non-uniform coating.
  let coat = null;
  const jC = f.names.indexOf(coatBand);
  if (jC >= 0 && jC !== jG && Q.length >= 10) {
    const IC = col("intensity", jC);
    const R = Q.map(p => IC[p] / IG[p]), sR = Q.map(p => Math.sqrt(Math.max(0, ratioVariance(cov, p * m * m, m, jC, jG, IC[p], IG[p]))));
    const zc = Q.map(p => zOf(p, jC)), nC = zc.filter(z => z >= kd).length;
    const rho = spearman(Q.map(p => p), R), rhoCI = spearmanCI(rho, Q.length);
    const w = Math.max(21, Math.min(201, 2 * Math.floor(Q.length / 20) + 1)), hw = (w - 1) / 2;
    const trend = R.map((_, i) => median(R.slice(Math.max(0, i - hw), Math.min(R.length, i + hw + 1))));
    const mean = R.reduce((s, v) => s + v, 0) / R.length, tenth = Math.max(1, Math.floor(R.length / 10));
    const drift = Number.isFinite(rho) && Math.abs(rho) >= 0.1 && (rhoCI[0] > 0 || rhoCI[1] < 0);
    const use = drift ? R.map((v, i) => v - trend[i] + mean) : R;
    const raw = heterogeneity(R, sR), H = drift ? heterogeneity(use, sR) : raw;
    if (H) {
      const zdev = use.map((v, i) => (v - H.mu) / sR[i]);
      // 0 below, 1 within, 2 above the band the noise alone would allow (95 %)
      const cat = new Int16Array(n).fill(-1), cnt = [0, 0, 0];
      Q.forEach((p, i) => { const k2 = zdev[i] > 1.959964 ? 2 : zdev[i] < -1.959964 ? 0 : 1; cat[p] = k2; cnt[k2]++; });
      const moran = moranI(use, Q.map(p => pts.x[p]), Q.map(p => pts.y[p]));
      const rhoSize = spearman(Q.map(p => pts.ecd[p]), use);
      coat = { band: coatBand, n: Q.length, Q, R, sR, use, trend, w, zc, nC, raw, H, drift, rho, rhoCI, zdev, cat, below: cnt[0], within: cnt[1], above: cnt[2],
               spectra: groupSpectra(p => cat[p], 3),
               ends: [median(trend.slice(0, tenth)), median(trend.slice(-tenth))], moran, rhoSize, rhoSizeCI: spearmanCI(rhoSize, Q.length) };
    }
  }

  return (S.pfa = { key, n, fitted, kd, kq, snr, zG, atBound, det, quant, Q, Qd, label, k: cats, how, pops, la, laQ, noD, lambda, size, classes, coat,
                    vwRatio: sw > 0 ? swr / sw : NaN, nwRatio: Qd.length ? sr / Qd.length : NaN, rho, spectra, jD, ratio, rlab, nNd, popBand });
}

function renderPfAnalysis() {
  const A = pfAnalysis(), lab = A.rlab;
  const popColour = l => (A.pops[l] && A.pops[l].nd ? T.label : PARTICLE_COLOURS[l % PARTICLE_COLOURS.length]);
  const card = (k, v, s) => el("div", { class: "card" }, el("div", { class: "k" }, k), el("div", { class: "v" }, v), el("div", { class: "s" }, s));
  const nDet = A.det.reduce((s, v) => s + v, 0), nQ = A.Q.length, nBound = A.atBound.reduce((s, v) => s + v, 0);
  const [dlo, dhi] = wilson(nDet, A.fitted);
  $("#pf-detect-cards").replaceChildren(
    card("Spectra", `${fmt(A.fitted)} of ${fmt(A.n)}`, A.n > A.fitted ? `${fmt(A.n - A.fitted)} without a spectrum (flat or missing)` : "every particle measured"),
    card("Carbon detected", `${fmt(nDet)} of ${fmt(A.fitted)}`, `${pct(nDet / A.fitted)} (95 % CI ${pct(dlo)}–${pct(dhi)}) · G ≥ ${A.kd} SE`),
    card("Quantified", fmt(nQ), `G ≥ ${A.kq} × noise, inside its bounds`),
    card("Low signal", fmt(nDet - A.Q.filter(p => A.det[p]).length), "detected, not quantified"),
    card("G on a fit bound", fmt(nBound), "fitting noise: left out"));

  renderCoating(A, card);

  // populations
  $(".pf-cut").classList.toggle("hidden", $("#pf-pop-mode").value !== "cut");
  $("#pf-cut-label").textContent = `${lab} cut-offs`;
  const t = $("#pf-pop-table"); t.textContent = "";
  if (A.k) {
    t.append(el("tr", {}, ...["Population", "n", "By number", "By volume", `${lab} median`, "G width", "La (nm)"].map(h => el("th", {}, h))));
    const ci = (v, lo, hi) => [el("div", {}, pct(v)), el("div", { class: "sub" }, `${pct(lo)}–${pct(hi)}`)];
    A.pops.forEach((pp, l) => t.append(el("tr", {},
      el("td", {}, el("span", { class: "sw", style: `background:${popColour(l)}` }), el("b", {}, pp.name)),
      el("td", {}, fmt(pp.n)), el("td", {}, ...ci(pp.share, pp.ci[0], pp.ci[1])),
      el("td", {}, ...(pp.vol ? ci(pp.vol[0], pp.vol[1], pp.vol[2]) : ["–"])),
      el("td", { class: pp.nd ? "sub" : "" }, pp.nd ? "below detection" : fmt(pp.ratio, 3)), el("td", {}, fmt(pp.gw, 3)), el("td", {}, fmt(pp.la, 3)))));
  }
  const ndNote = A.nNd ? ` ${A.popBand} n.d.: quantified particles whose ${A.popBand} is below ${A.kd} × noise (${pct(A.nNd / nQ)}); their ${lab} is noise, so they are counted apart.` : "";
  $("#pf-pop-note").textContent = !A.ratio ? `The fit has no ${A.popBand} band.` : !nQ ? "No quantified particles." : A.k
    ? `${A.how}${ndNote} Shares of the quantified particles by number: Wilson 95 % intervals. By volume: the ParticleFinder volume of each particle, with a bootstrap 95 % interval over particles (500 resamples, populations held fixed); a mass share only if every population has the same density.`
    : $("#pf-pop-mode").value === "none" ? "" : A.how;

  // crystallite size
  const lam = A.lambda, fromFile = S.ds.laser_nm;
  const laserNote = fromFile ? `the file records ${fromFile} nm` : "the file does not record it";
  const ls = Float64Array.from(A.laQ).sort();
  const below = A.laQ.filter(v => v < 2).length, above = A.laQ.filter(v => v > 500).length;
  $("#pf-la-cards").replaceChildren(
    card("Median La", Number.isFinite(lam) && ls.length ? `${fmt(quantileSorted(ls, 0.5), 3)} nm` : "–",
      ls.length ? `IQR ${fmt(quantileSorted(ls, 0.25), 3)}–${fmt(quantileSorted(ls, 0.75), 3)} nm · n = ${fmt(ls.length)}` : "no particle with a D band"),
    card("No D band", fmt(A.noD), "D1 below detection: La larger than the method shows"),
    card("Outside the calibration", fmt(below + above), `${fmt(below)} below 2 nm · ${fmt(above)} above 500 nm`),
    card("Laser", Number.isFinite(lam) ? `${fmt(lam, 5)} nm` : "–", `532 nm edge by default; ${laserNote}`));
  $("#pf-la-note").textContent = A.jD < 0 ? "The fit has no D1 band." :
    "La = 2.4 × 10⁻¹⁰ λ⁴ (A_D1 / A_G)⁻¹ nm, λ in nm, from the fitted band areas (Cançado et al., Appl. Phys. Lett. 88, 163106, 2006). Calibrated on nanographite with La of about 20 to 500 nm; below about 2 nm the relation inverts (Ferrari and Robertson, Phys. Rev. B 61, 14095, 2000), so smaller values describe amorphous carbon only qualitatively.";

  // size
  const st = $("#pf-size-table"); st.textContent = "";
  st.append(el("tr", {}, ...["Diameter (µm)", "D10", "D50", "D90"].map(h => el("th", {}, h))),
    el("tr", {}, el("td", {}, "By number"), ...A.size.num.map(v => el("td", {}, fmt(v, 3)))),
    el("tr", {}, el("td", {}, "By volume"), ...A.size.vol.map(v => el("td", {}, fmt(v, 3)))));
  const ct = $("#pf-sizeclass-table"); ct.textContent = "";
  if (A.classes.length) {
    ct.append(el("tr", {}, ...["Diameter class (µm)", "n", `${lab} median`, "IQR", "La (nm)"].map(h => el("th", {}, h))));
    for (const cl of A.classes) ct.append(el("tr", {}, el("td", {}, `${fmt(cl.lo, 3)}–${fmt(cl.hi, 3)}`), el("td", {}, fmt(cl.n)),
      el("td", {}, fmt(cl.ratio, 3)), el("td", { class: "sub" }, `${fmt(cl.q1, 3)}–${fmt(cl.q3, 3)}`), el("td", {}, fmt(cl.la, 3))));
  }
  $("#pf-size-note").textContent = `D10, D50, D90: every particle, by number and weighted by the ParticleFinder volume. Classes: quartiles of diameter among the quantified particles with ${A.popBand} detected.` +
    (Number.isFinite(A.vwRatio) ? ` ${lab}: ${fmt(A.nwRatio, 3)} mean by number, ${fmt(A.vwRatio, 3)} weighted by volume; Spearman ρ with diameter ${Number.isFinite(A.rho) ? A.rho.toFixed(3) : "–"}.` : "");

  // the map: particles coloured by coating, by population, or by the fitted value
  const by = $("#pf-colour").value;
  if (by === "coat" && A.coat) {
    pmap.setCategories({ index: A.coat.cat, colours: COAT_COLOURS(), names: ["below", "within", "above"] });
    pmap.setTitle(`Coating <u>${escHTML(A.coat.band)}/G</u> against the noise band`);
  } else if (by === "pop" && A.k > 0) {
    pmap.setCategories({ index: A.label, colours: A.pops.map((_, l) => popColour(l)), names: A.pops.map(pp => pp.name) });
    pmap.setTitle(`Populations on <u>${escHTML(lab)}</u>`);
  } else pmap.setCategories(null);
  drawPopSpectra(); drawPfDist();
}

const pfSpecHost = $("#pf-spec"), pfDistHost = $("#pf-dist-plot");
new ResizeObserver(() => { if (S.pfa) drawPopSpectra(); }).observe(pfSpecHost);
new ResizeObserver(() => { if (S.pfa) drawPfDist(); }).observe(pfDistHost);

// small axes for the two plots under the map
function pfAxes(host, xr, yr, { logx = false, xlabel = "", ylabel = "" } = {}) {
  const W = host.clientWidth, H = host.clientHeight; if (W < 40 || H < 40) return null;
  const ctx = hidpi(host.querySelector("canvas"), W, H);
  const L = 44, R = 10, Tp = 10, B = 34, X = v => L + ((logx ? Math.log10(v) : v) - xr[0]) / (xr[1] - xr[0]) * (W - L - R), Y = v => Tp + (1 - (v - yr[0]) / (yr[1] - yr[0])) * (H - Tp - B);
  ctx.font = font(11); ctx.textBaseline = "middle"; ctx.textAlign = "right";
  for (const t of niceTicks(yr[0], yr[1], 4)) { ctx.fillStyle = T.grid; ctx.fillRect(L, Y(t), W - L - R, 1); ctx.fillStyle = T.ink; ctx.fillText(fmt(t, 3), L - 4, Y(t)); }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const xt = logx ? [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000].filter(v => Math.log10(v) >= xr[0] && Math.log10(v) <= xr[1]) : niceTicks(xr[0], xr[1], 6);
  for (const t of xt) { ctx.fillStyle = T.grid; ctx.fillRect(X(t), Tp, 1, H - Tp - B); ctx.fillStyle = T.ink; ctx.fillText(fmt(t, 3), X(t), H - B + 4); }
  ctx.fillStyle = T.label; ctx.fillText(xlabel, (L + W - R) / 2, H - 15);
  if (ylabel) { ctx.save(); ctx.translate(12, (Tp + H - B) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle"; ctx.fillText(ylabel, 0, 0); ctx.restore(); }
  return { ctx, X, Y, W, H, L, R, Tp, B };
}

function drawPopSpectra() {
  const A = S.pfa; if (!A) return;
  const byCoat = $("#pf-colour").value === "coat" && A.coat;
  $("#pf-spec-h").textContent = byCoat ? `Mean spectrum by coating group (${A.coat.band}/G against the noise band)` : "Mean spectrum by population";
  const sp = byCoat ? A.coat.spectra : A.spectra; if (!sp.wn.length) return;
  const colOf = g => (byCoat ? COAT_COLOURS()[g] : A.k ? (A.pops[g].nd ? T.label : PARTICLE_COLOURS[g % PARTICLE_COLOURS.length]) : T.strong);
  const nameOf = g => (byCoat ? ["below", "within", "above"][g] : A.k ? A.pops[g].name : "All quantified");
  let ymax = 0; sp.mean.forEach((mm, g) => mm.forEach((v, i) => { if (sp.n[g]) ymax = Math.max(ymax, v + sp.sd[g][i]); }));
  const ax = pfAxes(pfSpecHost, [sp.wn[0], sp.wn[sp.wn.length - 1]], [-0.05, Math.min(3, ymax * 1.05) || 1.2], { xlabel: "Raman shift (cm⁻¹)", ylabel: "Intensity / G height" }); if (!ax) return;
  const { ctx, X, Y } = ax;
  sp.mean.forEach((mm, g) => {
    if (!sp.n[g]) return;
    const col = colOf(g);
    ctx.globalAlpha = 0.16; ctx.fillStyle = col; ctx.beginPath();
    sp.wn.forEach((w, i) => (i ? ctx.lineTo(X(w), Y(mm[i] + sp.sd[g][i])) : ctx.moveTo(X(w), Y(mm[i] + sp.sd[g][i]))));
    for (let i = sp.wn.length - 1; i >= 0; i--) ctx.lineTo(X(sp.wn[i]), Y(mm[i] - sp.sd[g][i]));
    ctx.fill(); ctx.globalAlpha = 1;
    ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.beginPath();
    sp.wn.forEach((w, i) => (i ? ctx.lineTo(X(w), Y(mm[i])) : ctx.moveTo(X(w), Y(mm[i]))));
    ctx.stroke();
  });
  ctx.font = font(11, "bold"); ctx.textAlign = "left"; ctx.textBaseline = "top";
  sp.mean.forEach((_, g) => {
    if (!sp.n[g]) return;
    ctx.fillStyle = colOf(g);
    ctx.fillText(`${nameOf(g)} · n = ${fmt(sp.n[g])}`, ax.L + 8, ax.Tp + 4 + g * 15);
  });
}

function drawPfDist() {
  const A = S.pfa; if (!A) return;
  const kind = $("#pf-dist").value;
  if (kind === "funnel" || kind === "order" || kind === "dev") { drawCoatPlot(A, kind); return; }
  if (kind === "la") {
    const v = A.laQ; $("#pf-dist-note").textContent = v.length ? "quantified particles with a D band, log scale" : "";
    if (!v.length) { const ax = pfAxes(pfDistHost, [0, 3], [0, 1], { logx: true, xlabel: "La (nm)" }); if (ax) { ax.ctx.fillStyle = T.label; ax.ctx.textAlign = "center"; ax.ctx.fillText(Number.isFinite(A.lambda) ? "No particle with a D band" : "Enter the laser wavelength", ax.W / 2, ax.H / 2 - 10); } return; }
    const lv = v.map(x => Math.log10(x)), s = Float64Array.from(lv).sort(), x0 = Math.min(Math.log10(1), quantileSorted(s, 0.005)), x1 = Math.max(Math.log10(600), quantileSorted(s, 0.995));
    const nb = 40, groups = A.k || 1, H = Array.from({ length: groups }, () => new Float64Array(nb));
    A.Q.forEach(p => { const x = A.la[p]; if (!Number.isFinite(x)) return; const b = Math.floor((Math.log10(x) - x0) / (x1 - x0) * nb); if (b >= 0 && b < nb) H[A.k ? A.label[p] : 0][b]++; });
    let mx = 0; H.forEach(h => h.forEach(c => { mx = Math.max(mx, c); }));
    const ax = pfAxes(pfDistHost, [x0, x1], [0, mx * 1.1 || 1], { logx: true, xlabel: "La (nm)", ylabel: "Particles" }); if (!ax) return;
    const { ctx, X, Y } = ax;
    // the calibrated range
    ctx.setLineDash([4, 3]); ctx.strokeStyle = T.label;
    for (const lim of [2, 500]) if (Math.log10(lim) > x0 && Math.log10(lim) < x1) { ctx.beginPath(); ctx.moveTo(X(lim), ax.Tp); ctx.lineTo(X(lim), ax.H - ax.B); ctx.stroke(); }
    ctx.setLineDash([]);
    H.forEach((h, g) => {
      ctx.strokeStyle = A.k ? (A.pops[g].nd ? T.label : PARTICLE_COLOURS[g % PARTICLE_COLOURS.length]) : T.strong; ctx.lineWidth = 1.8; ctx.beginPath();
      for (let b = 0; b < nb; b++) {
        const xa = 10 ** (x0 + b / nb * (x1 - x0)), xb = 10 ** (x0 + (b + 1) / nb * (x1 - x0));
        if (b) ctx.lineTo(X(xa), Y(h[b])); else ctx.moveTo(X(xa), Y(h[b]));
        ctx.lineTo(X(xb), Y(h[b]));
      }
      ctx.stroke();
    });
    return;
  }
  // diameter: cumulative share by number and by volume
  $("#pf-dist-note").textContent = "cumulative, every particle";
  const pts = S.points, idx = []; for (let p = 0; p < S.nx; p++) if (pts.ecd[p] > 0) idx.push(p);
  if (!idx.length) return;
  idx.sort((a, b) => pts.ecd[a] - pts.ecd[b]);
  const x0 = Math.log10(pts.ecd[idx[0]]), x1 = Math.log10(pts.ecd[idx[idx.length - 1]]) + 1e-9;
  const ax = pfAxes(pfDistHost, [x0, x1], [0, 100], { logx: true, xlabel: "Diameter, circle equivalent (µm)", ylabel: "Cumulative %" }); if (!ax) return;
  const { ctx, X, Y } = ax;
  let tv = 0; for (const p of idx) tv += pts.volume[p] > 0 ? pts.volume[p] : 0;
  const line = (col, val) => {
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath();
    let cum = 0;
    idx.forEach((p, i) => { cum += val(p); const x = X(pts.ecd[p]), y = Y(100 * cum); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
    ctx.stroke();
  };
  line(PARTICLE_COLOURS[0], () => 1 / idx.length);
  if (tv > 0) line(PARTICLE_COLOURS[1], p => (pts.volume[p] > 0 ? pts.volume[p] : 0) / tv);
  ctx.font = font(11, "bold"); ctx.textAlign = "left"; ctx.textBaseline = "top";
  ctx.fillStyle = PARTICLE_COLOURS[0]; ctx.fillText(`By number · D50 ${fmt(A.size.num[1], 3)} µm`, ax.L + 8, ax.Tp + 4);
  if (tv > 0) { ctx.fillStyle = PARTICLE_COLOURS[1]; ctx.fillText(`By volume · D50 ${fmt(A.size.vol[1], 3)} µm`, ax.L + 8, ax.Tp + 19); }
}

// below / within / above the noise band
const COAT_COLOURS = () => [PARTICLE_COLOURS[2], PARTICLE_COLOURS[0], PARTICLE_COLOURS[1]];
const pFmt = p => (!Number.isFinite(p) ? "–" : p < 0.001 ? "< 0.001" : `= ${p.toFixed(3)}`);

function renderCoating(A, card) {
  const C = A.coat, box = $("#pf-verdict");
  if (!C) {
    box.className = "verdict"; box.textContent = "";
    $("#pf-coat-cards").replaceChildren();
    $("#pf-coat-note").textContent = A.Q.length < 10 ? "Too few quantified particles to judge the coating." : "Choose a coating band other than G.";
    return;
  }
  const H = C.H, b = C.band, cv = H.tau / H.mu, [wl, wh] = wilson(C.nC, C.n);
  let level, head;
  if (H.p >= 0.05) { level = "good"; head = "Uniform within the measurement"; }
  else if (H.I2 < 0.25) { level = "good"; head = "Nearly uniform"; }
  else if (H.I2 < 0.75) { level = "warn"; head = "Moderately non-uniform"; }
  else { level = "bad"; head = "Strongly non-uniform"; }
  const body = H.p >= 0.05
    ? `The particle-to-particle spread of ${b}/G is what the measurement noise alone gives. Any real variation between particles is below a CV of ${pct(H.tauHi / H.mu)} (upper 95 % limit).`
    : `${pct(H.I2)} of the particle-to-particle spread of ${b}/G is real (I², p ${pFmt(H.p)}); the rest is measurement noise. Between particles the coating index varies with a CV of ${pct(cv)} (95 % CI ${pct(H.tauLo / H.mu)}–${pct(H.tauHi / H.mu)}) around ${fmt(H.mu, 3)}.`;
  const more = [`${b} is detected in ${pct(C.nC / C.n)} of the quantified particles (${fmt(C.nC)} of ${fmt(C.n)}).`];
  if (C.nC / C.n < 0.2) more.push(`That is few: these numbers rest mostly on noise, and a longer acquisition per particle would tell more.`);
  if (C.drift) more.push(`${b}/G also drifts along the particle order (ρ = ${C.rho.toFixed(2)}, from ${fmt(C.ends[0], 3)} to ${fmt(C.ends[1], 3)}). The drift is removed before the numbers above (without removing it: CV ${pct(C.raw.tau / C.raw.mu)}, I² ${pct(C.raw.I2)}). Check the instrument (laser power, focus, background) before reading it as a property of the sample.`);
  if (C.moran && C.moran.p < 0.05 && C.moran.I > 0.05) more.push(`Neighbouring particles are ${C.moran.I < 0.2 ? "slightly " : ""}more alike than chance (Moran's I ${C.moran.I.toFixed(2)}): ${C.moran.I < 0.2 ? "a small" : "a"} part of the variation is across the sample rather than from particle to particle.`);
  if (Number.isFinite(C.rhoSize) && (C.rhoSizeCI[0] > 0 || C.rhoSizeCI[1] < 0)) more.push(`${b}/G ${C.rhoSize > 0 ? "rises" : "falls"} with particle size (ρ = ${C.rhoSize.toFixed(2)}).`);
  box.className = `verdict ${level}`;
  box.replaceChildren(el("b", {}, head), el("div", {}, body), ...more.map(t => el("p", {}, t)));

  $("#pf-coat-cards").replaceChildren(
    card(`Mean ${b}/G`, fmt(H.mu, 3), `median ${fmt(median(C.use), 3)} · n = ${fmt(C.n)}`),
    card("Particle-to-particle CV", pct(cv), `95 % CI ${pct(H.tauLo / H.mu)}–${pct(H.tauHi / H.mu)} · noise taken out`),
    card("Real share of the spread", pct(H.I2), `I² · p ${pFmt(H.p)}`),
    card("Spread seen vs noise", `${pct(H.sd / H.mean)} vs ${pct(H.noise / H.mean)}`, "CV of the values · what noise alone gives"),
    card(`${b} detected`, `${fmt(C.nC)} of ${fmt(C.n)}`, `${pct(C.nC / C.n)} (95 % CI ${pct(wl)}–${pct(wh)})`),
    card("Outside the noise band", `${pct(C.above / C.n)} above · ${pct(C.below / C.n)} below`, "2.5 % each if uniform"),
    card("Drift along the particle order", Number.isFinite(C.rho) ? `ρ = ${C.rho.toFixed(2)}` : "–", C.drift ? `${fmt(C.ends[0], 3)} → ${fmt(C.ends[1], 3)}, removed` : "none found"),
    card("Spatial clustering", C.moran ? `Moran's I ${C.moran.I.toFixed(2)}` : "–", C.moran ? `p ${pFmt(C.moran.p)} · ${C.moran.k} nearest neighbours` : ""),
    card("With particle size", Number.isFinite(C.rhoSize) ? `ρ = ${C.rhoSize.toFixed(2)}` : "–", Number.isFinite(C.rhoSizeCI[0]) ? `95 % CI ${C.rhoSizeCI[0].toFixed(2)} to ${C.rhoSizeCI[1].toFixed(2)}` : ""));

  $("#pf-coat-note").textContent =
    `How: ${b}/G is each particle's coating index, with its standard error from the particle's own fit (the delta method on the fitted heights). ` +
    `If the coating were the same on every particle, the values would scatter only by those errors; Cochran's Q tests that, I² is the share of the spread beyond them, ` +
    `and the CV is the SD of the true values between particles (Paule-Mandel, with its Q-profile 95 % interval) over their mean. ` +
    (C.drift ? `A drift along the particle order is removed first (running median of ${C.w} particles). ` : "") +
    `Limits: ParticleFinder records one spot per particle, so this compares particles; coverage within one particle needs a map of it. ` +
    `${b}/G rises with the amount of amorphous carbon in the probed volume but not in proportion to its thickness, so compare samples measured the same way.`;
}

function drawCoatPlot(A, kind) {
  const C = A.coat;
  if (!C) { const ax = pfAxes(pfDistHost, [0, 1], [0, 1], {}); if (ax) { ax.ctx.fillStyle = T.label; ax.ctx.textAlign = "center"; ax.ctx.fillText("No coating band to analyse", ax.W / 2, ax.H / 2 - 10); } return; }
  const H = C.H, CC = COAT_COLOURS(), b = C.band, alpha = Math.max(0.2, Math.min(0.8, 400 / C.n));
  const qs = a => Float64Array.from(a).filter(Number.isFinite).sort();
  if (kind === "funnel") {
    $("#pf-dist-note").textContent = `inside the funnel: what one uniform coating would give${C.drift ? " · drift removed" : ""}`;
    const sx = qs(C.sR), sy = qs(C.use), xmax = quantileSorted(sx, 0.995) * 1.05 || 1;
    const ylo = Math.min(quantileSorted(sy, 0.005), H.mu - 1.96 * xmax), yhi = Math.max(quantileSorted(sy, 0.995), H.mu + 1.96 * xmax);
    const ax = pfAxes(pfDistHost, [0, xmax], [ylo, yhi], { xlabel: `Standard error of ${b}/G`, ylabel: `${b}/G` }); if (!ax) return;
    const { ctx, X, Y } = ax;
    ctx.save(); ctx.beginPath(); ctx.rect(ax.L, ax.Tp, ax.W - ax.L - ax.R, ax.H - ax.Tp - ax.B); ctx.clip();
    ctx.globalAlpha = alpha;
    C.Q.forEach((p, i) => { ctx.fillStyle = CC[C.cat[p]]; ctx.fillRect(X(C.sR[i]) - 1.3, Y(C.use[i]) - 1.3, 2.6, 2.6); });
    ctx.globalAlpha = 1;
    ctx.strokeStyle = T.strong; ctx.lineWidth = 1.6; ctx.beginPath();
    ctx.moveTo(X(0), Y(H.mu)); ctx.lineTo(X(xmax), Y(H.mu + 1.96 * xmax)); ctx.moveTo(X(0), Y(H.mu)); ctx.lineTo(X(xmax), Y(H.mu - 1.96 * xmax)); ctx.stroke();
    ctx.setLineDash([5, 4]); ctx.strokeStyle = T.label; ctx.beginPath();
    for (const s of [1, -1]) for (let i = 0; i <= 40; i++) { const x = xmax * i / 40, y = H.mu + s * 1.96 * Math.sqrt(x * x + H.tau * H.tau); if (i) ctx.lineTo(X(x), Y(y)); else ctx.moveTo(X(x), Y(y)); }
    ctx.stroke(); ctx.setLineDash([]);
    ctx.restore();
    ctx.font = font(11, "bold"); ctx.textAlign = "left"; ctx.textBaseline = "top";
    [[`above · ${pct(C.above / C.n)}`, CC[2]], [`within · ${pct(C.within / C.n)}`, CC[1]], [`below · ${pct(C.below / C.n)}`, CC[0]]].forEach(([t, col], i) => { ctx.fillStyle = col; ctx.fillText(t, ax.L + 8, ax.Tp + 4 + i * 15); });
    ctx.fillStyle = T.label; ctx.font = font(10); ctx.fillText("solid: noise only · dashed: with the real spread", ax.L + 8, ax.Tp + 50);
    return;
  }
  if (kind === "order") {
    $("#pf-dist-note").textContent = `measured values, running median of ${C.w}${C.drift ? ": a drift" : ""}`;
    const sy = qs(C.R), ylo = quantileSorted(sy, 0.005), yhi = quantileSorted(sy, 0.995);
    const ax = pfAxes(pfDistHost, [1, S.nx], [ylo, yhi], { xlabel: "Particle number (the order measured)", ylabel: `${b}/G` }); if (!ax) return;
    const { ctx, X, Y } = ax;
    ctx.save(); ctx.beginPath(); ctx.rect(ax.L, ax.Tp, ax.W - ax.L - ax.R, ax.H - ax.Tp - ax.B); ctx.clip();
    ctx.globalAlpha = alpha;
    C.Q.forEach((p, i) => { ctx.fillStyle = CC[C.cat[p]]; ctx.fillRect(X(p + 1) - 1.2, Y(C.R[i]) - 1.2, 2.4, 2.4); });
    ctx.globalAlpha = 1;
    ctx.strokeStyle = T.strong; ctx.lineWidth = 2.2; ctx.beginPath();
    C.Q.forEach((p, i) => (i ? ctx.lineTo(X(p + 1), Y(C.trend[i])) : ctx.moveTo(X(p + 1), Y(C.trend[i]))));
    ctx.stroke(); ctx.restore();
    return;
  }
  // deviations from the mean in units of each particle's own error: N(0, 1) if uniform
  const lim = 6, nb = 48, h = new Float64Array(nb);
  let ss = 0; for (const z of C.zdev) { ss += z * z; const k = Math.floor((z + lim) / (2 * lim) * nb); if (k >= 0 && k < nb) h[k]++; }
  const sdz = Math.sqrt(ss / C.zdev.length), bw = 2 * lim / nb, dens = Array.from(h, c => c / (C.zdev.length * bw));
  $("#pf-dist-note").textContent = `SD ${sdz.toFixed(2)}: 1 if the coating were uniform`;
  const ymax = Math.max(0.42, ...dens) * 1.1;
  const ax = pfAxes(pfDistHost, [-lim, lim], [0, ymax], { xlabel: `(${b}/G − mean) / its standard error`, ylabel: "Density" }); if (!ax) return;
  const { ctx, X, Y } = ax;
  ctx.fillStyle = PARTICLE_COLOURS[0]; ctx.globalAlpha = 0.45;
  dens.forEach((d, k) => { const x0 = -lim + k * bw; ctx.fillRect(X(x0) + 0.5, Y(d), Math.max(1, X(x0 + bw) - X(x0) - 1), Y(0) - Y(d)); });
  ctx.globalAlpha = 1; ctx.strokeStyle = T.strong; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i <= 120; i++) { const z = -lim + 2 * lim * i / 120, y = Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI); if (i) ctx.lineTo(X(z), Y(y)); else ctx.moveTo(X(z), Y(y)); }
  ctx.stroke();
  ctx.font = font(11, "bold"); ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillStyle = T.strong;
  ctx.fillText("curve: one uniform coating", ax.L + 8, ax.Tp + 4);
}

for (const id of ["#pf-kd", "#pf-kq", "#pf-coat-band", "#pf-colour", "#pf-pop-ratio", "#pf-pop-mode", "#pf-cut1", "#pf-cut2", "#pf-laser"]) $(id).addEventListener("change", () => { if (S.points && S.fit) renderPF(); });
$("#pf-dist").onchange = () => drawPfDist();
$("#pf-csv").onclick = () => {
  if (!S.points || !S.fit) return;
  const A = pfAnalysis(), pts = S.points, ratio = A.ratio;
  const jG = S.fit.names.findIndex(b => b.toLowerCase() === "g"), m = S.fit.m;
  const num = v => (Number.isFinite(v) ? String(+(+v).toPrecision(6)) : "");
  const C = A.coat, ci = C ? new Map(C.Q.map((p, i) => [p, i])) : new Map(), cb = C ? C.band : "coating";
  const rows = [["Particle", "X (um)", "Y (um)", "Diameter (um)", "Volume (um3)", "G signal/noise", "G height/SE", "G on a bound", "Carbon detected", "Quantified",
    "Population", `${A.rlab} intensity`, "G width (cm-1)", "G position (cm-1)", "A_D1/A_G", "La (nm)",
    `${cb}/G`, `${cb}/G SE`, `${cb} height/SE`, `${cb}/G against the noise band`]];
  for (let p = 0; p < S.nx; p++) {
    const aD = A.jD >= 0 ? S.fit.area[p * m + A.jD] : NaN, aG = jG >= 0 ? S.fit.area[p * m + jG] : NaN;
    rows.push([String(p + 1), num(pts.x[p]), num(pts.y[p]), num(pts.ecd[p]), num(pts.volume[p]), num(A.snr[p]), num(A.zG[p]), A.atBound[p] ? "yes" : "no",
      A.det[p] ? "yes" : "no", A.quant[p] ? "yes" : "no", A.label[p] >= 0 && A.k ? A.pops[A.label[p]].name : "",
      ratio ? num(ratio[p]) : "", jG >= 0 ? num(S.fit.width[p * m + jG]) : "", jG >= 0 ? num(S.fit.position[p * m + jG]) : "", num(aD / aG), num(A.la[p]),
      ...(ci.has(p) ? [num(C.R[ci.get(p)]), num(C.sR[ci.get(p)]), num(C.zc[ci.get(p)]), ["below", "within", "above"][C.cat[p]]] : ["", "", "", ""])]);
  }
  download(`${S.ds.name}_particles.csv`, rows.map(r => r.join(",")).join("\n") + "\n", "text/csv");
};

// =========================================================================
// report
// =========================================================================
// Report figures are drawn off-screen at a fixed size, so they are current
// whichever tab is open, on a tile in the current theme's plot colour.
function offscreen(w, h) {
  const host = el("div", { style: `position:fixed;left:-10000px;top:0;width:${w}px;height:${h}px;display:flex;flex-direction:column` });
  document.body.append(host);
  return host;
}
function onDark(canvas) {
  const c = document.createElement("canvas"); c.width = canvas.width; c.height = canvas.height;
  const g = c.getContext("2d"); g.fillStyle = T.paper; g.fillRect(0, 0, c.width, c.height); g.drawImage(canvas, 0, 0);
  return c.toDataURL("image/png");
}
function reportFigures() {
  const figs = { maps: [] };
  let overlay = null;
  maps.forEach((src, i) => {
    const host = offscreen(520, 440), mv = new MapView(host, {});
    Object.assign(mv, { cmap: src.cmap, scale: src.scale, manual: src.manual, smooth: src.smooth, filter: src.filter });
    mv.setGeometry({ nx: S.nx, ny: S.ny, x: S.x, y: S.y, points: S.points });
    const r = resolve(S.specs[i]);
    mv.setTitle(r.title);
    if (r.values) mv.setData(r.values, { valid: r.useValid ? S.valid : null, label: r.label, unit: r.unit });
    mv.setCursor(null, S.pinned);
    figs.maps.push({ url: onDark(mv.canvas), title: mv.titleEl.textContent.replace("▾", "").trim() });
    if (i === S.active) overlay = mv; else host.remove();
  });
  let host = offscreen(760, 360);
  const sp = new SpectrumPlot(host);
  if (S.pinned != null && S.cube) {
    const sm = fitCube(), raw = S.cube.subarray(S.pinned * S.k, (S.pinned + 1) * S.k);
    sp.setData({ wn: S.wn, raw: sm ? sm.subarray(S.pinned * S.k, (S.pinned + 1) * S.k) : raw, raw0: sm ? raw : null, range: S.fit.range, fit: evalFit(S.pinned) });
  } else if (S.pinned != null && S.fit && S.fit.resultsOnly) sp.setData(modelSpectrum(S.pinned));
  figs.cursor = onDark(sp.canvas); host.remove();
  host = offscreen(520, 360);
  const hp = new HistogramPlot(host); hp.setData(hist.d); figs.hist = onDark(hp.canvas); host.remove();
  const vc = document.createElement("canvas"); drawVideo(vc, 520, 400, overlay); figs.video = vc.toDataURL("image/png");
  overlay.host.remove();
  return figs;
}

function makeReport() {
  if (!S.fit) { $("#report-paper").replaceChildren(el("p", {}, "Open and fit a map first.")); return; }
  buildReport($("#report-paper"), {
    title: $("#rp-title").value, ds: S.ds, cfg: S.cfg, fit: S.fit, nx: S.nx, ny: S.ny, x: S.x, y: S.y,
    figs: reportFigures(), quality: $("#quality"), stats: statRows(),
    pinned: S.pinned, pstats: S.pstats, splat: splatState().result, splatImg: sview && splatState().field ? sview.snapshot() : null,
  });
}
$("#rp-refresh").onclick = makeReport;
$("#rp-title").onchange = makeReport;
$("#rp-print").onclick = () => { makeReport(); setTimeout(() => window.print(), 150); };

// =========================================================================
// light / dark theme
// =========================================================================
function themeButton() {
  const next = currentTheme() === "light" ? "dark" : "light";
  $("#theme-toggle").title = `Switch to ${next} mode`;
}
$("#theme-toggle").onclick = () => setTheme(currentTheme() === "light" ? "dark" : "light");
window.addEventListener("themechange", () => { themeButton(); repaintAll(); });
// canvas text is drawn in the web font: draw everything again once it has loaded
if (document.fonts) Promise.all(["400 12px Inter", "600 12px Inter"].map(f => document.fonts.load(f)))
  .then(() => { moveInk(); fitMapRow(); repaintAll(); }).catch(() => {});
function repaintAll() {
  buildPeaks();                                   // band swatches
  for (const m of [...maps, pmap, sliceData, sliceFieldView]) { m._dirty = true; m.render(); }
  if (S.fit) refreshAll(); else { renderCursor(); renderVideo(); }
  if (S.cube) renderDensity();
  if (S.seg) { drawStrip(); renderParticles(); }
  const st = splatState(); if (st.history) drawLoss(st.history);
}
themeButton();

// =========================================================================
// boot
// =========================================================================
(async function boot() {
  try {
    // no Python server behind this page (the web build): run the engine in the page
    const server = await fetch("/api/health").then(r => r.ok && (r.headers.get("content-type") || "").includes("json")).catch(() => false);
    if (!server) {
      const { createLocalBackend } = await import("./engine/local-api.js");
      setLocalBackend(createLocalBackend());
      document.body.classList.add("browser-only");
    }
    S.health = await api("/api/health");
    S.defaultCfg = await api("/api/config/default");
    S.cfg = structuredClone(S.defaultCfg);
    fillSide();
    setActive(0);
    moveInk();
    renderRecent();
    if (!S.health.splat) {
      $("#sp-status").textContent = S.health.splat_why;
      if (S.health.browser) {
        $("#gl-empty-h").textContent = "Splat 3D: work in progress";
        $("#gl-empty-p").textContent = "Spectral Gaussian splatting (each map as a field of 3-D Gaussians in x, y and Raman shift) is being brought to the web version. It is not available here yet.";
      }
      $("#sp-run").disabled = $("#sp-run2").disabled = true;
    }
  } catch (e) { setStatus("Server not reachable", "err"); toast(e.message, "error"); }
})();
