// Carbo Metrics: browser side. Holds the open map, drives the fit (on the
// Python server, or in the page when there is none), and derives everything
// the instrument shows (maps, ratios, valid points, quality, statistics) plus
// the splat and particle views.

import { smoothSpectra } from "./engine/core.js";
import { $, $$, el, api, apiUpload, videoSrc, setLocalBackend, isBrowserOnly, f32, download, debounce, fmt, summary, otsu, toast, hidpi, quantile, niceTicks } from "./util.js";
import { MapView } from "./mapview.js";
import { SpectrumPlot, HistogramPlot, DensityPlot, PEAK_COLOURS } from "./plots.js";
import { SplatView, sliceField } from "./splat3d.js";
import { segment, outlineOverlay, distance, PARTICLE_COLOURS } from "./particles.js";
import { buildReport } from "./report.js";
import { rng, wilson, weightedQuantile, laCancado, robustScale, bestGmm, bootstrapShares, median, quantileSorted,
         heightCovariances, ratioVariance, heterogeneity, moranI, spearmanCI, coverage, pairRatio,
         binomLower, certifyCount, detectionPower, signalFloor, gini, noiseGini, logCV, lognormalPreferred,
         gridMoran, effectiveN, variogram, fitVariogram, hotSpots, blockShuffle, floorWithReference, hexBins,
         boundedCoverage, lowShare, qcd, qRatio, evenness, moments, spatialPermutation, neighbourNoise, blockBootstrap,
         ksTest, fitFreeHeights, nestedSize, robustSD, dispersion, nmf, nmfProject, fitSplats, splatScale, gaussSmooth,
         kruskalWallis, brownForsythe, groupGap, thinTail, mixtureCoverage, olsHC3, nForMean, nForSpread, tQuantile } from "./pf-analysis.js";
import { PRESET_21OCT, QC_SETTINGS, BANDS as D3G_BANDS, prepare as d3gPrepare, unmix, freeFits, calibrate, standards as d3gStandards,
         blocksOf, blockMedian, compareMedians, mapBlock, stageBlock, toJSON as calToJSON, fromJSON as calFromJSON,
         refineOnMedian, standardRatio, scaleIndex } from "./d3g-calibration.js";
import { T, font, setTheme, currentTheme } from "./theme.js";
import { lut } from "./colormaps.js";

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
  onBackdrop: on => { for (const m of [...maps, pmap]) { m.backdropOn = on; m._ptsCache = null; m.render(); } },
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
    // particles drawn as their shapes once the map's cached image would be enlarged (and blurred)
    if (mv.points && k > mv.off.width / (E.x1 - E.x0)) mv._paintPoints(ctx, k, ox, oy, 1.5, { x0: rx0, y0: ry0 }, false, [rx0, ry0, rx1, ry1]);
    else ctx.drawImage(mv.off, sx(E.x0), sy(E.y0), (E.x1 - E.x0) * k, (E.y1 - E.y0) * k);
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
  S.ds = info; S.nx = info.nx; S.ny = info.ny; S.k = info.k; S.openedAt = Date.now();
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
  // particles are drawn over their camera image (a grid map covers its own)
  const back = S.points && S.video && info.video ? { img: S.video, ...info.video } : null;
  for (const m of [...maps, pmap]) { m.setGeometry({ nx: S.nx, ny: S.ny, x: S.x, y: S.y, points: S.points }); m.values = null; m.backdrop = back; }
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
  S.ds = info; S.nx = info.nx; S.ny = info.ny; S.k = 0; S.openedAt = Date.now();
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
  for (const m of [...maps, pmap]) { m.setGeometry({ nx: S.nx, ny: S.ny, x: S.x, y: S.y }); m.values = null; m.backdrop = null; }
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
  const ps = particleStats(); if (!ps) { renderD3g(); return; }
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
  renderSegCoating(card);
  renderCoverage(card);
  renderChecks(card); renderAcq(); renderAccept(card); renderXc(card); renderBetween(card);
  renderD3g();
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
// Coating uniformity on a map: inside each particle, and between particles
// =========================================================================
// Each pixel inside a particle whose G is quantified carries its coating index
// (the coating band over G) with that value's standard error from its own fit.
// coverage() tests each particle's pixels for one common value, pools that
// within-particle spread, and compares the particle means between particles.
// Each pixel is then placed against its own particle's mean and noise band.
const SEG_KD = 3, SEG_KQ = 10;

// the fitted ratio against one read off the same spectra without a fit: rank agreement (a ratio stored
// inverted, or a fit that follows something else, shows as rho near or below 0), the scale between them,
// and the spread of each (robust CV) with the part the fit adds of its own
function ratioAgreement(fit, free) {
  const cv = v => robustSD(v) / median(v), cvFit = cv(fit), cvFree = cv(free);
  return { rho: spearman(fit, free), ratio: median(free.map((v, i) => v / fit[i]).filter(Number.isFinite)), cvFit, cvFree,
           adds: cvFit > cvFree ? Math.sqrt(cvFit * cvFit - cvFree * cvFree) : 0 };
}

// The fitted ratios against ones read off the same spectra without a fit (fitFreeHeights): the coating
// band over G, and D1 over G, which a height without a fit reads cleanly. A weak band such as D3 sits on
// the tails of D1 and G, so its height without a fit follows D1/G as well; how far is measured too.
function fitFreeCheck(list, jG, jC, jD, detected, kd, kq) {
  const f = S.fit, m = f.m, pos = j => median(list.map(p => f.position[p * m + j]));
  const js = [jG, jC, ...(jD >= 0 && jD !== jC ? [jD] : [])];
  const h = fitFreeHeights(S.cube, S.k, S.wn, list, js.map(pos)); if (!h) return null;
  const fitR = j => list.map(p => f.intensity[p * m + j] / f.intensity[p * m + jG]), freeR = k => list.map((_, i) => h.heights[k][i] / h.heights[0][i]);
  const same = list.filter((p, i) => detected(p) === (h.heights[1][i] >= kd * h.se[i])).length;
  return { se: h.se, sigma: h.sigma, sBase: h.sBase, cPos: pos(jC), coat: ratioAgreement(fitR(jC), freeR(1)), d1: js.length > 2 ? ratioAgreement(fitR(jD), freeR(2)) : null,
           follows: js.length > 2 ? spearman(freeR(1), freeR(2)) : NaN, agree: same / list.length, accepted: list.filter((_, i) => h.heights[0][i] >= kq * h.se[i]).length };
}

// a band's position or width on its fit bound: the fit stopped there, so the band is fitting noise. A
// shape held for every spectrum (by the user, or at the mean spectrum's) is not fitted per spectrum,
// so it can sit on a bound without the spectrum's fit having stopped there
function bandOnBound(f, pk, p, j) {
  if (!pk || pk.fit === false || (f.held || []).includes(pk.name)) return false;
  const m = f.m, near = (v, b, span) => Math.abs(v - b) < 1e-3 * span, x = f.position[p * m + j], w = f.width[p * m + j];
  const sx = pk.pos_max - pk.pos_min, sw = pk.width_max - pk.width_min;
  return near(x, pk.pos_min, sx) || near(x, pk.pos_max, sx) || near(w, pk.width_min, sw) || near(w, pk.width_max, sw);
}

function segCoating() {
  const f = S.fit, c = S.cfg, seg = S.seg, fc = f.config || c;
  const sel = $("#seg-coat-band"), others = f.names.filter(b => b.toLowerCase() !== "g");
  if ([...sel.options].map(o => o.value).join() !== others.join()) {
    const keep = sel.value;
    sel.replaceChildren(...others.map(b => el("option", { value: b }, `${b}/G`)));
    sel.value = others.includes(keep) ? keep : (others.find(b => b.toLowerCase() === "d3") || others[0] || "");
  }
  const band = sel.value;
  if (f.resultsOnly) return { why: "This file holds fitted values without spectra, so no pixel has a standard error. Open the map itself (.l6m) to judge the coating." };
  if (fc.smooth_window || fc.spatial_kernel) return { why: "The spectra were smoothed before the fit, so a pixel's residual no longer measures its noise (and spatial smoothing makes neighbours share it). Switch the smoothing off in the Carbon fit tab to judge the coating." };
  const m = f.m, jG = f.names.findIndex(b => b.toLowerCase() === "g"), jC = f.names.indexOf(band);
  if (jG < 0 || jC < 0) return { why: "Choose a coating band; the fit needs a G band." };
  const rim = Math.max(0, +$("#seg-rim").value || 0), rimRef = rim || 1;
  const key = [S.ds.id, f.seconds, f.names.join(), stableJSON(VIEW_KEYS.map(k => c[k])), band, rim].join("|");
  if (S.segCoat && S.segCoat.key === key && S.segCoat.seg === seg) return S.segCoat;
  // the pixels used: G at least SEG_KQ x the residual noise, inside its bounds, passing the constraints
  const gp = fc.peaks.find(q => q.name === f.names[jG]), near = (v, b, span) => Math.abs(v - b) < 1e-3 * span;
  const onBound = p => {
    if (!gp) return false;
    const x = f.position[p * m + jG], w = f.width[p * m + jG], sx = gp.pos_max - gp.pos_min, sw = gp.width_max - gp.width_min;
    return near(x, gp.pos_min, sx) || near(x, gp.pos_max, sx) || near(w, gp.width_min, sw) || near(w, gp.width_max, sw);
  };
  const quant = seg.parts.map(pt => pt.pixels.filter(p => f.std_error[p] > 0 && f.intensity[p * m + jG] >= SEG_KQ * f.std_error[p] && !onBound(p) && (!S.valid || S.valid[p])));
  // and not on the rim, where the laser spot lies partly off the particle: depth is the
  // distance from the pixel's centre to the particle's edge
  const step = Math.min(S.nx > 1 ? Math.abs(S.x[1] - S.x[0]) : Infinity, S.ny > 1 ? Math.abs(S.y[1] - S.y[0]) : Infinity);
  // the map border is not a rim: where the map cuts a particle, the particle goes on beyond it
  if (!S.segDepth || S.segDepth.seg !== seg) S.segDepth = { seg, d: distance(Uint8Array.from(seg.labels, l => (l ? 1 : 0)), S.nx, S.ny, true) };
  const depth = p => (S.segDepth.d[p] - 0.5) * step;
  const groups = quant.map(g => g.filter(p => depth(p) >= rim)), rimPx = quant.map(g => g.filter(p => depth(p) < rimRef));
  const used = groups.flat();
  // spectra coverage cannot judge, counted to bound it: G on a fit bound (left out above),
  // or the coating band with a parameter on a bound or fitted with zero height
  const cp = fc.peaks.find(q => q.name === band);
  const cOut = p => f.intensity[p * m + jC] <= 0 || bandOnBound(f, cp, p, jC);
  const gOut = seg.parts.map(pt => pt.pixels.filter(p => f.std_error[p] > 0 && f.intensity[p * m + jG] >= SEG_KQ * f.std_error[p] && onBound(p) && (!S.valid || S.valid[p]) && depth(p) >= rim));
  const left = groups.map(g => g.filter(cOut));
  // band-height errors, once per fit and pixel (a new threshold only adds pixels)
  const ckey = `${S.ds.id}|${f.seconds}|${f.names.join()}`;
  if (!S.segCov || S.segCov.key !== ckey) S.segCov = { key: ckey, cov: new Float64Array(P() * m * m).fill(NaN), done: new Uint8Array(P()) };
  const fill = () => { const todo = used.filter(p => !S.segCov.done[p]); heightCovariances(f, fc, S.wn, todo, S.segCov.cov); for (const p of todo) S.segCov.done[p] = 1; };
  const nTodo = used.filter(p => !S.segCov.done[p]).length;
  if (nTodo > 1500) {
    // about half a millisecond a pixel: draw the rest of the tab first
    setTimeout(() => { if (S.segCov && S.segCov.key === ckey) fill(); if (S.seg === seg) renderParticles(); }, 30);
    return { why: `Working out the band-height errors of ${fmt(nTodo)} pixels…` };
  }
  if (nTodo) fill();
  const cov = S.segCov.cov, n = P();
  const y = new Float64Array(n).fill(NaN), se = new Float64Array(n).fill(NaN), z = new Float64Array(n).fill(NaN);
  for (const p of used) {
    const o = p * m * m, a = f.intensity[p * m + jC], g = f.intensity[p * m + jG];
    y[p] = a / g; se[p] = Math.sqrt(Math.max(0, ratioVariance(cov, o, m, jC, jG, a, g))); z[p] = a / Math.sqrt(cov[o + jC * m + jC]);
  }
  const C = coverage(y, se, groups);
  if (!C) return (S.segCoat = { key, seg, band, why: `No particle has 3 or more pixels with a quantified G${rim ? ` deeper than ${fmt(rim, 3)} µm inside it` : ""}.` });
  // the noise of the two band heights from neighbouring pixels, a check on the fits' errors
  const inMask = new Uint8Array(n), Dimg = new Float64Array(n), Gimg = new Float64Array(n);
  for (const p of used) { inMask[p] = 1; Dimg[p] = f.intensity[p * m + jC]; Gimg[p] = f.intensity[p * m + jG]; }
  const nnD = neighbourNoise(Dimg, inMask, S.nx, S.ny), nnG = neighbourNoise(Gimg, inMask, S.nx, S.ny);
  // the rim against the rest: the coating index over each particle's mean, G over its median
  let rs = 0, gs = 0, rn = 0;
  rimPx.forEach((g, k) => {
    const h = C.parts[k]; if (!h || !g.length) return;
    const gm = median(groups[k].map(p => f.intensity[p * m + jG]));
    for (const p of g) { const G = f.intensity[p * m + jG]; rs += f.intensity[p * m + jC] / G / h.mu; gs += G / gm; rn++; }
  });
  // each pixel against its own particle's mean: 0 below, 1 within, 2 above the 95 % noise band, 3 on the rim (left out)
  const cat = new Int16Array(n).fill(-1), lab = new Int32Array(n), cnt = [0, 0, 0];
  if (rim) for (const pt of seg.parts) for (const p of pt.pixels) if (depth(p) < rim) cat[p] = 3;
  const rows = seg.parts.map((pt, k) => {
    const h = C.parts[k], g = groups[k];
    const r = { id: pt.id, total: pt.pixels.length, used: g.length, h, det: g.filter(p => z[p] >= SEG_KD).length };
    if (!h) return r;
    for (const p of g) {
      if (!(se[p] > 0)) continue;
      const d = (y[p] - h.mu) / se[p], k2 = d > 1.959964 ? 2 : d < -1.959964 ? 0 : 1;
      cat[p] = k2; cnt[k2]++; lab[p] = pt.id;
    }
    return r;
  });
  // how far each particle runs off the map: the share of its outline on the map's border
  for (const [k, pt] of seg.parts.entries()) {
    let edge = 0, onBorder = 0;
    for (const p of pt.pixels) {
      const i = (p / S.nx) | 0, j = p - i * S.nx, l = seg.labels[p];
      const border = i === 0 || j === 0 || i === S.ny - 1 || j === S.nx - 1;
      if (border || seg.labels[p - 1] !== l || seg.labels[p + 1] !== l || seg.labels[p - S.nx] !== l || seg.labels[p + S.nx] !== l) { edge++; if (border) onBorder++; }
    }
    rows[k].cut = edge ? onBorder / edge : 0;
  }
  // beyond the CV, per particle with enough pixels: inequality against its own noise, the shape of the
  // distribution, how many pixels are worth independent values, the variogram (its nugget is a check on
  // the noise model that does not use it; its plateau the structured spread) and locally thin places
  // against a null that keeps the spatial correlation (whole blocks moved)
  for (const [k, g] of groups.entries()) {
    const r = rows[k];
    if (!r.h || g.length < 100) continue;
    let i0 = Infinity, i1 = -1, j0 = Infinity, j1 = -1;
    for (const p of g) { const i = (p / S.nx) | 0, j = p - i * S.nx; i0 = Math.min(i0, i); i1 = Math.max(i1, i); j0 = Math.min(j0, j); j1 = Math.max(j1, j); }
    const w = j1 - j0 + 1, hh = i1 - i0 + 1, img = new Float64Array(w * hh), sgi = new Float64Array(w * hh), ok = new Uint8Array(w * hh);
    const vals = [], ses = [];
    for (const p of g) { const i = (p / S.nx) | 0, j = p - i * S.nx, q = (i - i0) * w + (j - j0); img[q] = y[p]; sgi[q] = se[p]; ok[q] = 1; vals.push(y[p]); ses.push(se[p]); }
    const med = median(vals), noise2 = ses.reduce((t, v) => t + v * v, 0) / ses.length;
    const hmax = Math.max(4, Math.min(15, Math.floor(Math.min(w, hh) / 3))), vg = variogram(img, ok, w, hh, hmax);
    const fitC = fitVariogram(vg.lags, vg.classical, hmax), fitR = fitVariogram(vg.lags, vg.robust, hmax);
    const moran = gridMoran(img, ok, w, hh), hs = hotSpots(img, ok, w, hh), sh = blockShuffle(img, ok, w, hh, 8);
    const hsNull = sh ? hotSpots(sh.img, sh.ok, sh.nx, sh.ny) : null;
    r.u = { n: g.length, gini: gini(vals), giniNoise: noiseGini(med, ses), logn: lognormalPreferred(vals), lcv: logCV(vals),
            neff: effectiveN(g.length, moran), nugget: fitC ? fitC.nugget / noise2 : NaN,
            structured: fitR && med > 0 ? Math.sqrt(Math.max(fitR.sill - fitR.nugget, 0)) / med : NaN, range: fitR ? fitR.range * step : NaN, ranged: fitR ? fitR.identified : false,
            cold: hs.cold, coldNull: hsNull ? hsNull.cold : NaN,
            // more measures of the spread, the spatial tests by permutation, and intervals from 8 x 8 pixel blocks
            low: lowShare(vals), qcd: qcd(vals), qr: qRatio(vals), even: evenness(vals, 32), ...moments(vals),
            perm: spatialPermutation(img, ok, w, hh), boot: blockBootstrap(img, sgi, ok, w, hh) };
  }
  // pixel pairs against their errors: the nearest, about 1 µm and about 3 µm apart
  const lags = [...new Set([1, Math.round(1 / step), Math.round(3 / step)])].filter(L => L >= 1 && L < Math.max(S.nx, S.ny));
  const pairs = lags.map(L => ({ L, um: L * step, ...pairRatio(y, se, lab, S.nx, S.ny, L) })).filter(q => q.n >= 30);
  const nUsed = cnt[0] + cnt[1] + cnt[2], nDet = rows.reduce((s, r) => s + (r.h ? r.det : 0), 0);
  const nQuant = quant.reduce((s, g) => s + g.length, 0);
  // the coating band over G without a band fit, on the same pixels: a check on the fitted ratio
  const ff = S.cube && used.length >= 10 ? fitFreeCheck(used, jG, jC, f.names.findIndex(q => q.toLowerCase() === "d1"), p => z[p] >= SEG_KD, SEG_KD, SEG_KQ) : null;
  return (S.segCoat = { key, seg, band, C, rows, cat, px: { groups, y, se, left, gOut }, below: cnt[0], within: cnt[1], above: cnt[2], n: nUsed, nDet, step, pairs, rim,
                        nn: { sD: nnD.sd, sG: nnG.sd, n: nnD.n }, ff, outC: left.reduce((t, g) => t + g.length, 0), outG: gOut.reduce((t, g) => t + g.length, 0),
                        rimCmp: rn ? { ratio: rs / rn, g: gs / rn, n: rn, width: rimRef } : null, rimLeft: nQuant - used.length,
                        left: seg.parts.reduce((s, pt) => s + pt.pixels.length, 0) - nQuant, small: rows.filter(r => !r.h).length });
}

function renderSegCoating(card) {
  const A = segCoating(), box = $("#seg-verdict"), t = $("#seg-coat-table");
  const byCoat = $("#seg-colour").value === "coat" && A && A.C;
  if (byCoat) {
    pmap.setCategories({ index: A.cat, colours: [...COAT_COLOURS(), T.label], names: ["below", "within", "above", ...(A.rim ? ["rim, left out"] : [])] });
    pmap.setTitle(`Coating <u>${escHTML(A.band)}/G</u> against each particle's mean`);
  }
  // the cross-checks, once run: the decomposition against the ratio, or the splat ratio field
  const Xc = A && A.C ? xcNow() : null, col = $("#seg-colour").value;
  if (col === "decomp" && Xc && Xc.dec) {
    if (Xc.dec.ref) {
      pmap.setCategories({ index: Xc.dec.ref.cat, colours: XC_CAT(), names: ["coated both ways", "by the ratio only", "by the decomposition only", "neither"] });
      pmap.setTitle(`Coverage by <u>${escHTML(A.band)}/G</u> against the decomposition`);
    } else {
      const v = new Float64Array(P()).fill(NaN); Xc.dec.used.forEach((p, i) => { v[p] = Xc.dec.share[i]; });
      pmap.setData(v, { valid: null, label: "Coating-like share", unit: "" }); pmap.setTitle("Share of the coating-like component");
    }
  } else if (col === "splat" && Xc && Xc.splat) {
    pmap.setData(Xc.splat.ratioMap, { valid: null, label: `${A.band}/G (splats)`, unit: "" });
    pmap.setTitle(`<u>${escHTML(A.band)}/G</u> from the Gaussian splats`);
  }
  if (!A || !A.C) {
    box.className = "verdict"; box.textContent = ""; t.textContent = ""; $("#seg-ov").replaceChildren();
    $("#seg-coat-cards").replaceChildren(); $("#seg-more-cards").replaceChildren(); $("#seg-more-cards-h").textContent = "";
    $("#seg-coat-note").textContent = A ? A.why : "";
    return;
  }
  const { C, band: b } = A, W = C.within, B = C.between, mu = C.mu, cvs = h => `${pct(h.tauLo / mu)}–${pct(h.tauHi / mu)}`;
  // a standard error to the decimals of the 3-digit mean it belongs to
  const pm = (v, s) => `± ${s.toFixed(Math.max(0, 2 - Math.floor(Math.log10(Math.abs(v) || 1))))}`;
  const grade = h => (h.p >= 0.05 ? 0 : h.I2 < 0.25 ? 1 : h.I2 < 0.75 ? 2 : 3);
  // particle means rest on hundreds of pixels, so any real difference between them gives an I²
  // near 1: they are judged against the spread inside a particle instead
  const gw = grade(W), gb = !B || B.p >= 0.05 ? 0 : B.tau < W.tau ? 1 : 2;
  const head = ["Even within particles", "Nearly even within particles", "Moderately patchy within particles", "Strongly patchy within particles"][gw] + "; " +
    (B ? ["the same on every particle", "particles differ less than places within one", "particles differ more than places within one"][gb] : "too few particles to compare them");
  // the pixels scatter less than their errors predict (Q below its lower 2.5 % point): the
  // interval collapses to 0, and only the plain scatter still limits a real spread
  const low = h => h.p > 0.975;
  const body = W.p >= 0.05
    ? `Inside the particles, ${b}/G scatters from pixel to pixel only as much as each pixel's measurement error allows. ` + (low(W)
      ? `It scatters even less (Q/df ${fmt(W.Q / W.df, 2)}): the errors are on the large side, since misfit of the band model counts as noise, so real patchiness could hide in them. The plain scatter limits it to a CV of at most ${pct(W.sd / mu)}.`
      : `Any real patchiness is below a CV of ${pct(W.tauHi / mu)} (upper 95 % limit).`)
    : `${pct(W.I2)} of the pixel-to-pixel spread of ${b}/G inside the particles is real (I², p ${pFmt(W.p)}); the rest is measurement noise. Within a particle the coating index varies with a CV of ${pct(W.tau / mu)} (95 % CI ${cvs(W)}) around ${fmt(mu, 3)}.`;
  const more = [];
  more.push(!B ? `Only ${C.particles} particle${C.particles === 1 ? "" : "s"} could be analysed: comparing particles needs at least 3.`
    : B.p >= 0.05 ? `Between particles, the means differ only as their errors allow (any real difference below a CV of ${pct(B.tauHi / mu)}).`
    : `Between particles, the means differ with a CV of ${pct(B.tau / mu)} (95 % CI ${cvs(B)}; I² ${pct(B.I2)}, p ${pFmt(B.p)}).`);
  if (B && Number.isFinite(C.shareWithin)) more.push(`Of the real variation, ${pct(C.shareWithin)} lies within particles and ${pct(1 - C.shareWithin)} between them.`);
  const RC = A.rimCmp;
  if (RC) more.push(A.rim
    ? `In the outer ${fmt(A.rim, 3)} µm of the particles, ${b}/G is ${RC.ratio.toFixed(2)} × the value further in and G falls to ${RC.g.toFixed(2)} ×: the laser spot lies partly off the particle there, so these ${fmt(A.rimLeft)} pixels are left out (set the rim to 0 to include them).`
    : `Within 1 µm of the edges, ${b}/G is ${RC.ratio.toFixed(2)} × the particle's mean and G falls to ${RC.g.toFixed(2)} ×. These pixels are included, but the laser spot lies partly off the particle there, so that part of the spread may not be the coating.`);
  more.push(`${b} is detected (height ≥ ${SEG_KD} standard errors) in ${pct(A.nDet / A.n)} of the pixels used.` + (A.nDet / A.n < 0.2 ? " That is few: these numbers rest mostly on noise, and a longer acquisition per pixel would tell more." : ""));
  const nPatchy = A.rows.filter(r => r.h && r.h.p < 0.05).length;
  if (C.particles > 1) more.push(`${nPatchy} of ${C.particles} particles are patchy on their own (p < 0.05 each; ${fmt(0.05 * C.particles, 2)} would be by chance).`);
  // real patches differ more the further apart two pixels are; errors that are too small add the same excess at every distance
  const [p1, ...far] = A.pairs, pFar = far[far.length - 1];
  if (p1 && pFar && W.p < 0.05) {
    if (pFar.ratio > p1.ratio + 0.2) more.push(`The excess is organised in space: pixels ${fmt(p1.um, 2)} µm apart differ ${p1.ratio.toFixed(2)} × what their errors allow, pixels ${fmt(pFar.um, 2)} µm apart ${pFar.ratio.toFixed(2)} ×. These are patches of the coating index on the micrometre scale, not pixel noise.`);
    else if (p1.ratio > 1.3) more.push(`Neighbouring pixels differ by ${p1.ratio.toFixed(2)} × what their errors allow, and pixels ${fmt(pFar.um, 2)} µm apart hardly more (${pFar.ratio.toFixed(2)} ×), as noise would. The errors look too small, so the within-particle numbers overstate the patchiness.`);
  }
  // particles the map cuts: their outline there is the map's edge, not theirs
  const cut = A.rows.filter(r => r.cut > 0.02);
  if (cut.length) {
    const worst = cut.reduce((u, v) => (v.cut > u.cut ? v : u));
    more.push(`${cut.length === A.rows.length ? (cut.length === 1 ? "The particle runs" : "Every particle runs") : `${cut.length} of ${A.rows.length} particles run`} off the edge of the map (${pct(worst.cut)} of particle ${worst.id}'s outline lies on the map's border): what lies beyond is unmeasured, and the cut is not a rim.`);
  }
  // beyond the CV, the median over the particles that have enough pixels
  const U = A.rows.filter(r => r.u).map(r => r.u), mid = k => median(U.map(u => u[k]).filter(Number.isFinite)), midOf = fn => median(U.map(fn).filter(Number.isFinite));
  if (U.length) {
    const nug = mid("nugget"), cold = mid("cold"), coldNull = mid("coldNull"), nLog = U.filter(u => u.logn).length;
    if (Number.isFinite(nug)) more.push(nug >= 0.5 && nug <= 2
      ? `The variogram's nugget, which never uses the fits' noise model, agrees with it (${nug.toFixed(2)} × the predicted noise), an independent check on the noise correction above.`
      : `The variogram's nugget is ${nug.toFixed(2)} × the noise the fits predict, outside a factor of 2: the noise model is off on this map, so the noise-corrected numbers above should not be relied on.`);
    if (Number.isFinite(cold) && Number.isFinite(coldNull)) more.push(cold > coldNull + 0.02
      ? `Gi* marks ${pct(cold)} of the pixels as locally thin; the same field with whole 8 × 8 pixel blocks moved about (which keeps any structure up to that size) gives ${pct(coldNull)}. The excess, ${pct(cold - coldNull)}, is thin regions larger than the blocks.`
      : `Gi* marks ${pct(cold)} of the pixels as locally thin, no more than the same field with whole 8 × 8 pixel blocks moved about (${pct(coldNull)}): any thin places are no larger than the blocks, which this test cannot tell from ordinary correlation.`);
    if (nLog > U.length / 2) more.push(`${b}/G is right-skewed inside the particles (a lognormal fits better than a normal on ${nLog} of ${U.length}), so a uniformity limit belongs on its logarithm or its quantiles, not as a band about the mean.`);
    const nClus = U.filter(u => u.perm.pI < 0.05 && u.perm.pC < 0.05).length;
    if (nClus) more.push(`Neighbouring pixels are more alike than a shuffle of the same values gives on ${nClus} of ${U.length} particle${U.length > 1 ? "s" : ""} (Moran's I and Geary's c, 199 shuffles each, p < 0.05): the spread is organised in space.`);
  }
  box.className = `verdict ${["good", "good", "warn", "bad"][Math.max(gw, gb === 2 ? 2 : 0)]}`;
  box.replaceChildren(el("b", {}, head), el("div", {}, body), ...more.map(s => el("p", {}, s)));
  $("#seg-more-cards-h").textContent = U.length ? `Beyond the CV, inside the particles${U.length > 1 ? ` (median of ${U.length})` : ""}` : "";
  $("#seg-more-cards").replaceChildren(...(U.length ? [
    card("Inequality (Gini)", fmt(mid("gini"), 3), `noise alone gives ${fmt(mid("giniNoise"), 3)} · excess ${fmt(mid("gini") - mid("giniNoise"), 2)}`),
    card("Noise check (nugget)", `${fmt(mid("nugget"), 3)} ×`, "variogram nugget over the fits' noise · 0.5 to 2 agrees"),
    card("Structured spread", pct(mid("structured")), `variogram plateau · range ${fmt(mid("range"), 2)} µm${U.every(u => u.ranged) ? "" : " or more"}`),
    card("Thin places (Gi*)", pct(mid("cold")), `${pct(mid("coldNull"))} with 8 × 8 blocks moved · excess ${pct(Math.max(0, mid("cold") - mid("coldNull")))}`),
    card("Distribution", U.filter(u => u.logn).length > U.length / 2 ? "right-skewed" : "symmetric", `log-scale CV ${pct(mid("lcv"))}`),
    card("Independent pixels", fmt(mid("neff"), 3), `of ${fmt(mid("n"))} · ${fmt(mid("n") / mid("neff"), 2)} × fewer than the pixel count`),
    card("Spread without a mean", `QCD ${fmt(mid("qcd"), 3)}`, `q90/q10 ${fmt(mid("qr"), 3)} · the lowest tenth holds ${pct(mid("low"))} (10 % if even)`),
    card("Shape", `skew ${fmt(mid("skew"), 2)}`, `excess kurtosis ${fmt(mid("kurt"), 2)} · evenness ${fmt(mid("even"), 2)} (32 bins)`),
    card("Clustering", `Moran's I ${fmt(midOf(u => u.perm.I), 2)}`, `p ${pFmt(midOf(u => u.perm.pI))} · Geary's c ${fmt(midOf(u => u.perm.c), 2)}, p ${pFmt(midOf(u => u.perm.pC))} · 199 shuffles`),
    card("Excess CV (8 × 8 blocks)", pct(midOf(u => u.boot && u.boot.est.cvExcess)), "robust CV beyond the pixels' errors · each particle's interval in the table"),
    card("Uniformity U", fmt(midOf(u => u.boot && u.boot.est.U), 3), `(q95 − q5) / (q95 + q5) of ${b}/G · 0 if even`)] : []));

  $("#seg-coat-cards").replaceChildren(
    card(`Mean ${b}/G`, fmt(mu, 3), `${B ? `${pm(mu, B.seMu)} · ` : ""}${C.particles} particle${C.particles === 1 ? "" : "s"} · ${fmt(W.n)} pixels`),
    card("Within-particle CV", pct(W.tau / mu), low(W) ? `at most ${pct(W.sd / mu)} (plain scatter) · errors on the large side` : `95 % CI ${cvs(W)} · noise taken out`),
    card("Real share of the spread inside", pct(W.I2), `I² · p ${pFmt(W.p)}`),
    card("Between-particle CV", B ? pct(B.tau / mu) : "–", B ? `95 % CI ${cvs(B)} · I² ${pct(B.I2)}` : "needs 3 particles"),
    card("Real variation within particles", B ? pct(C.shareWithin) : "–", B ? "the rest between particles" : ""),
    card(`${b} detected`, pct(A.nDet / A.n), `${fmt(A.nDet)} of ${fmt(A.n)} pixels · height ≥ ${SEG_KD} SE`),
    card("Outside the noise band", `${pct(A.above / A.n)} above · ${pct(A.below / A.n)} below`, "each particle's mean · 2.5 % each if even"),
    card("Patchy particles", `${nPatchy} of ${C.particles}`, `p < 0.05 each · ${fmt(0.05 * C.particles, 2)} by chance`),
    card("Pixel pairs vs their errors", p1 ? `${p1.ratio.toFixed(2)} × at ${fmt(p1.um, 2)} µm` : "–",
      far.length ? far.map(q => `${q.ratio.toFixed(2)} × at ${fmt(q.um, 2)} µm`).join(" · ") + " · 1 if no real change" : "1 if the errors are right"),
    card("Rim vs the rest", RC ? `${RC.ratio.toFixed(2)} × ${b}/G` : "–", RC ? `outer ${fmt(RC.width, 3)} µm · G ${RC.g.toFixed(2)} × · ${A.rim ? "left out" : "included"}` : "no rim pixels"));

  t.textContent = "";
  t.append(el("tr", {}, ...["#", "Pixels", `${b}/G`, "CV inside", "I²", `${b} detected`, "Excess CV (blocks)", "U", "Cut"].map(h => el("th", {}, h))));
  const ciTd = (bt, k, f) => (bt ? el("td", {}, f(bt.est[k]), el("div", { class: "sub" }, `${f(bt.ci[k][0])}–${f(bt.ci[k][1])}`)) : el("td", { class: "sub" }, "–"));
  for (const r of A.rows) t.append(el("tr", {},
    el("td", {}, el("b", { style: `color:${PARTICLE_COLOURS[(r.id - 1) % PARTICLE_COLOURS.length]}` }, `● ${r.id}`)),
    el("td", {}, fmt(r.used), el("div", { class: "sub" }, `of ${fmt(r.total)}`)),
    ...(r.h ? [el("td", {}, fmt(r.h.mu, 3), el("div", { class: "sub" }, pm(r.h.mu, r.h.seMu))),
               el("td", {}, pct(r.h.tau / r.h.mu), el("div", { class: "sub" }, low(r.h) ? `≤ ${pct(r.h.sd / r.h.mu)}` : `${pct(r.h.tauLo / r.h.mu)}–${pct(r.h.tauHi / r.h.mu)}`)),
               el("td", {}, pct(r.h.I2), el("div", { class: "sub" }, `p ${pFmt(r.h.p)}`)), el("td", {}, pct(r.det / r.used)),
               ciTd(r.u && r.u.boot, "cvExcess", pct), ciTd(r.u && r.u.boot, "U", v => fmt(v, 3))]
             : [el("td", { colspan: 6, class: "sub" }, "fewer than 3 pixels with a quantified G")]),
    el("td", { class: r.cut > 0.02 ? "" : "sub", title: "Share of the particle's outline on the map's border" }, r.cut > 0.02 ? pct(r.cut) : "–")));

  const spot = A.step < 1 ? `The step here is ${fmt(A.step, 2)} µm and a 532 nm laser spot is about 1 µm across, so one spot covers about ${fmt(Math.PI * (0.5 / A.step) ** 2, 2)} pixels and neighbouring spectra share most of what they probe. ` : `The step here (${fmt(A.step, 2)} µm) is at least the size of a 532 nm laser spot (about 1 µm), so neighbouring spectra hardly overlap. `;
  $("#seg-coat-note").textContent =
    `How: every pixel inside a particle whose G is at least ${SEG_KQ} × its noise (inside its fit bounds and the Constraints) carries ${b}/G, with its standard error from that pixel's own fit (the delta method on the fitted heights). ` +
    `Inside each particle, Cochran's Q tests whether the pixels share one value; I² is the share of the spread beyond the errors, and the CV is the SD of the true values (Paule-Mandel, with its Q-profile 95 % interval) over the mean, pooled over the particles in the cards and per particle in the table. ` +
    `Each particle's mean, with the error its pixels give, is then compared between particles in the same way. The map colours each pixel against its own particle's mean: within, above or below the band its error allows (95 %). ` +
    `Pixel pairs: the mean squared difference of two pixels of one particle over what their errors predict; 1 when the errors are right and nothing changes over that distance. Below 1 means neighbours share part of their error (misfit of the band model), and a rise with distance means real patches. ` +
    `Where the pixels scatter less than their errors predict, the interval collapses to 0 and the plain scatter (noise and real spread together) is quoted as the upper limit (≤ in the table). ` +
    `Rim: pixels closer than the rim width to a particle's edge are left out. The laser spot lies partly off the particle there and G falls; ${b}/G rises at the edges of an uncoated reference map as well, so the rim says little about the coating. ` +
    (A.left ? `${fmt(A.left)} more pixels inside the particles were left out (G too weak or on a bound, or outside the Constraints)${A.small ? `, and ${A.small} particle${A.small > 1 ? "s" : ""} had too few pixels` : ""}. ` : "") +
    `Excess CV (blocks) and U: 500 resamples of whole 8 × 8 pixel blocks of each particle (95 % percentile intervals), so that neighbours sharing their noise or their patch move together; U = (q95 − q5) / (q95 + q5) of ${b}/G, a uniformity that needs no mean. Clustering: Moran's I and Geary's c against 199 shuffles of the same pixels. ` +
    `Limits: neighbouring pixels are not independent samples. ${spot}` +
    `Each spectrum still carries its own noise, so the test of one common value holds. But patches smaller than the spot are averaged away, so the CV describes the coating on the scale of the spot; and the intervals and the particle means' errors count the pixels as more independent than they are, so they are too narrow and the between-particle part is somewhat overstated. ` +
    `${b}/G rises with the amount of amorphous carbon in the probed volume but not in proportion to its thickness, so compare maps measured the same way.`;

  // the overview at the top, and the verdict's notes folded under it
  const lvl = ["good", "good", "warn", "bad"][Math.max(gw, gb === 2 ? 2 : 0)];
  const noise2 = W.noise ** 2, within2 = W.tau ** 2, between2 = B ? B.tau ** 2 : 0, tot = noise2 + within2 + between2;
  const rowsP = A.rows.filter(r => r.h).map(r => ({ id: r.id, est: r.h.tau / r.h.mu, lo: r.h.tauLo / r.h.mu, hi: r.h.tauHi / r.h.mu, patchy: r.h.p < 0.05, n: r.used }));
  // every pixel over its own particle's mean, for the distribution; the noise curve from at most 1 500 of them
  const vals = [], rel = [];
  A.px.groups.forEach((g, k) => { const h = C.parts[k]; if (!h || !(h.mu > 0)) return; for (const p of g) { vals.push(A.px.y[p] / h.mu); rel.push(A.px.se[p] / h.mu); } });
  const relS = rel.filter((_, i) => i % Math.max(1, Math.ceil(rel.length / 1500)) === 0).filter(e => e > 0), wRel = W.tau / mu;
  ovRender($("#seg-ov"), {
    level: lvl, head, sub: body,
    meter: { share: W.p >= 0.05 ? 0 : W.I2, title: "Real share of the spread inside the particles (I²)", label: W.p >= 0.05 ? `${pct(W.I2)} · within the noise` : pct(W.I2) },
    tiles: [
      { k: `Mean ${b}/G`, v: fmt(mu, 3), s: `${C.particles} particle${C.particles === 1 ? "" : "s"} · ${fmt(W.n)} pixels` },
      low(W) ? { k: "CV inside", v: `≤ ${pct(W.sd / mu)}`, s: "within particles · plain scatter" }
             : { k: "CV inside", v: pct(wRel), ci: [wRel, W.tauLo / mu, W.tauHi / mu, Math.max(0.2, 1.3 * W.tauHi / mu)], s: `within particles · 95 % CI ${cvs(W)}` },
      B ? { k: "CV between", v: pct(B.tau / mu), ci: [B.tau / mu, B.tauLo / mu, B.tauHi / mu, Math.max(0.2, 1.3 * B.tauHi / mu)], s: `particle to particle · 95 % CI ${cvs(B)}` }
        : { k: "CV between", v: "–", s: "particle to particle · needs 3 particles" }],
    splitTitle: "Where the spread comes from (share of the variance)",
    split: [{ name: "pixel noise", share: noise2 / tot, colour: T.muted }, { name: "real, within particles", share: within2 / tot, colour: T.viz1 },
            ...(between2 > 0 ? [{ name: "real, between particles", share: between2 / tot, colour: T.viz2 }] : [])],
    plots: [
      ...(rowsP.length ? [{ title: "Each particle: how patchy inside", height: 52 + 26 * rowsP.length, draw: h => drawPartStrip(h, rowsP),
                            legend: [[T.viz2, "patchy (p < 0.05)", "box"], [T.viz1, "even within the noise", "box"]] }] : []),
      { title: `${b}/G of every pixel over its particle's mean`,
        draw: h => drawDistPlot(h, { values: vals, xlabel: `${b}/G over the particle's mean`, unit: "pixels",
          curves: [{ name: "noise", f: x => { let s = 0; for (const e of relS) s += npdfAt(x, 1, e); return s / relS.length; }, colour: T.muted },
                   ...(wRel > 1e-6 ? [{ name: "true", f: x => npdfAt(x, 1, wRel), colour: T.viz1, wash: true }] : [])] }),
        legend: [[T.label, "measured", "box"], [T.muted, "noise alone (an even coating)", "line"], ...(wRel > 1e-6 ? [[T.viz1, "real spread, noise taken out", "line"]] : [])] }],
  });
  foldNotes(box);
}
$("#seg-coat-band").onchange = () => { if (S.seg) renderParticles(); };
$("#seg-colour").onchange = () => { if (S.seg) renderParticles(); };
$("#seg-rim").onchange = () => { if (S.seg) renderParticles(); };

// =========================================================================
// Cross-checks on a map: a decomposition of the spectra, and Gaussian splats
// =========================================================================
// Both run on demand (a decomposition takes a second or two, the splats a few seconds a band) and are kept
// until the map's analysis changes. Neither is used for coverage itself: the decomposition gives a fit-free
// abundance of a coating-like component to compare with it, the splats a smooth ratio field and a spatial
// scale to compare with the within-particle variation.
const XC_LO = 1100, XC_HI = 1700, XC_BIN = 8;
const xcKey = () => (S.segCoat && S.segCoat.C && S.segCoat.seg === S.seg && S.fit ? `${S.segCoat.key}|${S.fit.seconds}` : null);
const xcNow = () => (S.xc && S.xc.key === xcKey() ? S.xc : null);
const xcTick = () => new Promise(res => setTimeout(res, 0));

// the spectra of chosen pixels prepared for decomposition: a straight baseline through 1030-1090 and
// 1700-1740 cm-1 taken off, the band region 1100-1700 cm-1 binned to 8 cm-1, negatives set to 0
function xcSpectra(cube, K, wn, list) {
  const nb = Math.round((XC_HI - XC_LO) / XC_BIN), bin = new Int32Array(K).fill(-1), cnt = new Float64Array(nb), base = [];
  for (let k = 0; k < K; k++) {
    if ((wn[k] >= 1030 && wn[k] <= 1090) || (wn[k] >= 1700 && wn[k] <= 1740)) base.push(k);
    if (wn[k] >= XC_LO && wn[k] < XC_HI) { bin[k] = Math.floor((wn[k] - XC_LO) / XC_BIN); cnt[bin[k]]++; }
  }
  if (base.length < 6 || !(wn[0] <= 1030 && wn[K - 1] >= 1740) || cnt.some(c => !c)) return null;
  const mx = base.reduce((t, k) => t + wn[k], 0) / base.length, sxx = base.reduce((t, k) => t + (wn[k] - mx) ** 2, 0), V = new Float64Array(list.length * nb);
  list.forEach((p, i) => {
    const o = p * K; let my = 0, sxy = 0;
    for (const k of base) my += cube[o + k] / base.length;
    for (const k of base) sxy += (wn[k] - mx) * (cube[o + k] - my);
    const slope = sxy / sxx;
    for (let k = 0; k < K; k++) if (bin[k] >= 0) V[i * nb + bin[k]] += (cube[o + k] - my - slope * (wn[k] - mx)) / cnt[bin[k]];
  });
  for (let q = 0; q < V.length; q++) if (V[q] < 0) V[q] = 0;
  return { V, K: nb, axis: Array.from({ length: nb }, (_, j) => XC_LO + (j + 0.5) * XC_BIN) };
}

async function runDecomposition() {
  const key = xcKey(); if (!key) return;
  const A = S.segCoat, f = S.fit, m = f.m, used = A.px.groups.flat(), y = A.px.y, status = $("#xc-status");
  status.textContent = "Decomposing the spectra…"; await xcTick();
  const sp = xcSpectra(S.cube, S.k, S.wn, used);
  if (!sp) { status.textContent = "The decomposition needs spectra covering 1030 to 1740 cm⁻¹."; return; }
  // components learnt on up to 3 000 pixels spread through the list, starting from the mean spectra of the
  // fifth with the lowest and the fifth with the highest coating ratio, and a flat one for what the
  // straight baseline leaves; then every pixel's amounts with the components held
  const ord = used.map((_, i) => i).sort((p, q) => y[used[p]] - y[used[q]]), fifth = Math.max(1, Math.floor(used.length / 5));
  const meanOf = ids => { const h = new Float64Array(sp.K); for (const i of ids) for (let c = 0; c < sp.K; c++) h[c] += sp.V[i * sp.K + c] / ids.length; return h; };
  const level = sp.V.reduce((t, v) => t + v, 0) / sp.V.length;
  const H0 = [meanOf(ord.slice(0, fifth)), meanOf(ord.slice(-fifth)), new Float64Array(sp.K).fill(level)];
  const stride = Math.max(1, Math.ceil(used.length / 3000)), learn = []; for (let i = 0; i < used.length; i += stride) learn.push(i);
  const VL = new Float64Array(learn.length * sp.K); learn.forEach((i, r) => VL.set(sp.V.subarray(i * sp.K, (i + 1) * sp.K), r * sp.K));
  const { H, k } = nmf(VL, learn.length, sp.K, H0, { iters: 200 });
  const W = nmfProject(sp.V, used.length, sp.K, H, k);
  // the coating-like component: one with a G band (its 1570-1610 peak at least twice its level at the window's
  // ends) and, of those, the most signal in the D3 valley (1480-1530) against that peak
  const at = (lo, hi) => sp.axis.flatMap((v, j) => (v >= lo && v < hi ? [j] : []));
  const val = at(1480, 1530), gw = at(1570, 1610), ends = [0, 1, 2, 3, sp.K - 4, sp.K - 3, sp.K - 2, sp.K - 1];
  const comps = Array.from({ length: k }, (_, a) => {
    const h = j => H[a * sp.K + j], g = Math.max(...gw.map(h)), edge = ends.reduce((t, j) => t + h(j), 0) / ends.length;
    return { g, hasG: g > 2 * edge, valley: val.reduce((t, j) => t + h(j), 0) / val.length / (g || 1) };
  });
  const cand = comps.map((c, a) => [c, a]).filter(([c]) => c.hasG).sort((p, q) => q[0].valley - p[0].valley);
  if (!cand.length) { status.textContent = "No component of the decomposition has a G band: it found no carbon component to compare."; return; }
  const jc = cand[0][1], share = Float64Array.from(used, (_, i) => { let t = 0; for (let a = 0; a < k; a++) t += W[i * k + a]; return t > 0 ? W[i * k + jc] / t : NaN; });
  const dec = { H, k, jc, axis: sp.axis, comps, used, share, rho: spearman(Array.from(share), used.map(p => y[p])) };
  // the coverage cross-check: a threshold from the uncoated reference projected on the same components (its
  // 99th percentile, over its spectra above the signal floor), then each judged pixel called both ways
  const R = S.covR && S.covR.mode === "map" ? S.covR : null, rf = $("#cov-ref").value ? fileOf($("#cov-ref").value) : null;
  if (R && rf) {
    const rs = refSpectra(rf, A.band), rIdx = rs ? rs.idx.filter((_, i) => rs.g[i] >= R.gmin) : [];
    if (rIdx.length >= 100) {
      status.textContent = `Projecting the ${fmt(rIdx.length)} spectra of ${rf.name} above the floor…`; await xcTick();
      const cube = new Float32Array(await api(`/api/ds/${rf.id}/cube`)), rsp = xcSpectra(cube, rf.info.k, f32(rf.info.wn), rIdx);
      if (rsp && rsp.K === sp.K) {
        const Wr = nmfProject(rsp.V, rIdx.length, rsp.K, H, k), rShare = [];
        for (let i = 0; i < rIdx.length; i++) { let t = 0; for (let a = 0; a < k; a++) t += Wr[i * k + a]; if (t > 0) rShare.push(Wr[i * k + jc] / t); }
        const tauD = quantileSorted(Float64Array.from(rShare).sort(), 0.99), G = p => f.intensity[p * m + f.names.findIndex(q => q.toLowerCase() === "g")];
        const pos = new Map(used.map((p, i) => [p, i])), cat = new Int16Array(P()).fill(-1), parts = [];
        let both = 0, rOnly = 0, dOnly = 0, none = 0;
        A.px.groups.forEach((g, gi) => {
          const out = new Set(A.px.left[gi]); let nJ = 0, cR = 0, cD = 0;
          for (const p of g) {
            if (G(p) < R.gmin || out.has(p)) continue;
            const r1 = y[p] > R.F.tau, d1 = share[pos.get(p)] > tauD; nJ++; if (r1) cR++; if (d1) cD++;
            if (r1 && d1) { both++; cat[p] = 0; } else if (r1) { rOnly++; cat[p] = 1; } else if (d1) { dOnly++; cat[p] = 2; } else { none++; cat[p] = 3; }
          }
          if (nJ >= 10) parts.push({ id: A.rows[gi].id, n: nJ, ratio: cR / nJ, dec: cD / nJ });
        });
        const N = both + rOnly + dOnly + none, po = (both + none) / N, pr = (both + rOnly) / N, pd = (both + dOnly) / N, pe = pr * pd + (1 - pr) * (1 - pd);
        dec.ref = { name: rf.name, n: rShare.length, tau: tauD, N, agree: po, kappa: pe < 1 ? (po - pe) / (1 - pe) : NaN, both, rOnly, dOnly, none, parts, cat };
      }
    }
  }
  S.xc = { ...(xcNow() || {}), key, dec };
  status.textContent = "";
  renderParticles();
}

async function runSplats() {
  const key = xcKey(); if (!key) return;
  const A = S.segCoat, f = S.fit, m = f.m, used = A.px.groups.flat(), status = $("#xc-status");
  const jG = f.names.findIndex(q => q.toLowerCase() === "g"), jC = f.names.indexOf(A.band);
  // the particles' pixels in their bounding box, scaled to [0, 1] by its longer side; a fifth held out at random
  let i0 = Infinity, i1 = -1, j0 = Infinity, j1 = -1;
  for (const p of used) { const i = (p / S.nx) | 0, j = p - i * S.nx; i0 = Math.min(i0, i); i1 = Math.max(i1, i); j0 = Math.min(j0, j); j1 = Math.max(j1, j); }
  const w = j1 - j0 + 1, hh = i1 - i0 + 1, side = Math.max(w, hh, 2), Rg = rng(1000);
  const X = new Float64Array(2 * used.length), loc = used.map(p => { const i = (p / S.nx) | 0, j = p - i * S.nx; return (i - i0) * w + (j - j0); });
  used.forEach((p, q) => { const i = (p / S.nx) | 0, j = p - i * S.nx; X[2 * q] = (j - j0) / (side - 1); X[2 * q + 1] = (i - i0) / (side - 1); });
  const hold = used.map(() => Rg() < 0.2), trainAll = used.map((_, q) => q).filter(q => !hold[q]);
  let train = trainAll;
  if (train.length > 4000) { train = [...trainAll]; for (let i = 0; i < 4000; i++) { const j = i + Math.floor(Rg() * (train.length - i)); [train[i], train[j]] = [train[j], train[i]]; } train = train.slice(0, 4000); }
  const heldIdx = used.map((_, q) => q).filter(q => hold[q]), trainMask = new Uint8Array(w * hh);
  for (const q of trainAll) trainMask[loc[q]] = 1;
  const out = {};
  for (const [name, j] of [[A.band, jC], ["G", jG]]) {
    const v = used.map(p => f.intensity[p * m + j]), img = new Float64Array(w * hh); used.forEach((_, q) => { img[loc[q]] = v[q]; });
    const Xt = new Float64Array(2 * train.length), yt = new Float64Array(train.length); train.forEach((q, i) => { Xt[2 * i] = X[2 * q]; Xt[2 * i + 1] = X[2 * q + 1]; yt[i] = v[q]; });
    const { field } = await fitSplats(Xt, yt, { nMax: 64, iters: 600, seed: 1000, sMin: 0.6 / (side - 1), sMax: 0.5,
      onTick: t => { status.textContent = `Fitting Gaussian splats to ${name} (${Math.round(100 * t)} %)…`; } });
    if (xcKey() !== key) { status.textContent = ""; return; }                     // the map changed meanwhile
    const pred = field.predict(X), sm = gaussSmooth(img, trainMask, w, hh, 1), rm = a => Math.sqrt(a.reduce((t, e) => t + e * e, 0) / a.length);
    const sc = splatScale(field, side);
    out[name === "G" ? "G" : "C"] = { pred, n: field.n, rmse: rm(heldIdx.map(q => pred[q] - v[q])), smooth: rm(heldIdx.map(q => sm[loc[q]] - v[q]).filter(Number.isFinite)),
                                    noise: neighbourNoise(img, trainMask, w, hh).sd, radius: sc.radius * A.step, elongation: sc.elongation };
  }
  // the splat ratio field on the particles' pixels, and its robust CV per particle against the excess CV
  const ratioMap = new Float64Array(P()).fill(NaN), pos = new Map(used.map((p, q) => [p, q]));
  used.forEach((p, q) => { if (out.G.pred[q] > 0) ratioMap[p] = out.C.pred[q] / out.G.pred[q]; });
  const parts = A.rows.map((r, gi) => {
    const vals = A.px.groups[gi].map(p => ratioMap[p]).filter(Number.isFinite);
    return vals.length >= 100 ? { id: r.id, cv: robustSD(vals) / median(vals), excess: r.u && r.u.boot ? r.u.boot.est.cvExcess : NaN } : null;
  }).filter(Boolean);
  const splat = { ...out, ratioMap, parts, pixels: used.length, held: heldIdx.length, train: train.length,
                  compression: 2 * used.length / (6 * out.C.n + 1 + 6 * out.G.n + 1) };
  S.xc = { ...(xcNow() || {}), key, splat };
  status.textContent = "";
  renderParticles();
}
$("#xc-nmf").onclick = () => runDecomposition().catch(e => { $("#xc-status").textContent = ""; toast(e.message, "error"); });
$("#xc-splat").onclick = () => runSplats().catch(e => { $("#xc-status").textContent = ""; toast(e.message, "error"); });

const XC_CAT = () => [PARTICLE_COLOURS[3], COAT_COLOURS()[2], PARTICLE_COLOURS[4], T.label];
function renderXc(card) {
  const X = xcNow(), box = $("#xc-verdict"), cards = $("#xc-cards"), note = $("#xc-note"), plot = $("#xc-comp");
  const ready = !!xcKey();
  $("#xc-nmf").disabled = $("#xc-splat").disabled = !ready;
  const items = [], out = [], b = S.segCoat ? S.segCoat.band : "";
  const D = X && X.dec, Sp = X && X.splat;
  if (D) {
    const c = D.comps[D.jc];
    out.push(card("Coating-like component", `${pct(median(Array.from(D.share)))} of the signal`, `median share · ${b} valley ${fmt(c.valley, 2)} × its G peak · ρ ${fmt(D.rho, 2)} with the fitted ${b}/G`));
    items.push(`The spectra split into ${D.k} components without any band fit; the coating-like one (a G band with the most signal in the ${b} valley) carries a median ${pct(median(Array.from(D.share)))} of the signal, and its share ranks the pixels as the fitted ${b}/G does at ρ = ${fmt(D.rho, 2)}.`);
    if (D.ref) {
      const Rf = D.ref;
      out.push(card("Coverage, decomposed", pct((Rf.both + Rf.dOnly) / Rf.N), `against ${pct((Rf.both + Rf.rOnly) / Rf.N)} from the ratio · agree on ${pct(Rf.agree)} of ${fmt(Rf.N)} judged pixels · κ ${fmt(Rf.kappa, 2)}`));
      const thr = Rf.tau > 0 ? `the 99th percentile of its share, ${fmt(Rf.tau, 3)}, over ${fmt(Rf.n)} spectra above the floor` : `the coating-like component is absent from at least 99 % of its ${fmt(Rf.n)} spectra above the floor, so any share above 0 counts`;
      const prev = Math.max(Rf.both + Rf.rOnly, Rf.both + Rf.dOnly) / Rf.N > 0.9 || Math.min(Rf.both + Rf.rOnly, Rf.both + Rf.dOnly) / Rf.N < 0.1;
      items.push(`Against a threshold from ${Rf.name} projected on the same components (${thr}), the decomposition calls ${pct((Rf.both + Rf.dOnly) / Rf.N)} of the judged pixels coated and the ratio ${pct((Rf.both + Rf.rOnly) / Rf.N)}; they agree on ${pct(Rf.agree)} (Cohen's κ ${fmt(Rf.kappa, 2)}${prev ? "; κ runs low when nearly every pixel falls on one side, so the agreement says more here" : ""}). ` +
        Rf.parts.map(q => `Particle ${q.id}: ${pct(q.dec)} decomposed, ${pct(q.ratio)} by the ratio`).join("; ") + ".");
    } else items.push("Choose an uncoated reference under Coverage, and set the floor, to compare the decomposition's coverage with the ratio's.");
  }
  if (Sp) {
    out.push(card(`Splats, ${b}`, `${fmt(Sp.C.rmse, 3)} held out`, `noise ${fmt(Sp.C.noise, 3)} · a 1-pixel smoother ${fmt(Sp.C.smooth, 3)} · ${Sp.C.n} splats, ${fmt(Sp.C.radius, 2)} µm`));
    out.push(card("Splats, G", `${fmt(Sp.G.rmse, 3)} held out`, `noise ${fmt(Sp.G.noise, 3)} · a 1-pixel smoother ${fmt(Sp.G.smooth, 3)} · ${Sp.G.n} splats, ${fmt(Sp.G.radius, 2)} µm`));
    const cvS = median(Sp.parts.map(q => q.cv)), cvE = median(Sp.parts.map(q => q.excess));
    out.push(card("Splat ratio field", `CV ${pct(cvS)}`, `against the excess CV ${pct(cvE)} from the pixels · ${fmt(Sp.compression, 3)} pixels per parameter`));
    const fitsNoise = Sp.C.rmse <= 1.2 * Sp.C.noise;
    items.push(`Gaussian splats fitted to ${fmt(Sp.train)} of the pixels predict the ${fmt(Sp.held)} held out to ${fmt(Sp.C.rmse, 3)} counts in ${b} (pixel noise ${fmt(Sp.C.noise, 3)}, a 1-pixel smoother ${fmt(Sp.C.smooth, 3)})${fitsNoise ? ", about the noise: the field carries no structure the splats miss" : ", above the noise: there is structure finer than the splats"}. ${b} varies on a scale of about ${fmt(Sp.C.radius, 2)} µm (G ${fmt(Sp.G.radius, 2)} µm).`);
    if (Number.isFinite(cvS) && Number.isFinite(cvE)) {
      const k2 = cvE > 0 ? cvS / cvE : Infinity;
      items.push(`The splat ratio field has a CV of ${pct(cvS)} inside the particles, against an excess CV of ${pct(cvE)} from the pixels with their noise taken out (${Number.isFinite(k2) ? `${fmt(k2, 2)} ×` : "the pixels show no excess"}; median over the particles with 100 pixels or more)${k2 >= 2 / 3 && k2 <= 1.5 ? ": two independent routes to the within-particle variation roughly agree" : ": they differ by more than half, so read the within-particle variation with care"}. Coverage is never taken from the splat field: it fills gaps smaller than its splats and cannot rescue weak signal.`);
    }
  }
  box.className = "verdict"; box.replaceChildren(...(items.length ? [el("b", {}, "Cross-checks"), ...items.map(t => el("p", {}, t))] : []));
  cards.replaceChildren(...out);
  plot.classList.toggle("hidden", !D);
  if (D) {
    const ax = pfAxes(plot, [XC_LO, XC_HI], [0, 1.05], { xlabel: "Raman shift (cm⁻¹)", yticks: false });
    if (ax) {
      for (let a = 0; a < D.k; a++) {
        const row = Array.from({ length: D.axis.length }, (_, j) => D.H[a * D.axis.length + j]), mx = Math.max(...row) || 1;
        ax.ctx.beginPath(); row.forEach((v, j) => { const x = ax.X(D.axis[j]), yy = ax.Y(v / mx); if (j) ax.ctx.lineTo(x, yy); else ax.ctx.moveTo(x, yy); });
        ax.ctx.strokeStyle = a === D.jc ? COAT_COLOURS()[2] : PARTICLE_COLOURS[a % PARTICLE_COLOURS.length]; ax.ctx.lineWidth = a === D.jc ? 2.4 : 1.2; ax.ctx.stroke();
      }
      ax.ctx.font = font(10); ax.ctx.fillStyle = T.label; ax.ctx.textAlign = "left"; ax.ctx.textBaseline = "top";
      ax.ctx.fillText(`${D.k} components, each scaled to its peak · thick: coating-like`, ax.L + 4, ax.Tp + 2);
    }
  }
  note.textContent = ready ? "Decomposition: non-negative matrix factorisation (hierarchical alternating least squares, Cichocki and Phan 2009) of the particle pixels' spectra, a straight baseline through 1030–1090 and 1700–1740 cm⁻¹ taken off and 1100–1700 cm⁻¹ binned to 8 cm⁻¹, into 3 components started from the mean spectra of the pixels with the lowest and highest fitted ratio and a flat one; each pixel's share is the coating-like component's part of its signal. The reference is projected on the same components, so the threshold means the same on both. " +
    "Splats: a port of the 2-D Gaussian splatting used for the coating maps (anisotropic Gaussians, Adam on the analytic gradient, amplitudes by least squares, splats added where the residual is largest), fitted to 80 % of the pixels (at most 4 000) and scored on the other 20 %; noise from neighbouring training pixels; the scale is the median splat radius weighted by amplitude and area."
    : "Find the particles and analyse the coating first.";
}

// =========================================================================
// Between maps: whether samples differ, with the particle or the map as the unit
// =========================================================================
// Each map's coating analysis is kept when it is drawn (the latest per open file). Pixels of one map are
// not independent, so samples are compared through what does not pretend they are: the shape of each map's
// distribution against the others (a KS distance, with maps relabelled at random as the null), and the
// particles' excess CVs; the same questions asked of pooled pixels are shown beside them, to see how far
// counting every pixel overstates the evidence.
const MAPSTORE = new Map();
// a first guess at a map's sample: its name before the first underscore, less a replicate number ("-2")
const sampleOf = name => name.split("_")[0].replace(/[-\s]+\d{1,2}$/, "").trim() || name;
function keepMap(A) {
  const used = A.px.groups.flat(), U = A.rows.filter(r => r.u);
  const infl = U.length ? median(U.map(u => u.u.n / u.u.neff)) : 1;
  const inU = U.reduce((t, r) => t + r.u.n, 0), neff = U.reduce((t, r) => t + r.u.neff, 0) + (used.length - inU) / infl;
  const old = MAPSTORE.get(S.ds.id);
  MAPSTORE.set(S.ds.id, { id: S.ds.id, name: S.ds.name, band: A.band, rim: A.rim, group: old ? old.group : sampleOf(S.ds.name),
                          pixels: Float64Array.from(used, p => A.px.y[p]), neff, particles: U.filter(r => r.u.boot).map(r => r.u.boot.est.cvExcess) });
}

function renderBetween(card) {
  const A = S.segCoat && S.segCoat.seg === S.seg ? S.segCoat : null, t = $("#bm-table"), box = $("#bm-verdict"), cards = $("#bm-cards"), note = $("#bm-note");
  if (A && A.C) keepMap(A);
  const open = new Set(F.files.map(f => f.id)), all = [...MAPSTORE.values()].filter(e => open.has(e.id));
  const band = A ? A.band : $("#seg-coat-band").value, rim = A ? A.rim : Math.max(0, +$("#seg-rim").value || 0);
  const maps = all.filter(e => e.band === band && e.rim === rim), other = all.filter(e => !maps.includes(e));
  t.textContent = ""; box.className = "verdict"; box.textContent = ""; cards.replaceChildren();
  if (maps.length) {
    t.append(el("tr", {}, ...["Map", "Sample", "Particles", "Pixels", "Independent", `${band}/G`, "Excess CV"].map(h => el("th", {}, h))));
    for (const e of maps) {
      const inp = el("input", { type: "text", value: e.group, class: "bm-group", title: "Maps with the same label are replicates of one sample" });
      inp.onchange = () => { e.group = inp.value.trim() || sampleOf(e.name); renderParticles(); };
      t.append(el("tr", { class: S.ds && e.id === S.ds.id ? "on" : "" }, el("td", {}, e.name), el("td", {}, inp), el("td", {}, fmt(e.particles.length)),
        el("td", {}, fmt(e.pixels.length)), el("td", {}, fmt(Math.round(e.neff))), el("td", {}, fmt(median(e.pixels), 3)),
        el("td", {}, e.particles.length ? pct(median(e.particles)) : "–")));
    }
  }
  note.textContent = (other.length ? `${other.map(e => e.name).join(", ")} ${other.length > 1 ? "were" : "was"} analysed with another coating band or rim: open ${other.length > 1 ? "them" : "it"} and analyse again to compare. ` : "") +
    "KS distance: the largest gap between two maps' cumulative distributions of the coating index, each divided by its own median, so it compares the shape of the spread and not its level. Relabelling: the sample labels shuffled over the maps 4 999 times; the gap between samples is the mean distance between maps of different samples minus the mean between replicates. " +
    "By pixels, KS p-values count every pixel as an independent value; by independent pixels, at each map's effective number (from its pixels' correlation). Uniformity: the excess CVs of the particles (each particle with 100 pixels or more is one value) by Kruskal–Wallis, beside a Brown–Forsythe test of the pooled pixels' spread.";
  if (maps.length < 2) { if (!maps.length) note.textContent = "Analyse two or more maps (open each, find its particles) to compare them here. " + note.textContent; return; }
  // the distance between every pair of maps, on the index over each map's median
  const norm = maps.map(e => { const m = median(e.pixels); return Float64Array.from(e.pixels, v => v / m); });
  const d = maps.map(() => new Array(maps.length).fill(0)), pairs = [];
  for (let i = 0; i < maps.length; i++) for (let j = i + 1; j < maps.length; j++) {
    const k = ksTest(norm[i], norm[j]); d[i][j] = d[j][i] = k.d;
    pairs.push({ i, j, same: maps[i].group === maps[j].group, d: k.d, pRaw: ksTest(maps[i].pixels, maps[j].pixels).p, pEff: ksTest(maps[i].pixels, maps[j].pixels, maps[i].neff, maps[j].neff).p });
  }
  const labels = maps.map(e => e.group), groups = [...new Set(labels)], reps = pairs.filter(q => q.same);
  const G = groups.length >= 2 && reps.length ? groupGap(d, labels) : null;
  const items = [];
  if (G) {
    cards.append(card("Shape distance (KS)", `${fmt(G.within, 2)} within · ${fmt(G.between, 2)} between`, `replicates against different samples · relabelling p ${pFmt(G.p)}${G.pMin > 0.05 ? ` (no lower than ${fmt(G.pMin, 2)} with ${G.groupings} groupings)` : ""}`));
    items.push(G.p < 0.05
      ? `Maps of different samples differ more in the shape of their coating index (KS ${fmt(G.between, 2)}) than replicate maps of one sample (${fmt(G.within, 2)}); shuffling the sample labels over the maps gives a gap this large in ${pct(G.p)} of ${fmt(4999)} tries.`
      : `Maps of different samples differ in the shape of their coating index by KS ${fmt(G.between, 2)}, replicates by ${fmt(G.within, 2)}; shuffling the labels gives a gap as large in ${pct(G.p)} of tries, so the samples are not told apart.${G.pMin > 0.05 ? ` With ${maps.length} maps there are only ${G.groupings} ways to group them, so no test of this kind could reach p < 0.05: map more replicates.` : ""}`);
  } else items.push(groups.length < 2 ? "Every map carries the same sample label: give the maps of each sample their own label to compare samples." : "No sample has two maps yet: replicate maps of a sample are what the difference between samples is judged against.");
  if (reps.length) {
    const sigRaw = reps.filter(q => q.pRaw < 0.05).length, sigEff = reps.filter(q => q.pEff < 0.05).length;
    cards.append(card("Replicates by pixel", `${sigRaw} of ${reps.length} differ`, `KS p < 0.05 counting every pixel · ${sigEff} of ${reps.length} at the independent pixels`));
    if (sigRaw > sigEff) items.push(`Counting every pixel as independent, ${sigRaw} of ${reps.length} pairs of replicate maps of one sample "differ" (KS p < 0.05); at each map's effective number of independent pixels, ${sigEff}. Pixel-level tests find differences between maps that are only the correlation between neighbouring pixels.`);
  }
  if (groups.length >= 2) {
    const kw = kruskalWallis(groups.map(g => maps.filter(e => e.group === g).flatMap(e => e.particles)));
    const bf = brownForsythe(groups.map(g => maps.filter(e => e.group === g).flatMap(e => Array.from(e.pixels))));
    cards.append(card("Uniformity by sample", kw ? `p ${pFmt(kw.p)}` : "–", `${kw ? `Kruskal–Wallis over ${fmt(kw.n)} particles` : "too few particles"} · pooled pixels: p ${bf ? pFmt(bf.p) : "–"} over ${bf ? fmt(bf.n) : "–"}`));
    if (kw) items.push(`Between samples, the particles' excess CVs ${kw.p < 0.05 ? "differ" : "do not differ"} (Kruskal–Wallis p ${pFmt(kw.p)}, ${fmt(kw.n)} particles)${bf ? `; the pooled pixels' spread gives p ${pFmt(bf.p)} over ${fmt(bf.n)} pixels, a test that treats every pixel as an independent particle` : ""}.`);
  }
  box.className = `verdict ${G && G.p < 0.05 ? "good" : "warn"}`;
  box.replaceChildren(el("b", {}, `${maps.length} maps of ${groups.length} sample${groups.length > 1 ? "s" : ""}`), ...items.map(s => el("p", {}, s)));
}

// =========================================================================
// a reference, or a map for a calibration, read from a file of its own
// =========================================================================
// The uncoated reference (Coverage, D3/G calibration) and the maps of a calibration build can be
// chosen among the files open in the window or read from disk with the last entry of their list,
// "Choose a file…". The file opens beside the current one, which stays on screen, and is fitted
// with the Carbon fit settings (Coverage reads the reference's fit). One uncoated reference serves
// both sections: chosen in one, it is taken in the other too where none is chosen yet.
const SIDE_FILE = "__file";
const REF_PICK = { cov: null, d3g: null, b1: null, b2: null };     // chosen once the file is open (and, for coverage, fitted)
const sideOption = () => el("option", { value: SIDE_FILE }, "Choose a file…");
let sideTarget = null;

async function openSideFile(files) {
  const { groups, tables } = await mapGroups(files);
  if (!groups.length) throw new Error(tables.length ? "a fitted-result table holds no spectra: choose the map itself (.l6m)"
    : "choose a map (.l6m, with its .l6v), a ParticleFinder spectra list with its .l6pf, or a text export of the spectra");
  setStatus(isBrowserOnly() ? `Reading ${groups[0][0].name}…` : `Uploading ${groups[0][0].name}…`, "busy");
  const info = await apiUpload(groups[0]);
  if (info.resultsOnly) throw new Error(`${info.name}: a fitted-result table holds no spectra: choose the map itself`);
  addFile(info);
  return fileOf(info.id);
}
// fitted with the open map's settings, after any fit already running
async function fitSide(f) {
  const cfg = structuredClone(S.cfg);
  if (f.result && f.key === fitKey(cfg, f.info)) return;
  while (S.fitRunning || F.running || F.busy) await sleep(200);
  setStatus(`Fitting ${f.name}…`, "busy");
  await fitOne(f, cfg);
  setStatus(f.state === "error" ? `Could not fit ${f.name}: ${f.error}` : `Fitted ${f.name}`, f.state === "error" ? "err" : "");
}
const rerenderParticles = () => { if (S.points && S.fit) renderPF(); else if (S.seg) renderParticles(); else renderD3g(); };

// what each list does with a file read for it
const SIDE_USE = {
  "#cov-ref": f => { REF_PICK.cov = f.id; if (!$("#d3g-ref").value) REF_PICK.d3g = f.id; },
  "#d3g-ref": f => { REF_PICK.d3g = f.id; if (!$("#cov-ref").value && !Number.isFinite(parseFloat($("#cov-tau").value))) REF_PICK.cov = f.id; },
  "#d3g-b1": f => { REF_PICK.b1 = f.id; },
  "#d3g-b2": f => { REF_PICK.b2 = f.id; },
};
for (const id of Object.keys(SIDE_USE)) {
  const sel = $(id);
  sel.addEventListener("focus", () => { sel.dataset.prev = sel.value; });
  // ahead of the list's own change handler: "Choose a file…" opens the picker instead
  sel.addEventListener("change", e => {
    if (sel.value !== SIDE_FILE) { sel.dataset.prev = sel.value; return; }
    e.stopImmediatePropagation();
    sel.value = sel.dataset.prev && sel.dataset.prev !== SIDE_FILE ? sel.dataset.prev : "";
    sideTarget = id; $("#side-file").click();
  }, { capture: true });
}
$("#side-file").onchange = async e => {
  const files = [...e.target.files], id = sideTarget; e.target.value = ""; sideTarget = null;
  if (!files.length || !id) return;
  try {
    const f = await openSideFile(files);
    SIDE_USE[id](f);
    const both = !!(REF_PICK.cov && REF_PICK.d3g);
    setStatus(`Opened ${f.name} beside ${S.ds ? S.ds.name : "this file"}`);
    rerenderParticles();
    toast(`${f.name}: ${id === "#d3g-b1" || id === "#d3g-b2" ? "chosen for the calibration build" : `the uncoated reference${both ? " for Coverage and the D3/G calibration" : ""}`}`, "good");
    await fitSide(f);                                 // Coverage needs its fit; the calibration needs only its spectra
    rerenderParticles();
  } catch (err) { setStatus("Could not open", "err"); toast(err.message, "error", 9000); }
};

// =========================================================================
// D3/G calibration: band shapes calibrated once and held for every sample
// =========================================================================
// The shapes come from a calibration (d3g-calibration.js): the one of 21 Oct 2026, one built here
// from a graphite and a glassy carbon spectrum and an uncoated and a coated map, or one read from a
// file. Every carbon spectrum's five heights and background are solved with the positions and widths
// held, so the median I(D3)/I(G) compares between samples, and against an uncoated reference fitted
// the same way. The work runs once per file and calibration step, and is kept.
const D3G = { cals: [PRESET_21OCT], pick: "0:2", std: {}, prep: new Map(), res: new Map(), cmp: new Map(), check: new Map(), running: new Set(), building: false };

function d3gFill(sel, opts, want) {
  const ids = opts.map(o => o[0]).join("|"), keep = want ?? sel.value;
  if (sel.dataset.ids !== ids) { sel.replaceChildren(...opts.map(([v, t]) => el("option", { value: v }, t))); sel.dataset.ids = ids; }
  sel.value = opts.some(o => o[0] === keep) ? keep : opts.length ? opts[0][0] : "";
}
function d3gSelects() {
  d3gFill($("#d3g-cal"), D3G.cals.flatMap((c, ci) => c.steps.map((s, si) => [`${ci}:${si}`, `${c.name} · ${s.name}`])), D3G.pick);
  D3G.pick = null;
  const files = F.files.map(f => [f.id, f.name]), cur = S.ds ? S.ds.id : null, others = files.filter(([id]) => id !== cur), side = [SIDE_FILE, "Choose a file…"];
  // a file read for one of these is chosen here once it is open
  const pick = (k, list) => { const id = REF_PICK[k]; if (id && list.some(([v]) => v === id)) { REF_PICK[k] = null; return id; } return undefined; };
  d3gFill($("#d3g-ref"), [["", others.length ? "none: enter its median" : "none open: enter its median"], ...others, side], pick("d3g", others));
  d3gFill($("#d3g-b1"), [["", "an open map…"], ...files, side], pick("b1", files));
  d3gFill($("#d3g-b2"), [["", "an open map…"], ...files, side], pick("b2", files));
  $("#d3g-ref-med").disabled = !!$("#d3g-ref").value;
}
function d3gChoice() {
  const [ci, si] = ($("#d3g-cal").value || "0:2").split(":").map(Number), cal = D3G.cals[ci] || D3G.cals[0];
  const step = cal.steps[si] || cal.steps[cal.steps.length - 1];
  return { cal, step, key: `${D3G.cals.indexOf(cal)}:${cal.steps.indexOf(step)}` };
}
const d3gKey = (f, ch) => `${f.id}|${ch.key}|${$("#pf-incl-excl").checked}`;

// a file's spectra cut, despiked and judged (kept for the last few files: every spectrum in double precision)
async function d3gPrep(f, settings) {
  const key = `${f.id}|${stableJSON(settings)}`;
  if (!D3G.prep.has(key)) {
    const cube = S.ds && f.id === S.ds.id && S.cube ? S.cube : new Float32Array(await api(`/api/ds/${f.id}/cube`));
    const i = f.info, prep = d3gPrepare(cube, i.k, f32(i.wn), i.nx * i.ny, settings), pts = i.points;
    prep.place = pts ? { pf: true, x: f32(pts.x), y: f32(pts.y), cls: pts.cls ? f32(pts.cls) : null, seam: pts.seam ? f32(pts.seam) : null }
      : { pf: false, nx: i.nx, x: f32(i.x), y: f32(i.y) };
    if (D3G.prep.size >= 4) D3G.prep.delete(D3G.prep.keys().next().value);
    D3G.prep.set(key, prep);
  }
  return D3G.prep.get(key);
}
// the spectra a file contributes: carbon ones and, for ParticleFinder data, the particles the rest of
// this tab uses (measured, off the mosaic's seams, the run's filter-excluded ones when asked for)
function d3gInRun(pl, p) { return !pl.pf || ((!pl.cls || pl.cls[p] === 1 || ($("#pf-incl-excl").checked && pl.cls[p] === 0)) && !(pl.seam && pl.seam[p])); }
const d3gList = prep => prep.carbon.filter(p => d3gInRun(prep.place, p));

async function d3gResult(f, ch) {
  const key = d3gKey(f, ch);
  if (D3G.res.has(key)) return D3G.res.get(key);
  const prep = await d3gPrep(f, ch.cal.settings), list = d3gList(prep), pl = prep.place;
  let measured = 0; for (let p = 0; p < prep.P; p++) if (prep.sig[p] > 0 && d3gInRun(pl, p)) measured++;
  let r = { f, key, prep, n: list.length, measured };
  if (list.length) {
    const U = unmix(prep, ch.step.shapes, list), r3 = U.ratio["ID3/IG"], nb = D3G_BANDS.length, j3 = D3G_BANDS.indexOf("D3");
    const blocks = blocksOf(Array.from(r3), Array.from(U.list, p => (pl.pf ? stageBlock(pl.x[p], pl.y[p]) : mapBlock(p, pl.nx))));
    const vals = new Float64Array(prep.P).fill(NaN); U.list.forEach((p, i) => { vals[p] = r3[i]; });
    const fin = Array.from(r3).filter(Number.isFinite);
    r = { ...r, U, vals, blocks, med: blockMedian(blocks), mean: fin.reduce((t, v) => t + v, 0) / fin.length,
          area: median(U.ratio["AD3/AG"]), rn: median(Array.from(U.list, (p, i) => U.rmse[i] / prep.sig[p])),
          z3: median(Array.from(U.list, (_, i) => U.h[i * nb + j3] / U.se[i * nb + j3])), snr: median(Array.from(U.list, p => prep.snr[p])) };
  }
  D3G.res.set(key, r);
  return r;
}
// the result if it is ready; otherwise the work starts and the section redraws when it ends
function d3gGet(f, ch) {
  const key = d3gKey(f, ch);
  if (D3G.res.has(key)) return D3G.res.get(key);
  if (!D3G.running.has(key)) {
    D3G.running.add(key);
    setTimeout(() => d3gResult(f, ch).catch(e => D3G.res.set(key, { f, key, error: e.message }))
      .finally(() => { D3G.running.delete(key); renderD3g(); }), 30);
  }
  return null;
}

// free fits for a build or a check, spread over workers when the page can start them
async function d3gFreeFits(prep, list, init, onProgress) {
  const nb = D3G_BANDS.length, M = list.length, { nu, N, Y, settings } = prep, chunk = 64, jobs = [];
  const out = { list: Int32Array.from(list), h: new Float64Array(M * nb), c: new Float64Array(M * nb), w: new Float64Array(M * nb), r2: new Float64Array(M) };
  for (let a = 0; a < M; a += chunk) jobs.push(a);
  let done = 0;
  const put = (a, r) => { out.h.set(r.h, a * nb); out.c.set(r.c, a * nb); out.w.set(r.w, a * nb); out.r2.set(r.r2, a); done += r.r2.length; if (onProgress) onProgress(done, M); };
  const sub = a => { const ids = list.slice(a, a + chunk), y = new Float64Array(ids.length * N); ids.forEach((p, i) => y.set(Y.subarray(p * N, (p + 1) * N), i * N)); return y; };
  let pool = [];
  try { for (let i = 0, n = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1)); i < n; i++) pool.push(new Worker(new URL("./d3g-worker.js", import.meta.url), { type: "module" })); }
  catch { pool.forEach(w => w.terminate()); pool = []; }
  if (!pool.length) {
    for (const a of jobs) { const y = sub(a); put(a, freeFits({ nu, N, Y: y, settings }, Array.from({ length: y.length / N }, (_, i) => i), init)); await xcTick(); }
    return out;
  }
  try {
    let next = 0;
    await Promise.all(pool.map(w => new Promise((res, rej) => {
      const go = () => {
        if (next >= jobs.length) { res(); return; }
        const a = jobs[next++], y = sub(a);
        w.onmessage = e => { if (e.data.error) rej(new Error(e.data.error)); else { put(a, e.data); go(); } };
        w.onerror = e => rej(new Error(e.message || "a calibration worker failed"));
        w.postMessage({ id: a, nu, N, Y: y, settings, init }, [y.buffer]);
      };
      go();
    })));
  } finally { pool.forEach(w => w.terminate()); }
  return out;
}

// the band shapes of every step of the chosen calibration, the step in use in bold
function d3gTable(ch) {
  const t = $("#d3g-table"), steps = ch.cal.steps, s2 = v => (Number.isFinite(v) ? v.toFixed(1) : "–");
  t.replaceChildren(el("tr", {}, el("th", {}, "Band"), el("th", {}, "Shape"), ...steps.map(s => el("th", { title: "position / FWHM, cm⁻¹" }, s === ch.step ? el("b", {}, s.name) : s.name))));
  for (const b of D3G_BANDS) {
    t.append(el("tr", {}, el("td", {}, b), el("td", { class: "sub" }, ch.cal.settings.bands[b].shape),
      ...steps.map(s => {
        const v = s.shapes[b], med = /^median spectrum/.test(v.note || ""), kept = !med && (/kept|inherit/i.test(v.note || "") || v.support < 0);
        const tip = med ? `measured on the map's ${v.note}${s.d3median ? `: height ${s.d3median.z.toFixed(1)} × the fit's residual, fitted at ${s2(s.d3median.pos)} / ${s2(s.d3median.fwhm)}` : ""}`
          : v.support != null ? `${v.support < 0 ? `measured in ${-v.support} spectra only: ` : v.support > 1 ? `median over ${v.support} spectra` : ""}${kept ? "the step before's shape kept" : v.support > 1 ? "" : v.note || ""}` : v.note || "";
        return el("td", { title: tip }, `${s2(v.pos)} / ${s2(v.fwhm)}`, kept ? el("span", { class: "sub" }, " kept") : med ? el("span", { class: "sub" }, " median spectrum") : "");
      })));
  }
  // the top of the uncoated-to-amorphous scale with each step's shapes
  t.append(el("tr", { title: "The glassy carbon standard's I(D3)/I(G) with each step's shapes held: 1 on the scale from the uncoated reference to amorphous carbon" },
    el("td", { colspan: 2, class: "sub" }, "Glassy carbon I(D3)/I(G)"), ...steps.map(s => el("td", { class: "sub" }, Number.isFinite(d3gTop(s)) ? d3gTop(s).toFixed(4) : "–"))));
}

// glassy carbon's I(D3)/I(G) with a step's shapes: kept with the step, or from the glassy carbon
// spectrum chosen under Build a calibration
function d3gTop(step) {
  if (Number.isFinite(step.glassy)) return step.glassy;
  const gl = D3G.std.glassy; if (!gl) return NaN;
  if (!step.topCache || step.topCache.std !== gl) step.topCache = { std: gl, v: standardRatio(gl, step.shapes).ratio };
  return step.topCache.v;
}

// The open file's calibrated result against its reference, as the section and the report show it:
// { pending } while the heights are being fitted, { missing } with the reason when there is none
function d3gSummary() {
  const ch = d3gChoice(), cur = S.ds && fileOf(S.ds.id);
  if (!cur || !S.cube) return { ch, missing: S.fit && S.fit.resultsOnly ? "The calibration needs the spectra: open the map itself." : "" };
  const refF = $("#d3g-ref").value ? fileOf($("#d3g-ref").value) : null, res = d3gGet(cur, ch), ref = refF ? d3gGet(refF, ch) : null;
  if (!res || (refF && !ref)) return { ch, cur, pending: true };
  if (res.error || (ref && ref.error)) return { ch, cur, missing: `Could not apply the calibration: ${res.error || ref.error}` };
  const pf = res.prep.place.pf, unit = pf ? "particles" : "spectra";
  if (!res.n) return { ch, cur, missing: `No carbon ${unit} here: none has its G at least ${res.prep.settings.snrMin} × its noise.` };
  const M = res.med, top = d3gTop(ch.step);
  let cmp = null, refMed = NaN, refName = "";
  if (ref && ref.n) {
    const k = `${res.key}~${ref.key}~${top}`;
    if (!D3G.cmp.has(k)) D3G.cmp.set(k, compareMedians(res.blocks, ref.blocks, { top }));
    cmp = D3G.cmp.get(k); refMed = ref.med.median; refName = refF.name;
  } else if (!refF) {
    refMed = parseFloat($("#d3g-ref-med").value);
    if (refMed > 0) {
      cmp = { diff: M.median - refMed, lo: M.lo - refMed, hi: M.hi - refMed, ratio: M.median / refMed, hand: true };
      const index = scaleIndex(M.median, refMed, top);
      if (Number.isFinite(index)) Object.assign(cmp, { index, indexLo: scaleIndex(M.lo, refMed, top), indexHi: scaleIndex(M.hi, refMed, top) });
    }
    refName = "the median entered";
  }
  const why = !cmp ? "no uncoated reference" : !Number.isFinite(top) ? "glassy carbon not known for these shapes: choose its spectrum under Build a calibration"
    : `glassy carbon reads ${top.toFixed(4)} with these shapes, not above the reference`;
  // fixed decimals, enough to show the interval's width to two figures
  const dp = Math.min(6, Math.max(2, 1 - Math.floor(Math.log10(M.hi - M.lo || 1e-3))));
  return { ch, cur, refF, res, ref, pf, unit, M, top, cmp, refMed, refName, why, dp };
}

function renderD3g() {
  const box = $("#d3g-verdict"), cards = $("#d3g-cards"), note = $("#d3g-note"), chk = $("#d3g-check-cards");
  d3gSelects();
  const Z = d3gSummary(), ch = Z.ch;
  d3gTable(ch);
  S.d3gScale = null;
  const clear = msg => { box.className = "verdict"; box.textContent = ""; cards.replaceChildren(); chk.replaceChildren(); note.textContent = msg; };
  if (Z.pending) { clear(""); box.replaceChildren(el("b", {}, `Fitting the heights with ${ch.cal.name} · ${ch.step.name}…`)); return; }
  if (!Z.res) return clear(Z.missing || "");
  const { cur, refF, res, ref, pf, unit, M, top, cmp, refMed, refName, why, dp } = Z;
  const fx = v => (Number.isFinite(v) ? `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(dp)}` : "–"), signed = v => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(dp)}`;
  // per spectrum, for the export: where each one sits on the same scale
  S.d3gScale = cmp && Number.isFinite(cmp.index) ? { ref: refMed, top } : null;
  const warn = res.rn > 2, items = [];
  items.push(`Median over the ${fmt(res.n)} carbon ${unit} of ${cur.name} (G at least ${res.prep.settings.snrMin} × its noise), with the band shapes of ${ch.cal.name} · ${ch.step.name}. The interval resamples whole blocks of neighbouring ${pf ? "particles (150 µm squares)" : "pixels (8 × 8)"}, since neighbouring map spectra share their signal.`);
  if (cmp) items.push(`${fmt(cmp.ratio, 3)} × the uncoated reference (${refName}, ${fx(refMed)}): a difference of ${signed(cmp.diff)} (95 % CI ${fx(cmp.lo)} to ${fx(cmp.hi)}${cmp.hand ? "; the reference's own spread is not counted" : ""}). ` +
    (cmp.lo > 0 ? "More D3 than the reference, beyond the spread of the measurement." : cmp.hi < 0 ? "Less D3 than the reference, beyond the spread of the measurement." : "Not told apart from the reference at this spread."));
  else items.push(refF ? `${refF.name} has no carbon spectra to compare with.` : "Choose an uncoated reference open in this window, or enter its median, to compare with it.");
  if (cmp && Number.isFinite(cmp.index)) items.push(`On the scale from the uncoated reference (0) to glassy carbon, the amorphous reference (1, I(D3)/I(G) ${top.toFixed(4)} with the same shapes): ${cmp.index.toFixed(3)} (95 % CI ${cmp.indexLo.toFixed(3)} to ${cmp.indexHi.toFixed(3)}). ` +
    "It places the sample's D3/G between the two references; it is not a mass or thickness fraction, since D3 per unit of amorphous carbon differs between glassy carbon and a coating.");
  if (warn) items.push(`The held shapes leave a residual of ${res.rn.toFixed(2)} × the noise (median; about 1 where they describe the spectra), so the bands here differ from the calibration's. Strong spectra show it most, since the misfit grows with the signal. Check: shapes from this file says how far I(D3)/I(G) moves with this file's own shapes; a G position that moved points at the instrument's calibration (silicon at 520.7 cm⁻¹).`);
  if (ref && ref.n && Math.max(res.snr / ref.snr, ref.snr / res.snr) > 2)
    items.push(`The two were measured at different signal-to-noise (median G ${fmt(res.snr, 3)} × noise here, ${fmt(ref.snr, 3)} × in the reference). Noise raises I(D3)/I(G), since a height cannot go below zero, so the noisier set reads high: compare samples measured alike.`);
  box.className = `verdict${warn ? " warn" : ""}`;
  box.replaceChildren(el("b", {}, `I(D3)/I(G) ${fx(M.median)} (95 % CI ${fx(M.lo)}–${fx(M.hi)})`), el("div", {}, items[0]), ...items.slice(1).map(t => el("p", {}, t)));
  const card = (k, v, s) => el("div", { class: "card" }, el("div", { class: "k" }, k), el("div", { class: "v" }, v), el("div", { class: "s" }, s));
  cards.replaceChildren(
    card("I(D3)/I(G)", fx(M.median), `median · 95 % CI ${fx(M.lo)}–${fx(M.hi)} · mean ${fx(res.mean)}`),
    card(`Carbon ${unit}`, `${fmt(res.n)} of ${fmt(res.measured)}`, `G ≥ ${res.prep.settings.snrMin} × noise · ${fmt(M.blocks)} blocks${res.prep.spikes ? ` · ${fmt(res.prep.spikes)} spikes removed` : ""}`),
    card("Against the reference", cmp ? signed(cmp.diff) : "–", cmp ? `95 % CI ${fx(cmp.lo)} to ${fx(cmp.hi)} · ${fmt(cmp.ratio, 3)} × ${refName}` : "no reference"),
    card("Uncoated 0 → glassy carbon 1", cmp && Number.isFinite(cmp.index) ? cmp.index.toFixed(3) : "–",
      cmp && Number.isFinite(cmp.index) ? `95 % CI ${cmp.indexLo.toFixed(3)} to ${cmp.indexHi.toFixed(3)} · glassy carbon ${top.toFixed(4)}` : why),
    card("A(D3)/A(G)", fx(res.area), "median · areas of the held line shapes"),
    card("D3 in one spectrum", `${fmt(res.z3, 2)} × its error`, "median height over its error: a single spectrum barely carries D3, the median of many does"),
    card("Fit residual", `${res.rn.toFixed(2)} × noise`, "median · about 1 where the shapes fit, above 2 flagged"));
  const C = D3G.check.get(res.key);
  chk.replaceChildren(...(C ? [
    card("With this file's shapes", fx(C.median), `I(D3)/I(G) median · ${C.median >= M.median ? "+" : "−"}${fmt(Math.abs(C.median / M.median - 1) * 100, 2)} % against the calibration's`),
    card("Residual, own shapes", `${C.rn.toFixed(2)} × noise`, `${res.rn.toFixed(2)} × with the calibration's`),
    card("G here", `${C.shapes.G.pos.toFixed(1)} / ${C.shapes.G.fwhm.toFixed(1)}`, `cm⁻¹, position / FWHM · calibration ${ch.step.shapes.G.pos.toFixed(1)} / ${ch.step.shapes.G.fwhm.toFixed(1)}`),
    card("Measured here", D3G_BANDS.filter(b => C.shapes[b].support > 0).join(", ") || "none", `in at least 5 % of the ${unit}; ${D3G_BANDS.filter(b => !(C.shapes[b].support > 0)).join(", ") || "none"} kept`),
  ] : []));
  note.textContent = `How: each spectrum is cut to ${res.prep.settings.range[0]}–${res.prep.settings.range[1]} cm⁻¹ and cleared of cosmic rays (points more than 8 × the noise above a 7-point running median, at most 2 points wide). Its noise is the robust SD of the second difference over √6; it counts as carbon when its smoothed G peak (1540–1630 cm⁻¹, Savitzky-Golay 11 points) stands at least ${res.prep.settings.snrMin} × that above a straight line through the two ends of the range. ` +
    `The five heights (0 to ${res.prep.settings.hmax} counts) and a poly-2 background are then solved by bounded least squares with every position and width held (D3 Gaussian, the others Lorentzian); a height's error is the linear fit's, at its residual. Areas are the line shapes' own (h × w × π/2 for a Lorentzian, h × w × 1.064 for a Gaussian). ` +
    `The spectra are the raw ones: the Carbon fit tab's smoothing and its free shapes do not enter, and its D3/G is not comparable with this one. ` +
    (pf ? "ParticleFinder: the particles the rest of this tab uses (measured, off the mosaic's seams). " : "") +
    "A calibration is built in steps: the standards fitted in the Q-Carbon windows (G from graphite, D1 to D4 from glassy carbon), then each band's median position and width over the uncoated map's carbon spectra where the band is at least 3 × the noise (R² above 0.5), then the same over the coated map. A band found in fewer than 5 % of the spectra keeps the step before's shape. " +
    "D3 is that weak on an uncoated map, so a build can measure it on the uncoated map's median spectrum instead: taken when its height is at least 3 × that fit's residual, a position or width on its window limit keeping glassy carbon's value. The 21 Oct calibration was made without this (D3 is glassy carbon's). " +
    "The scale: (R − R_uncoated) / (R_glassy − R_uncoated), R the median I(D3)/I(G), glassy carbon's from its standard spectrum with the same shapes, the interval from the same block resamples. The check applies the last step to this file, for comparison only.";
  if ($("#d3g-map").checked && res.vals) {
    pmap.setCategories(null);
    pmap.setTitle("D3/G calibration : <u>I(D3)/I(G)</u>");
    pmap.setData(res.vals, { valid: null, label: "I(D3)/I(G)", unit: "" });
  }
}
for (const id of ["#d3g-cal", "#d3g-ref", "#d3g-ref-med"]) $(id).addEventListener("change", () => renderD3g());
$("#d3g-map").addEventListener("change", () => { if (S.points && S.fit) renderPF(); else if (S.seg) renderParticles(); else renderD3g(); });

$("#d3g-check").onclick = async () => {
  const cur = S.ds && fileOf(S.ds.id), status = $("#d3g-status");
  if (!cur || !S.cube || D3G.building) return;
  const ch = d3gChoice();
  try {
    D3G.building = true; status.textContent = "Fitting this file's spectra freely…"; await xcTick();
    const res = await d3gResult(cur, ch);
    if (!res.n) { status.textContent = "No carbon spectra to fit."; return; }
    const prep = res.prep, list = Array.from(res.U.list);
    const Fr = await d3gFreeFits(prep, list, ch.step.shapes, (a, b) => { status.textContent = `Fitting this file's spectra freely: ${fmt(a)} of ${fmt(b)}…`; });
    const own = calibrate(Fr, prep.sig, ch.step.shapes, ch.cal.settings), U2 = unmix(prep, own, list);
    D3G.check.set(res.key, { shapes: own, median: median(U2.ratio["ID3/IG"]), rn: median(list.map((p, i) => U2.rmse[i] / prep.sig[p])) });
    status.textContent = "";
    renderD3g();
  } catch (e) { status.textContent = `The check failed: ${e.message}`; }
  finally { D3G.building = false; }
};

$("#d3g-csv").onclick = () => {
  const cur = S.ds && fileOf(S.ds.id); if (!cur) return;
  const res = D3G.res.get(d3gKey(cur, d3gChoice()));
  if (!res || !res.U) { toast("Nothing to export yet: the calibration is still running, or found no carbon spectra.", "error"); return; }
  const { prep, U } = res, pl = prep.place, nb = D3G_BANDS.length, at = new Map(Array.from(U.list, (p, i) => [p, i]));
  const num = v => (Number.isFinite(v) ? String(+(+v).toPrecision(7)) : ""), sc = S.d3gScale;
  const rows = [[pl.pf ? "Particle" : "Point", "X (um)", "Y (um)", ...(pl.pf ? [] : ["Column", "Row"]), "G signal/noise", "Noise (counts)", "In the result",
    ...D3G_BANDS.flatMap(b => [`${b} height`, `${b} SE`]), "I(D3)/I(G)", "I(D3)/I(G) SE", "A(D3)/A(G)", "I(D1)/I(G)", "I(D4)/I(G)", "Residual RMS", "R2",
    "Scale (uncoated 0, glassy carbon 1)"]];
  for (let p = 0; p < prep.P; p++) {
    if (!Number.isFinite(prep.sig[p])) continue;                       // never measured
    const i = at.get(p), col = p % (pl.nx || 1), row = pl.pf ? 0 : Math.floor(p / pl.nx);
    const fit = i == null ? Array(2 * nb + 8).fill("") : [...D3G_BANDS.flatMap((_, j) => [num(U.h[i * nb + j]), num(U.se[i * nb + j])]),
      ...["ID3/IG", "ID3/IG_se", "AD3/AG", "ID1/IG", "ID4/IG"].map(k => num(U.ratio[k][i])), num(U.rmse[i]), num(U.r2[i]),
      sc ? num(scaleIndex(U.ratio["ID3/IG"][i], sc.ref, sc.top)) : ""];
    rows.push([String(p + 1), num(pl.pf ? pl.x[p] : pl.x[col]), num(pl.pf ? pl.y[p] : pl.y[row]), ...(pl.pf ? [] : [String(col), String(row)]),
      num(prep.snr[p]), num(prep.sig[p]), i == null ? "no" : "yes", ...fit]);
  }
  download(`${cur.name}_D3G.csv`, rows.map(r => r.join(",")).join("\n") + "\n", "text/csv");
};

// the standards for a build: a LabSpec single spectrum or a two-column text spectrum
$("#d3g-gr").onclick = () => { D3G.picking = "graphite"; $("#d3g-std-file").click(); };
$("#d3g-gl").onclick = () => { D3G.picking = "glassy"; $("#d3g-std-file").click(); };
$("#d3g-std-file").onchange = async e => {
  const file = e.target.files[0], which = D3G.picking; e.target.value = "";
  if (!file || !which) return;
  try {
    const { readL6s, readText } = await import("./engine/readers.js");
    let s;
    if (/\.l6s$/i.test(file.name)) s = readL6s(await file.arrayBuffer(), file.name);
    else {
      const m = readText(await file.text(), file.name);
      if (m.nx * m.ny !== 1) throw new Error(`${file.name}: ${fmt(m.nx * m.ny)} spectra; a standard is one spectrum`);
      s = { wn: Float64Array.from(m.wn), y: Float64Array.from(m.cube) };
    }
    D3G.std[which] = { wn: s.wn, y: s.y, name: file.name };
    $(which === "graphite" ? "#d3g-gr-name" : "#d3g-gl-name").textContent = `${file.name} · ${s.wn.length} ch, ${Math.round(s.wn[0])}–${Math.round(s.wn[s.wn.length - 1])} cm⁻¹`;
  } catch (err) { toast(err.message, "error"); }
};

$("#d3g-build").onclick = async () => {
  if (D3G.building) return;
  const status = $("#d3g-status"), f1 = fileOf($("#d3g-b1").value), f2 = fileOf($("#d3g-b2").value), g = D3G.std.graphite, gl = D3G.std.glassy;
  if (!g || !gl) { status.textContent = "Choose the graphite and the glassy carbon spectra first."; return; }
  if (!f1 || !f2) { status.textContent = "Choose the uncoated and the coated map (open them in this window first)."; return; }
  D3G.building = true; $("#d3g-build").disabled = true;
  try {
    const settings = QC_SETTINGS;
    status.textContent = "Fitting the standards…"; await xcTick();
    const s0 = d3gStandards(g, gl, settings);
    const step = async (f, prev, what, d3Median) => {
      status.textContent = `${what}: reading ${f.name}…`; await xcTick();
      const prep = await d3gPrep(f, settings), list = d3gList(prep);
      if (list.length < 20) throw new Error(`${f.name} has only ${list.length} carbon spectra`);
      const fits = await d3gFreeFits(prep, list, prev, (a, b) => { status.textContent = `${what}: fitting ${f.name} freely, ${fmt(a)} of ${fmt(b)} carbon spectra…`; });
      const shapes = calibrate(fits, prep.sig, prev, settings);
      // D3 too weak in single spectra of the uncoated map: measured on its median spectrum instead
      if (!d3Median) return { shapes, median: null };
      status.textContent = `${what}: D3 on the median spectrum of ${f.name}…`; await xcTick();
      return refineOnMedian(shapes, prep, list, prev, "D3");
    };
    const d3Med = $("#d3g-d3med").checked;
    const r1 = await step(f1, s0.shapes, "Baseline 1", d3Med), r2 = await step(f2, r1.shapes, "Baseline 2", false), now = new Date();
    const steps = [{ name: `Standards (${g.name}, ${gl.name})`, shapes: s0.shapes }, { name: `Baseline 1 (${f1.name})`, shapes: r1.shapes, d3median: r1.median },
                   { name: `Baseline 2 (${f2.name})`, shapes: r2.shapes }];
    for (const s of steps) s.glassy = standardRatio(gl, s.shapes).ratio;
    D3G.cals.push({ name: `Built ${stamp(now)}`, date: now.toISOString().slice(0, 10), settings, steps });
    D3G.pick = `${D3G.cals.length - 1}:2`;
    const m = r1.median, d3 = !d3Med ? "D3 left to the spectrum-by-spectrum rule. "
      : !m ? `D3 measured spectrum by spectrum on ${f1.name}. `
      : m.shape ? `D3 from the median spectrum of ${f1.name}: ${m.z.toFixed(1)} × that fit's residual, at ${r1.shapes.D3.pos.toFixed(1)} / ${r1.shapes.D3.fwhm.toFixed(1)} cm⁻¹${m.posLim || m.wLim ? " (a value on its window limit kept glassy carbon's)" : ""}. `
      : `D3 not found on the median spectrum of ${f1.name} (${m.z.toFixed(1)} × that fit's residual, under 3; the fit put it at ${m.pos.toFixed(1)} / ${m.fwhm.toFixed(1)} cm⁻¹): glassy carbon's shape kept. `;
    status.textContent = `Built from ${g.name}, ${gl.name}, ${f1.name} and ${f2.name}. ${d3}Its baseline 2 is now in use; Save calibration keeps it.`;
    renderD3g();
  } catch (e) { status.textContent = `Could not build the calibration: ${e.message}`; }
  finally { D3G.building = false; $("#d3g-build").disabled = false; }
};

$("#d3g-save").onclick = () => { const { cal } = d3gChoice(); download(`${cal.name.replace(/[^\w.-]+/g, "_")}_D3G_calibration.json`, calToJSON(cal), "application/json"); };
$("#d3g-load").onclick = () => $("#d3g-json-file").click();
$("#d3g-json-file").onchange = async e => {
  const file = e.target.files[0]; e.target.value = "";
  if (!file) return;
  try {
    const cal = calFromJSON(await file.text(), file.name.replace(/\.json$/i, "")), use = cal.steps.findIndex(s => /use for all/i.test(s.name));
    D3G.cals.push(cal); D3G.pick = `${D3G.cals.length - 1}:${use >= 0 ? use : cal.steps.length - 1}`;
    toast(`Loaded ${file.name}: ${cal.steps.length} step${cal.steps.length > 1 ? "s" : ""}`, "good");
    renderD3g();
  } catch (err) { toast(err.message, "error"); }
};

// =========================================================================
// Coverage: the share of particles (or of a particle's area) that carries the coating
// =========================================================================
// A spectrum is judged only above a signal floor: the G height at which a coating of the
// thinnest functional index rc is found against the threshold tau with the chosen power,
// P = 1 - Phi[(tau - rc) / sigma_R(g)], sigma_R(g) = sqrt(sigmaD² + rc² sigmaG²) / g, from the
// noise of the band heights. Below it a spectrum cannot tell coated from uncoated, and
// counting it would measure the signal instead of the coating. tau comes out of sample, from
// an uncoated reference measured in the same session: the 99th percentile of its coating
// index, so 1 % of uncoated spectra pass it (the false-detection rate).
const COV_FPR = 0.01;

// an uncoated reference open in this window: the G height and coating index of its quantified
// spectra (G at least 10 x noise and inside its bounds; ParticleFinder particles measured and off
// the seams). Its threshold is the 99th percentile of the index over those at least as bright as
// the floor, since dim spectra scatter more (floorWithReference).
function refSpectra(f, band) {
  const r = f && f.result; if (!r) return null;
  const m = r.m, jG = r.names.findIndex(b => b.toLowerCase() === "g"), jC = r.names.indexOf(band);
  if (jG < 0 || jC < 0) return null;
  const I = f32(r.intensity), X = f32(r.position), Wd = f32(r.width), se = f32(r.std_error);
  const gp = ((r.config && r.config.peaks) || []).find(q => q.name === r.names[jG]);
  const pts = f.info.points, st = pts && pts.cls ? f32(pts.cls) : null, sm = pts && pts.seam ? f32(pts.seam) : null;
  const near = (v, b, span) => Math.abs(v - b) < 1e-3 * span, vals = [];
  for (let p = 0; p < se.length; p++) {
    const g = I[p * m + jG];
    if (!(se[p] > 0) || !(g >= 10 * se[p]) || (st && st[p] !== 1) || (sm && sm[p])) continue;
    if (gp) {
      const x = X[p * m + jG], w = Wd[p * m + jG], sx = gp.pos_max - gp.pos_min, sw = gp.width_max - gp.width_min;
      if (near(x, gp.pos_min, sx) || near(x, gp.pos_max, sx) || near(w, gp.width_min, sw) || near(w, gp.width_max, sw)) continue;
    }
    vals.push([g, I[p * m + jC] / g, p]);
  }
  return vals.length ? { g: vals.map(v => v[0]), r: vals.map(v => v[1]), idx: vals.map(v => v[2]), n: vals.length, name: f.name, median: median(vals.map(v => v[1])) } : null;
}

function covSettings(band) {
  // the reference: any other open file that has been fitted
  const sel = $("#cov-ref"), others = F.files.filter(f => f.result && (!S.ds || f.id !== S.ds.id)), ids = others.map(f => f.id).join("|");
  if (sel.dataset.ids !== ids) {
    const keep = sel.value;
    sel.replaceChildren(el("option", { value: "" }, others.length ? "none: enter τ" : "none open: enter τ"), ...others.map(f => el("option", { value: f.id }, f.name)), sideOption());
    sel.dataset.ids = ids; sel.value = others.some(f => f.id === keep) ? keep : "";
  }
  // a reference read from a file: chosen once its fit is in
  if (REF_PICK.cov && others.some(f => f.id === REF_PICK.cov)) { sel.value = REF_PICK.cov; REF_PICK.cov = null; }
  const ref = sel.value ? refSpectra(fileOf(sel.value), band) : null, tauIn = $("#cov-tau");
  tauIn.disabled = !!ref;
  const rcIn = parseFloat($("#cov-rc").value);
  return { tau: ref ? NaN : parseFloat(tauIn.value), rcIn: Number.isFinite(rcIn) ? rcIn : NaN, ref, noise: $("#cov-noise").value,
           power: Math.min(0.9999, Math.max(0.5, +$("#cov-power").value || 0.95)), spec: Math.min(0.9999, Math.max(0.5, +$("#cov-spec").value || 0.99)) };
}

// the floor, the threshold and the thinnest coating for the spectra judged: with a reference, the
// floor and tau are found together (floorWithReference); entered by hand, tau is used as given. The
// thinnest coating is the specification's value when entered, otherwise a stated default: halfway
// between the uncoated reference's median and this sample's (2 x tau without a reference).
function covFloor(cs, sD, sG, sampleMedian) {
  const rc = Number.isFinite(cs.rcIn) ? cs.rcIn : cs.ref ? 0.5 * (cs.ref.median + sampleMedian) : 2 * cs.tau, rcDefault = !Number.isFinite(cs.rcIn);
  if (!cs.ref) return { rc, rcDefault, tau: cs.tau, gmin: signalFloor(rc, cs.tau, sD, sG, cs.power), nRef: 0 };
  const t = floorWithReference(cs.ref.g, cs.ref.r, rc, sD, sG, cs.power);
  if (t) return { rc, rcDefault, tau: t.tau, gmin: t.g, nRef: t.n };
  // why not: even the reference's brightest 100 spectra leave a threshold that needs a brighter floor than they reach
  const top = cs.ref.g.map((g, i) => [g, cs.ref.r[i]]).sort((a, b) => b[0] - a[0]).slice(0, 100);
  const tauTop = quantileSorted(Float64Array.from(top, v => v[1]).sort(), 0.99);
  return { rc, rcDefault, tau: NaN, gmin: Infinity, nRef: 0, none: true, tauTop, gTop: top.length ? top[top.length - 1][0] : NaN, gNeed: signalFloor(rc, tauTop, sD, sG, cs.power) };
}
function showFloor(F, band) {
  $("#cov-tau").value = Number.isFinite(F.tau) ? +F.tau.toPrecision(4) : "";
  $("#cov-rc").placeholder = `${fmt(F.rc, 3)} (default)`;
  $("#cov-tau-note").textContent = F.nRef ? `${band}/G · 99th percentile of the ${fmt(F.nRef)} reference spectra above the floor` : `${band}/G · uncoated above it: 1 %`;
}

// the noise of the two band heights: their median standard error over the spectra judged
function heightNoise(cov, m, jC, jG, list) {
  const sd = j => median(list.map(p => Math.sqrt(cov[p * m * m + j * m + j])).filter(Number.isFinite));
  return [sd(jC), sd(jG)];
}

// ParticleFinder: particle-level coverage with its exact lower bound, and the count that certifies
function pfCoverage(A, cs, band) {
  const f = S.fit, m = f.m, cov = S.pfCov.cov, jG = f.names.findIndex(b => b.toLowerCase() === "g"), jC = f.names.indexOf(band);
  if (jG < 0 || jC < 0) return null;
  // the particles of the analysis (measured, in the run, off the seams) passing the Constraints; those with G on
  // a fit bound, or the coating band on one or fitted with zero height, cannot be judged and are counted apart
  const pool = [], out = [];
  for (let p = 0; p < A.n; p++) if (Number.isFinite(A.snr[p]) && (!S.valid || S.valid[p])) (A.atBound[p] || A.cOut[p] ? out : pool).push(p);
  if (pool.length < 3) return null;
  const G = p => f.intensity[p * m + jG], R = p => f.intensity[p * m + jC] / G(p), fitNoise = heightNoise(cov, m, jC, jG, pool);
  // the noise of the band heights from each fit or, chosen, the error of a height without a fit (channel
  // noise and baseline variation, the same for both bands)
  const ffs = A.ff ? median(A.ff.se) : NaN, diff = cs.noise === "diff" && Number.isFinite(ffs), [sD, sG] = diff ? [ffs, ffs] : fitNoise;
  const F = covFloor(cs, sD, sG, median(pool.map(R))), { gmin, tau, rc } = F;
  const meas = pool.filter(p => G(p) >= gmin), n = meas.length, coated = meas.filter(p => R(p) > tau), x = coated.length;
  const e = out.filter(p => G(p) >= gmin).length, bc = boundedCoverage(x, n, e);
  const piBar = n ? meas.reduce((t, p) => t + detectionPower(rc, tau, sD, sG, G(p)), 0) / n : NaN;
  const lower = n ? bc.lower : NaN, need = cs.spec * piBar + (1 - cs.spec) * COV_FPR;
  // coverage without a threshold: a mixture over the quantified particles, its uncoated population the
  // reference's own distribution when one is chosen (kept between redraws)
  const C = A.coat, mk = C ? `${A.key}|${cs.ref ? `${cs.ref.name}|${cs.ref.n}` : "free"}` : null;
  if (C && (!S.mixCache || S.mixCache.key !== mk)) S.mixCache = { key: mk, val: cs.ref ? mixtureCoverage(C.R, C.sR, { ref: cs.ref.r }) : C.mixFree };
  const mix = C ? S.mixCache.val : null;
  return { mode: "pf", band, F, pool: pool.length, n, x, e, c: bc.c, lo: bc.lo, hi: bc.hi, lower, gmin, sD, sG, fitNoise, diff, piBar, pass: lower >= need, mix,
           judged: new Set(meas), coated: new Set(coated),
           certified: Math.max(0, Math.min(1, (lower - COV_FPR) / (piBar - COV_FPR))), needed: Number.isFinite(piBar) ? certifyCount(cs.spec, piBar, COV_FPR) : null };
}

// a map: the covered area of each particle above the floor, its lower bound counting the pixels
// as the independent values they are worth (the effective sample size, from the lag-one correlation)
function mapCoverage(A2, cs) {
  const f = S.fit, m = f.m, cov = S.segCov.cov, jG = f.names.findIndex(b => b.toLowerCase() === "g"), jC = f.names.indexOf(A2.band);
  const all = A2.px.groups.flat(); if (!all.length) return null;
  const G = p => f.intensity[p * m + jG], y = A2.px.y, fitNoise = heightNoise(cov, m, jC, jG, all);
  // the noise of the band heights from each fit or, chosen, from neighbouring pixels
  const diff = cs.noise === "diff" && Number.isFinite(A2.nn.sD) && Number.isFinite(A2.nn.sG), [sD, sG] = diff ? [A2.nn.sD, A2.nn.sG] : fitNoise;
  const F = covFloor(cs, sD, sG, median(all.map(p => y[p]))), { gmin, tau } = F, parts = [];
  A2.px.groups.forEach((g, k) => {
    // pixels whose coating band, or G, sits on a fit bound (or the band at zero height) are left out and counted
    const out = new Set(A2.px.left[k]), above = g.filter(p => G(p) >= gmin), meas = above.filter(p => !out.has(p)), n = meas.length;
    if (n < 10) return;
    const e = above.length - n + A2.px.gOut[k].filter(p => G(p) >= gmin).length, N = n + e;
    const x = meas.filter(p => y[p] > tau).length, u = A2.rows[k].u, bc = boundedCoverage(x, n, e);
    // the exact bound on the worst case, counting the pixels as the independent values they are worth
    const neff = u ? Math.max(1, u.neff * N / u.n) : N;
    const lower = binomLower(Math.max(1, Math.round(neff)), Math.round(neff * (1 - bc.lo)));
    // and by resampling 8 x 8 blocks of the judged pixels, the left-out ones counted as uncovered
    let i0 = Infinity, i1 = -1, j0 = Infinity, j1 = -1;
    for (const p of meas) { const i = (p / S.nx) | 0, j = p - i * S.nx; i0 = Math.min(i0, i); i1 = Math.max(i1, i); j0 = Math.min(j0, j); j1 = Math.max(j1, j); }
    const w = j1 - j0 + 1, hh = i1 - i0 + 1, img = new Float64Array(w * hh), ok = new Uint8Array(w * hh);
    for (const p of meas) { const i = (p / S.nx) | 0, j = p - i * S.nx, q = (i - i0) * w + (j - j0); img[q] = y[p]; ok[q] = 1; }
    const boot = blockBootstrap(img, img, ok, w, hh, { tau, coverOnly: true }), bootLo = boot ? boot.ci.cover[0] * n / N : NaN;
    parts.push({ id: A2.rows[k].id, n, e, total: g.length, c: bc.c, lo: bc.lo, hi: bc.hi, neff, lower, bootLo,
                 bound: Number.isFinite(bootLo) ? Math.min(lower, bootLo) : lower });
  });
  const nMeas = parts.reduce((t, q) => t + q.n, 0);
  return { mode: "map", band: A2.band, F, parts, gmin, sD, sG, fitNoise, diff, share: nMeas / all.length, nOut: parts.reduce((t, q) => t + q.e, 0),
           mean: parts.length ? parts.reduce((t, q) => t + q.c, 0) / parts.length : NaN, lowest: parts.length ? Math.min(...parts.map(q => q.bound)) : NaN };
}

function renderCoverage(card) {
  const box = $("#cov-verdict"), cards = $("#cov-cards"), note = $("#cov-note");
  const clear = msg => { box.className = "verdict"; box.textContent = ""; cards.replaceChildren(); note.textContent = msg; $("#cov-mix-h").textContent = ""; $("#cov-mix-cards").replaceChildren(); };
  $("#cov-mix-h").textContent = ""; $("#cov-mix-cards").replaceChildren();
  const band = S.points ? $("#pf-coat-band").value : $("#seg-coat-band").value;
  S.covR = null;
  $("#cov-noise-diff").textContent = S.points ? "a height without a fit" : "neighbouring pixels";
  if (!S.fit || !band || S.fit.resultsOnly) return clear(S.fit && S.fit.resultsOnly ? "Coverage needs the spectra: open the map itself." : "");
  const cs = covSettings(band);
  if (!cs.ref && !Number.isFinite(cs.tau)) return clear(`Choose an uncoated reference measured in the same session (an open file, or Choose a file… to read one from disk), or enter its threshold τ: the ${band}/G that only 1 % of uncoated spectra exceed.`);
  const R = S.points ? (S.pfa && S.pfCov ? pfCoverage(S.pfa, cs, band) : null) : (S.segCoat && S.segCoat.C && S.segCov ? mapCoverage(S.segCoat, cs) : null);
  if (!R) return clear(S.points ? "Too few particles to judge coverage." : "Find the particles first.");
  showFloor(R.F, band);
  if (R.F.none) return clear(Number.isFinite(R.F.gNeed)
    ? `No floor can be set. Finding a coating of ${band}/G ${fmt(R.F.rc, 3)} with ${pct(cs.power)} power at this sample's noise (${band} ${fmt(R.sD, 3)}, G ${fmt(R.sG, 3)} counts) needs G of at least ${fmt(R.F.gNeed, 3)} counts even against the reference's lowest threshold (${fmt(R.F.tauTop, 3)}, over its brightest 100 spectra), but those reach only ${fmt(R.F.gTop, 3)} counts, so the threshold at that brightness is unknown. Measure a brighter reference, enter a thicker thinnest coating, or acquire longer.`
    : `A coating of ${band}/G ${fmt(R.F.rc, 3)} is not above the reference's threshold even over its brightest spectra (${fmt(R.F.tauTop, 3)}): enter a thicker thinnest coating.`);
  if (!(R.F.rc > R.F.tau)) return clear(`The thinnest coating to be found (${fmt(R.F.rc, 3)}) must lie above τ (${fmt(R.F.tau, 3)}).`);
  S.covR = R;
  const b = band, f2 = v => fmt(v, 3), rcText = `${b}/G ${f2(R.F.rc)}${R.F.rcDefault ? " (a default: halfway between the uncoated median and this sample's; enter the specification's value)" : ""}`;
  const src = R.diff ? (R.mode === "pf" ? "the error of a height without a fit" : "from neighbouring pixels") : cs.noise === "diff" ? "from the fits, since the other estimate is not available here" : "from the fits";
  const how = `G at least ${f2(R.gmin)} counts, where a coating of ${rcText} is found against τ ${f2(R.F.tau)} with ${pct(cs.power)} power (noise of the band heights, ${src}: ${b} ${f2(R.sD)}, G ${f2(R.sG)} counts)` +
    (R.F.nRef ? `; τ is the 99th percentile of the ${fmt(R.F.nRef)} reference spectra at least that bright, since dimmer spectra scatter more` : "");
  const handTau = cs.ref ? "" : " τ was entered by hand, so 1 % false detections is assumed.";
  if (R.mode === "pf") {
    const more = [];
    if (R.e) more.push(`${fmt(R.e)} more particle${R.e > 1 ? "s" : ""} above the floor could not be judged (G or ${b} on a fit bound, or ${b} fitted with zero height) and are counted apart: coverage lies between ${pct(R.lo)} (all of them uncoated) and ${pct(R.hi)} (all coated), and the lower bound above is taken on the first case.`);
    if (R.pool > R.n) more.push(`${fmt(R.pool - R.n)} of ${fmt(R.pool)} particles (${pct(1 - R.n / R.pool)}) are below the floor and are not judged: a longer acquisition, or a sharper focus, would bring them in.`);
    more.push(`That certifies at least ${pct(R.certified)} of particles coated (mean detection power ${pct(R.piBar)}, false detections ${pct(COV_FPR)}).${handTau}`);
    more.push(R.needed ? `Certifying ${pct(cs.spec)} takes ${fmt(R.needed)} particles above the floor at this detection power, all of them found coated.` : `At this detection power, certifying ${pct(cs.spec)} would take more than 20 000 particles: raise the power (a longer acquisition, or a higher floor) first.`);
    box.className = `verdict ${R.pass ? "good" : R.lower >= 0.9 ? "warn" : "bad"}`;
    box.replaceChildren(el("b", {}, R.pass ? `Coverage certified: at least ${pct(cs.spec)} of particles coated` : `Coverage not certified at ${pct(cs.spec)}`),
      el("div", {}, R.n ? `${fmt(R.x)} of ${fmt(R.n)} particles bright enough to judge carry the coating above τ (${pct(R.c)}; 95 % lower bound ${pct(R.lower)}). Bright enough: ${how}.` : `No particle is bright enough to judge: ${how}.`),
      ...more.map(t => el("p", {}, t)));
    cards.replaceChildren(
      card("Threshold τ", f2(R.F.tau), cs.ref ? `99th percentile of ${cs.ref.name} above the floor` : "entered"),
      card("Signal floor", `${f2(R.gmin)} counts`, `G for ${pct(cs.power)} power at ${b}/G ${f2(R.F.rc)}${R.F.rcDefault ? " (default)" : ""}`),
      card("Judged", `${fmt(R.n)} of ${fmt(R.pool)}`, `particles above the floor${R.e ? ` · ${fmt(R.e)} more left out (fit on a bound)` : ""}`),
      card("Coated", pct(R.c), `${fmt(R.x)} above τ · 95 % lower bound ${pct(R.lower)}${R.e ? ` · bounds ${pct(R.lo)}–${pct(R.hi)}` : ""}`),
      card("Certified", `≥ ${pct(R.certified)}`, `95 % confidence · power ${pct(R.piBar)}`),
      card(`For ${pct(cs.spec)}`, R.needed ? fmt(R.needed) : "–", "particles needed, all found coated"));
    const X = R.mix;
    if (X) {
      $("#cov-mix-h").textContent = "Without a threshold";
      if (X.mode === "reference") {
        $("#cov-mix-cards").replaceChildren(
          card("Uncoated, by mixture", pct(X.uncoated), `95 % CI ${pct(X.ci[0])}–${pct(X.ci[1])} · against none: p ${pFmt(X.p)}`),
          card("The coated population", fmt(X.coated.m, 3), `${b}/G · true spread ${fmt(X.coated.tau, 3)} · ${fmt(X.n)} particles`));
        box.append(el("p", {}, `Without any threshold, with the quantified particles taken as a mix of ${cs.ref.name}'s own distribution (uncoated) and one coated population, ${pct(X.uncoated)} of them are uncoated (95 % CI ${pct(X.ci[0])}–${pct(X.ci[1])}; against none, p ${pFmt(X.p)}). Above the threshold, ${pct(1 - R.c)} of the judged particles count as uncoated.`));
      } else {
        $("#cov-mix-cards").replaceChildren(X.two
          ? card("A low group, by mixture", pct(X.uncoated), `95 % CI ${pct(X.ci[0])}–${pct(X.ci[1])} · at ${b}/G ${fmt(X.lo.m, 3)} against ${fmt(X.hi.m, 3)} · separation ${fmt(X.D, 1)}`)
          : card("A low group, by mixture", "none", `one population fits${X.bic2 < X.bic1 ? ", the two do not separate" : " better (BIC)"} · separation ${fmt(X.D, 1)}`));
        box.append(el("p", {}, X.two
          ? `Without any threshold, the quantified particles fall into two populations (BIC; separation ${fmt(X.D, 1)}${X.scale === "asinh" ? ", on an asinh scale, which fits these values better" : ""}): a low one at ${b}/G ${fmt(X.lo.m, 3)} holding ${pct(X.uncoated)} of them (95 % CI ${pct(X.ci[0])}–${pct(X.ci[1])}), against ${fmt(X.hi.m, 3)}. Whether the low group is uncoated needs a reference: choose one to tie it to uncoated material.`
          : `Without any threshold, the quantified particles form one population${X.scale === "asinh" ? " (on an asinh scale, which fits these values better)" : ""}: no low group stands apart, so no particles show up as uncoated.`));
      }
    }
  } else {
    const ok = R.parts.length;
    box.className = `verdict ${!ok ? "warn" : R.lowest >= cs.spec ? "good" : R.mean >= cs.spec ? "warn" : "bad"}`;
    box.replaceChildren(el("b", {}, ok ? `Covered area ${pct(R.mean)}${ok > 1 ? " (mean over particles)" : ""}` : "No particle has enough pixels above the floor"),
      el("div", {}, `${pct(R.share)} of the particle pixels are bright enough to judge: ${how}.` + (ok ? ` Among them, ${R.parts.map(q => `particle ${q.id}: ${pct(q.c)}${q.e ? ` (${pct(q.lo)}–${pct(q.hi)} with ${fmt(q.e)} pixels left out)` : ""}, 95 % lower bound ${pct(q.bound)}`).join("; ")}.` : "")),
      el("p", {}, `Each lower bound is the lower of two: an exact bound that counts the pixels as the independent values they are worth (the effective sample size from their correlation), and the 2.5th percentile of the covered share over 500 resamples of whole 8 × 8 pixel blocks.` +
        (R.nOut ? ` ${fmt(R.nOut)} pixels above the floor had ${b} or G on a fit bound, or ${b} at zero height: they are left out and counted as uncovered in the bounds.` : "") + handTau));
    cards.replaceChildren(
      card("Threshold τ", f2(R.F.tau), cs.ref ? `99th percentile of ${cs.ref.name} above the floor` : "entered"),
      card("Signal floor", `${f2(R.gmin)} counts`, `G for ${pct(cs.power)} power at ${b}/G ${f2(R.F.rc)}${R.F.rcDefault ? " (default)" : ""}`),
      card("Judged", pct(R.share), `of the particle pixels, above the floor${R.nOut ? ` · ${fmt(R.nOut)} left out (fit on a bound)` : ""}`),
      card("Covered area", ok ? pct(R.mean) : "–", ok ? `lowest particle's 95 % bound ${pct(R.lowest)}` : ""));
  }
  note.textContent = `How: only spectra above the signal floor are judged; below it a coated and an uncoated spectrum cannot be told apart, so counting them would measure the signal rather than the coating. ` +
    `Coverage is the share above τ, with an exact (Clopper-Pearson) 95 % lower bound. A particle counts as found coated with the detection power when it is coated and with the false-detection rate when it is not, so a coverage p is certified when the lower bound reaches p × power + (1 − p) × false detections. ` +
    `The particles needed are the count from which every larger count certifies, if all of them are found coated. ` +
    (S.points ? `Without a threshold: a mixture over the quantified particles (not only those above the floor), each with its own error; with a reference, the uncoated population is a kernel density of the reference's values (their noise included, so the reference should be measured the same way), and the coated one normal; without, two normal populations, claimed only when the BIC prefers them and they separate by Ashman's D above 2. Every model is fitted on the index and on asinh(index / median error), linear near the noise and logarithmic above it, and the scale whose best model has the lower BIC (its likelihood carried back by the Jacobian) is kept, so a right-skewed coated population is not split into a bulk and a tail. Intervals from 200 resamples of the particles.` : "");
}
for (const id of ["#cov-ref", "#cov-tau", "#cov-rc", "#cov-power", "#cov-spec", "#cov-noise"]) $(id).addEventListener("change", () => { if (S.points && S.fit) renderPF(); else if (S.seg) renderParticles(); });

// =========================================================================
// Checks on the measurement
// =========================================================================
// The noise of the band heights two ways, the spectra coverage leaves out, the coating ratio without a
// band fit and, for a ParticleFinder run, where the measured particles lie, whether the list follows
// position, and what the run's own size filter removed.
function renderChecks(card) {
  const box = $("#chk-verdict"), cards = $("#chk-cards"), note = $("#chk-note");
  const A = S.points ? S.pfa : S.segCoat && S.segCoat.seg === S.seg ? S.segCoat : null, f = S.fit;
  if (!A || !f || (!S.points && !A.C)) { box.className = "verdict"; box.textContent = ""; cards.replaceChildren(); note.textContent = ""; return; }
  const b = S.points ? $("#pf-coat-band").value : A.band, m = f.m, f2 = v => fmt(v, 3);
  const jG = f.names.findIndex(q => q.toLowerCase() === "g"), jC = f.names.indexOf(b);
  const items = [], out = []; let warn = false;
  // the noise: from each fit, and from neighbouring pixels (a map) or without a fit (ParticleFinder)
  const list = S.points ? A.Q : A.px.groups.flat(), cov = S.points ? S.pfCov && S.pfCov.cov : S.segCov && S.segCov.cov;
  const [fD, fG] = cov && jC >= 0 ? heightNoise(cov, m, jC, jG, list) : [NaN, NaN];
  if (!list.length) out.push(card("Band-height noise", "–", `no quantified ${S.points ? "particles" : "pixels"}`));
  else if (S.points) {
    out.push(card("Band-height noise", `${b} ${f2(fD)} · G ${f2(fG)}`, A.ff ? `from the fits · a 7-channel height without a fit: ${f2(median(A.ff.se))} (channel noise ${f2(A.ff.sigma)}, baseline ${f2(A.ff.sBase)})` : "from the fits"));
  } else {
    const rD = A.nn.sD / fD, rG = A.nn.sG / fG;
    out.push(card("Band-height noise", `${b} ${f2(fD)} · G ${f2(fG)}`, `from the fits · neighbouring pixels: ${f2(A.nn.sD)} and ${f2(A.nn.sG)} (${fmt(rD, 2)} × and ${fmt(rG, 2)} ×)`));
    // the coating band's noise sets the error of the ratio; G's enters only times the ratio squared
    if (rD > 2 && (warn = true)) items.push(`Neighbouring pixels differ in ${b} by ${fmt(rD, 2)} × the noise the fits give. Differences of neighbours also carry real change from one pixel to the next, so they err high; a factor this large means change on the scale of a pixel, or fits that understate their errors.`);
    else if (rD < 0.5) items.push(`Neighbouring pixels differ in ${b} by ${fmt(rD, 2)} × the noise the fits give: neighbours share part of their signal (the laser spot is larger than the step), so the fits' errors are the safer noise here.`);
    if (rG > 2) items.push(`G differs between neighbours by ${fmt(rG, 2)} × its fitted noise: real change in G from pixel to pixel (focus, the particle's surface). Its error enters ${b}/G only times the ratio squared, so at these values it hardly moves the ratio's error.`);
  }
  // spectra left out of the coverage
  let nGb = 0, nCb = 0, nCz = 0;
  const cp = f.config && f.config.peaks ? f.config.peaks.find(q => q.name === b) : null;
  if (S.points) {
    for (let p = 0; p < A.n; p++) {
      if (!Number.isFinite(A.snr[p])) continue;
      if (A.atBound[p]) nGb++;
      if (jC >= 0 && f.intensity[p * m + jC] <= 0) nCz++; else if (jC >= 0 && bandOnBound(f, cp, p, jC)) nCb++;
    }
  } else {
    nGb = A.outG;
    for (const p of A.px.left.flat()) { if (f.intensity[p * m + jC] <= 0) nCz++; else nCb++; }
  }
  const unit = S.points ? "particles" : "pixels";
  out.push(card("Left out of coverage", fmt(nGb + nCb + nCz), `${unit}: ${fmt(nGb)} G on a fit bound · ${fmt(nCb)} ${b} on a bound · ${fmt(nCz)} ${b} at zero height`));
  // the ratio without a fit
  const ff = A.ff;
  if (ff) {
    const C2 = ff.coat, D2 = ff.d1, chk = D2 || C2, chkName = D2 ? "D1/G" : `${b}/G`;
    out.push(card("Ratios without a fit", `ρ = ${fmt(chk.rho, 2)}`, `${chkName}, with the fitted one${D2 ? ` · ${b}/G: ρ ${fmt(C2.rho, 2)}, its fit-free height follows D1/G at ρ ${fmt(ff.follows, 2)}` : ""} · ${b} detection agrees for ${pct(ff.agree)}`));
    const spread = chk.adds > 0 ? `The fitted ${chkName} spreads more (robust CV ${pct(chk.cvFit)} against ${pct(chk.cvFree)}): the fit adds ${pct(chk.adds)} of its own, which any uniformity measure on a fitted ratio inherits.` : `The fitted ${chkName} spreads no more than the fit-free one (robust CV ${pct(chk.cvFit)} against ${pct(chk.cvFree)}).`;
    const lowNote = S.points ? "" : " On a map, where the real spread inside a particle is small beside the noise, ρ stays low even for two sound ratios.";
    if (chk.rho < 0.3 && (warn = true)) items.push(`The fitted ${chkName} and the ${chkName} read off the same spectra without a fit hardly rank the ${unit} alike (ρ = ${fmt(chk.rho, 2)}${chk.rho < 0 ? ", the wrong way round: is a ratio inverted?" : ""}). The band model, rather than the spectra, may set the fitted ratios: check the fit before relying on them.${lowNote} ${spread}`);
    else items.push(`The fitted ${chkName} ranks the ${unit} as the ${chkName} read off the same spectra without a fit does (ρ = ${fmt(chk.rho, 2)}${D2 ? "; D1 and G are strong enough to read without a fit, so this checks the band fit" : ""}). ${spread}`);
    if (D2) items.push(ff.follows > C2.rho
      ? `A height of ${b} read without a fit (at ${fmt(ff.cPos, 4)} cm⁻¹) follows D1/G (ρ = ${fmt(ff.follows, 2)}) more than the fitted ${b}/G (ρ = ${fmt(C2.rho, 2)}): it sits on the tails of D1 and G, so only a fit separates ${b}, and the fit-free ${b}/G is no check on the fitted one.`
      : `The fitted ${b}/G ranks the ${unit} as the ${b}/G read without a fit does (ρ = ${fmt(C2.rho, 2)}), and more closely than that height follows D1/G (ρ = ${fmt(ff.follows, 2)}); it reads ${fmt(C2.ratio, 2)} × the fitted ratio, since it also carries the tails of D1 and G.`);
  } else out.push(card("Ratios without a fit", "–", list.length < 10 ? `fewer than 10 quantified ${unit}` : "needs spectra covering 1030 to 1790 cm⁻¹"));
  if (S.points) {
    // the run: accepted spectra, where the measured particles lie, the list order, the size filter
    out.push(card("Spectra accepted", pct(A.Q.length / Math.max(1, A.located)), `${fmt(A.Q.length)} of ${fmt(A.located)} located · G ≥ ${A.kq} × noise${ff ? ` · without a fit ${pct(ff.accepted / Math.max(1, A.located))}` : ""}`));
    const R = A.run;
    if (R) {
      out.push(card("Measured vs selected", `${fmt(R.measured)} of ${fmt(R.located)}`, R.measured < R.located ? `left half: ${pct(R.leftMeas)} against ${pct(R.leftAll)} · KS on X ${fmt(R.ksX.d, 2)} (p ${pFmt(R.ksX.p)}), on Y ${fmt(R.ksY.d, 2)}` : "every selected particle was measured"));
      out.push(card("List order vs position", `ρ ${fmt(R.orderX, 2)} with X`, `${fmt(R.orderY, 2)} with Y · 0 for a random order`));
      if (R.measured < R.located && (R.ksX.p < 0.01 || R.ksY.p < 0.01) && (warn = true))
        items.push(`The run stopped before the end of its list, and the ${fmt(R.measured)} particles it measured do not lie over the field as the ${fmt(R.located)} it selected do (${pct(R.leftMeas)} of them in the left half against ${pct(R.leftAll)}; KS on X ${fmt(R.ksX.d, 2)}, p ${pFmt(R.ksX.p)}; on Y ${fmt(R.ksY.d, 2)}, p ${pFmt(R.ksY.p)}). The results describe part of the field only.`);
      if (Math.max(Math.abs(R.orderX), Math.abs(R.orderY)) > 0.5 && (warn = true))
        items.push(`The particle list runs across the field (ρ ${fmt(R.orderX, 2)} with X, ${fmt(R.orderY, 2)} with Y), so a run that stops early measures one side of it. A randomised order would keep an incomplete run representative.`);
    }
    const Fl = A.filt;
    if (Fl) {
      out.push(card("Run's own filter", `${fmt(Fl.n)} of ${fmt(Fl.of)} removed`, `${fmt(Fl.below)} smaller than the smallest kept (${f2(Fl.kmin)} µm) · ${fmt(Fl.above)} larger than the largest kept (${f2(Fl.kmax)} µm)`));
      if (Fl.above && (warn = true)) items.push(`The run's own filter removed ${fmt(Fl.above)} objects larger than the largest it kept (${f2(Fl.kmax)} µm): an upper size limit that leaves the big particles out of every result. The Size section reads coverage by size class.`);
    }
  }
  if (S.points && A.disp) {
    const Dp = A.disp, limD = parseFloat($("#acc-disp").value) / 100;
    out.push(card("Dispersion", `Clark–Evans ${fmt(Dp.clarkEvans, 2)}`, `1 at random, below 1 clumped · nearest neighbour ${fmt(Dp.nnMedian, 3)} µm (median) · ${pct(Dp.touching)} touch one · ${pct(Dp.merged)} likely merged`));
    const crowd = `${pct(Dp.crowded)} of the ${fmt(Dp.n)} located particles touch a neighbour (their equivalent circles overlap) or are so large, over 3 × the median area, that they are likely touching particles segmented as one; the Clark–Evans ratio is ${fmt(Dp.clarkEvans, 2)} (1 for particles dropped at random, below 1 when they clump)`;
    if (Dp.crowded > limD && (warn = true)) items.push(`The slide is more crowded than the method allows: ${crowd}. A spectrum may then sample two particles and sizes read high; redisperse the powder (a dry disperser, or less of it) and measure again.`);
    else items.push(`Dispersion: ${crowd}.`);
  }
  box.className = `verdict ${warn ? "warn" : "good"}`;
  box.replaceChildren(...(items.length ? [el("b", {}, "What the checks found"), ...items.map(t => el("p", {}, t))] : []));
  cards.replaceChildren(...out);
  note.textContent = (S.points
    ? "Noise without a fit: a straight baseline through 1030–1090 and 1700–1740 cm⁻¹; each height the mean of 7 channels at the band's median fitted position; its error combines channel noise (successive channel differences in 1700–1790 cm⁻¹, robust SD over √2) with baseline variation (the spread of the residual at 1765 cm⁻¹, where no band lies). "
    : "Neighbouring pixels: the robust SD of the differences between horizontally and vertically adjacent pixels used, over √2. ") +
    "Left out: spectra whose G or coating band sits on a fit bound, or whose coating band is fitted with zero height; coverage counts them both ways for its bounds. " +
    "Ratios without a fit use the same baseline and heights, at each band's median fitted position. The check is that the fitted D1/G ranks the spectra as the fit-free one does (Spearman ρ), with the spread the fit adds of its own (robust CV: 1.4826 × MAD over the median); a weak band such as D3 sits on the tails of D1 and G, so its fit-free height is reported, and how far it follows D1/G, but it cannot check its fit." +
    (S.points ? " Measured vs selected: two-sample Kolmogorov–Smirnov tests on the stage positions of the measured particles against all selected ones (status measured or not measured, off the seams). Dispersion: over the located particles, in the field the mosaic images; the Clark–Evans ratio has no edge correction, so it reads a little high." : "");
}

// =========================================================================
// The acquisition record: the settings saved with the spectra, against a method
// =========================================================================
const METHOD_KEY = "carbo.method", LIMITS_KEY = "carbo.limits";
const store = {
  get: k => { try { return JSON.parse(localStorage.getItem(k) || "null"); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage blocked */ } },
};
// settings that change from one acquisition to the next by nature (place, time, temperature) or name it
const VARIES = /^(X|Y|Z) \(|^Full ?time|^Date$|^Acquired$|^AC$|^Detector temperature|^(Project|Sample|Site|Title|Remark|Instrument|Detector|StageXY)$/;

// the acquisition records in a file's settings: each starts at its exposure ("Acq. time") and runs to the
// next; a name repeated inside one (the laser's filter and the camera's) is numbered
function acqBlocks(settings) {
  const blocks = []; let cur = null;
  for (const [k, v] of settings || []) { if (/^Acq\. time/.test(k)) blocks.push(cur = []); if (cur) cur.push([k, v]); }
  return blocks.map(bl => { const seen = {}; return bl.map(([k, v]) => { seen[k] = (seen[k] || 0) + 1; return [seen[k] > 1 ? `${k} (${seen[k]})` : k, v]; }); });
}
const comparable = block => Object.fromEntries(block.filter(([k, v]) => v != null && !VARIES.test(k.replace(/ \(\d+\)$/, "")))
  .map(([k, v]) => [k, typeof v === "string" ? v.trim() : v]));
function settingDiffs(a, b) {
  const out = [];
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
    const u = a[k], v = b[k];
    if (!(typeof u === "number" && typeof v === "number" ? Math.abs(u - v) <= 1e-9 * Math.max(1, Math.abs(u)) : u === v)) out.push({ k, a: u, b: v });
  }
  return out;
}
const stamp = d => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
// the calibration the spectra cite: accepted when it was made the same day as the acquisition, before it
function calibration(block) {
  const get = k => { const r = block.find(([n]) => n === k); return r ? String(r[1] ?? "").trim() : ""; };
  const ac = get("AC"), acq = get("Acquired");
  if (!ac) return { state: "none", text: "no calibration record" };
  const c = /^Calibrated (\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})/.exec(ac);
  if (!c) return { state: "fail", text: `"${ac}"` };
  const cal = new Date(+c[1], c[2] - 1, +c[3], +c[4], +c[5]), a = /^(\d{2})\.(\d{2})\.(\d{4}) (\d{2}):(\d{2})/.exec(acq);
  if (!a) return { state: "none", cal, text: `calibrated ${stamp(cal)}; the acquisition time is not recorded` };
  const at = new Date(+a[3], a[2] - 1, +a[1], +a[4], +a[5]), same = cal.toDateString() === at.toDateString() && cal <= at;
  return { state: same ? "pass" : "fail", cal, at, text: same ? `calibrated ${stamp(cal)}, ${fmt((at - cal) / 3.6e6, 2)} h before the acquisition (${stamp(at)})` : `calibrated ${stamp(cal)}, the acquisition at ${stamp(at)}: not a calibration of the same session` };
}

function renderAcq() {
  const box = $("#acq-verdict"), t = $("#acq-table"), note = $("#acq-note"), blocks = acqBlocks(S.ds && S.ds.settings), method = store.get(METHOD_KEY);
  t.textContent = ""; S.acq = null; $("#acq-save").disabled = !blocks.length;
  if (!blocks.length) {
    box.className = "verdict warn";
    box.replaceChildren(el("b", {}, "No acquisition record"), el("div", {}, S.ds && S.ds.settings && S.ds.settings.length
      ? "The file holds the camera's settings only (an image or a mosaic), not those of a spectral acquisition."
      : "Text exports and results tables carry no settings: open the instrument's own file (.l6m, .l6pf) to check them."));
    note.textContent = "";
    return;
  }
  const cur = comparable(blocks[0]), cal = calibration(blocks[0]);
  const inner = [...new Set(blocks.slice(1).flatMap(bl => settingDiffs(cur, comparable(bl)).map(d => d.k)))];
  const diffs = method ? settingDiffs(method.settings, cur) : null;
  S.acq = { cur, cal, diffs, method };
  const show = v => (v == null ? "not recorded" : String(v));
  const lines = [];
  if (diffs) for (const d of diffs) lines.push(`${d.k}: ${show(d.b)} here, ${show(d.a)} in the method.`);
  lines.push(`Calibration: ${cal.text}.`);
  if (inner.length) lines.push(`The file holds ${blocks.length} acquisition records and they differ in ${inner.join(", ")}.`);
  if (method) lines.push(`Method: the settings of ${method.from}, saved ${stamp(new Date(method.savedAt))}.`);
  const bad = (diffs && diffs.length) || cal.state === "fail";
  box.className = `verdict ${bad ? "bad" : method && cal.state === "pass" ? "good" : "warn"}`;
  box.replaceChildren(el("b", {}, !method ? "No method saved" : diffs.length ? `${diffs.length} setting${diffs.length > 1 ? "s differ" : " differs"} from the method` : "Settings as the method"),
    ...(method ? [] : [el("div", {}, "Save a file's settings as the method to check every later file against them.")]), ...lines.map(s => el("p", {}, s)));
  t.append(el("tr", {}, ...["Setting", "This file", ...(method ? ["Method"] : [])].map(h => el("th", {}, h))));
  const keys = [...new Set([...Object.keys(cur), ...(method ? Object.keys(method.settings) : [])])], differs = new Set((diffs || []).map(d => d.k));
  for (const k of keys) t.append(el("tr", { class: differs.has(k) ? "warn" : "" }, el("td", {}, k), el("td", {}, show(cur[k])), ...(method ? [el("td", { class: differs.has(k) ? "" : "sub" }, show(method.settings[k]))] : [])));
  note.textContent = "Read from the file: the settings LabSpec saved with the spectra. Position, time, detector temperature and labels change by nature and are listed apart from the comparison. " +
    "A calibration is accepted when it is from the same day as the acquisition and before it; the file records only its time stamp, not the calibration spectrum, so the size of the silicon band cannot be checked here. The method is kept in this browser; save it to a file to keep it with the written method.";
}
$("#acq-save").onclick = () => { if (S.acq) { store.set(METHOD_KEY, { settings: S.acq.cur, from: S.ds.name, savedAt: new Date().toISOString() }); renderParticles(); } };
$("#acq-download").onclick = () => download("carbo_method.json", JSON.stringify({ method: store.get(METHOD_KEY), limits: store.get(LIMITS_KEY) }, null, 2), "application/json");
$("#acq-load").onclick = () => $("#acq-file").click();
$("#acq-file").onchange = async e => {
  const file = e.target.files[0]; e.target.value = ""; if (!file) return;
  try {
    const j = JSON.parse(await file.text());
    if (!j || (!j.method && !j.limits) || (j.method && typeof j.method.settings !== "object")) throw new Error("not a method file");
    if (j.method) store.set(METHOD_KEY, j.method);
    if (j.limits) { store.set(LIMITS_KEY, j.limits); showLimits(); }
    renderParticles();
  } catch (err) { toast(`${file.name}: ${err.message}`, "error"); }
};

// =========================================================================
// Acceptance: limits fixed in advance, the decision on the bound of every interval, and how many to measure
// =========================================================================
const LIMIT_IDS = { cvb: "#acc-cvb", cvw: "#acc-cvw", run: "#acc-run", disp: "#acc-disp", lsl: "#acc-lsl" };
function showLimits() {
  const L = store.get(LIMITS_KEY); if (!L) return;
  for (const [k, id] of Object.entries(LIMIT_IDS)) $(id).value = Number.isFinite(L[k]) ? L[k] : "";
  if (Number.isFinite(L.spec)) $("#cov-spec").value = L.spec;
}
showLimits();
for (const id of [...Object.values(LIMIT_IDS), "#cov-spec"]) $(id).addEventListener("change", () => {
  const L = { savedAt: Date.now(), spec: parseFloat($("#cov-spec").value) };
  for (const [k, i] of Object.entries(LIMIT_IDS)) L[k] = parseFloat($(i).value);
  store.set(LIMITS_KEY, L); renderParticles();
});
for (const id of ["#acc-se", "#acc-delta", "#acc-ratio"]) $(id).addEventListener("change", () => renderParticles());

function renderAccept(card) {
  const box = $("#acc-verdict"), t = $("#acc-table"), note = $("#acc-note"), L = store.get(LIMITS_KEY) || {};
  const lim = k => { const v = parseFloat($(LIMIT_IDS[k]).value); return Number.isFinite(v) ? v / 100 : NaN; };
  const spec = Math.min(0.9999, Math.max(0.5, +$("#cov-spec").value || 0.99)), R = S.covR, acq = S.acq;
  const rows = [];      // [what, found, limit, pass | fail | open | untested, why]
  const judge = (v, l, below) => (!Number.isFinite(l) ? "open" : !Number.isFinite(v) ? "open" : (below ? v <= l : v >= l) ? "pass" : "fail");
  rows.push(["Settings as the method", !acq ? "no record" : !acq.method ? "no method saved" : acq.diffs.length ? `${acq.diffs.length} differ` : "as the method", "",
             !acq || !acq.method ? "open" : acq.diffs.length ? "fail" : "pass"]);
  rows.push(["Calibration of the session", !acq ? "no record" : acq.cal.state === "none" ? "not recorded" : acq.cal.state === "pass" ? "same day, before" : "not this session", "",
             !acq || acq.cal.state === "none" ? "open" : acq.cal.state]);
  let sizing = [];
  const se = Math.max(0.001, (+$("#acc-se").value || 5) / 100), delta = Math.max(0.001, (+$("#acc-delta").value || 10) / 100), ratio = Math.max(1.01, +$("#acc-ratio").value || 1.5);
  if (S.points) {
    const A = S.pfa, C = A && A.coat, share = A ? A.Q.length / Math.max(1, A.located) : NaN;
    rows.push(["Run: spectra accepted", pct(share), Number.isFinite(lim("run")) ? `≥ ${pct(lim("run"))}` : "no limit", judge(share, lim("run"))]);
    const crowded = A && A.disp ? A.disp.crowded : NaN;
    rows.push(["Dispersion: crowded particles", Number.isFinite(crowded) ? pct(crowded) : "–", Number.isFinite(lim("disp")) ? `≤ ${pct(lim("disp"))}` : "no limit", judge(crowded, lim("disp"), true)]);
    rows.push(["Particles coated", R && R.mode === "pf" ? (R.n ? `≥ ${pct(R.certified)} certified` : "no particle above the floor") : "needs τ (Coverage)", `≥ ${pct(spec)}`, R && R.mode === "pf" ? (R.pass ? "pass" : "fail") : "open"]);
    const Tn = C && C.thin, lsl = parseFloat($("#acc-lsl").value);
    rows.push(["Thinnest 5 %: coating index", Tn ? `≥ ${fmt(Tn.ci[0], 3)}` : "–", Number.isFinite(lsl) ? `≥ ${fmt(lsl, 3)}` : "no limit", judge(Tn ? Tn.ci[0] : NaN, lsl)]);
    const cvb = C ? C.H.tauHi / C.H.mu : NaN;
    rows.push(["Between particles: CV", C ? `≤ ${pct(cvb)}` : "–", Number.isFinite(lim("cvb")) ? `≤ ${pct(lim("cvb"))}` : "no limit", judge(cvb, lim("cvb"), true)]);
    rows.push(["Within particles: CV", "needs maps", Number.isFinite(lim("cvw")) ? `≤ ${pct(lim("cvw"))}` : "", "untested"]);
    rows.push(["Area covered", "needs maps", "", "untested"]);
    if (C) {
      const cv = C.H.tau / C.H.mu, cvN = C.H.noise / C.H.mean, ns = nestedSize(se, cv, 0, cvN, 1);
      sizing = [card("Particles for the mean", fmt(ns.population), `SE ${pct(se)} · CV between particles ${pct(cv)} · noise ${pct(cvN)}`)];
    }
    if (R && R.mode === "pf") sizing.push(card(`Particles for ${pct(spec)} coated`, R.needed ? fmt(R.needed) : "–", "above the floor, all found coated"));
    if (C) {
      const cvObs = Math.sqrt((C.H.tau / C.H.mu) ** 2 + (C.H.noise / C.H.mean) ** 2);
      sizing.push(card(`To tell a ${pct(delta)} difference`, `${fmt(nForMean(cvObs, delta))} particles`, `per sample, in the mean ${C.band}/G · CV per particle ${pct(cvObs)} · 80 % power, 5 % two-sided`),
                  card(`To tell a ${fmt(ratio, 2)} × spread`, `${fmt(nForSpread(ratio))} particles`, "per sample, in the particle-to-particle CV · 80 % power, 5 % two-sided"));
    }
  } else {
    const A2 = S.segCoat && S.segCoat.seg === S.seg ? S.segCoat : null, C = A2 && A2.C, B = C && C.between;
    const area = R && R.mode === "map" && R.parts.length ? R.lowest : NaN;
    rows.push(["Area covered, each particle", Number.isFinite(area) ? `≥ ${pct(area)} (lowest)` : R ? "–" : "needs τ (Coverage)", `≥ ${pct(spec)}`, judge(area, spec)]);
    const ub = A2 && A2.rows ? A2.rows.filter(r => r.u && r.u.boot).map(r => r.u.boot.ci.cvExcess[1]) : [], worst = ub.length ? Math.max(...ub) : NaN;
    rows.push(["Within particles: excess CV", ub.length ? `≤ ${pct(worst)} (highest)` : "needs 100 pixels a particle", Number.isFinite(lim("cvw")) ? `≤ ${pct(lim("cvw"))}` : "no limit", judge(worst, lim("cvw"), true)]);
    const cvb = B ? B.tauHi / C.mu : NaN;
    rows.push(["Between particles: CV", B ? `≤ ${pct(cvb)}` : "needs 3 particles", Number.isFinite(lim("cvb")) ? `≤ ${pct(lim("cvb"))}` : "no limit", B ? judge(cvb, lim("cvb"), true) : "untested"]);
    rows.push(["Particles coated", "needs a population run", "", "untested"]);
    if (C) {
      const W = C.within, k = median(A2.rows.filter(r => r.u).map(r => r.u.neff)) || W.n / Math.max(1, C.particles);
      const cvB = B ? B.tau / C.mu : 0, ns = nestedSize(se, cvB, W.tau / C.mu, W.noise / C.mu, k);
      sizing = [card("Maps for the mean", fmt(ns.maps), `SE ${pct(se)} · ${fmt(k, 3)} independent pixels a map · never fewer than ${fmt(ns.floor)}${B ? "" : " (between-particle CV unknown: 0 assumed)"}`),
                card("Particles, one spectrum each", fmt(ns.population), `CV between ${pct(cvB)} · within ${pct(W.tau / C.mu)} · noise ${pct(W.noise / C.mu)}`)];
      const cvMap = Math.sqrt(cvB * cvB + ((W.tau / C.mu) ** 2 + (W.noise / C.mu) ** 2) / k);
      const nMaps = nForMean(cvMap, delta);
      sizing.push(card(`To tell a ${pct(delta)} difference`, B ? `${fmt(nMaps)} map${nMaps === 1 ? "" : "s"}` : "–", B ? `per sample, in the mean ${A2.band}/G · CV per map ${pct(cvMap)} · 80 % power, 5 % two-sided` : "needs the between-particle CV: 3 or more particles"),
                  card(`To tell a ${fmt(ratio, 2)} × spread`, `${fmt(nForSpread(ratio))} particles`, "per sample mapped, in the within-particle CV · 80 % power, 5 % two-sided"));
    }
  }
  rows.push(["Thickness", "not measured", "", "untested", "needs calibration films of known thickness"]);
  const fails = rows.filter(r => r[3] === "fail"), passes = rows.filter(r => r[3] === "pass"), open = rows.filter(r => r[3] === "open"), untested = rows.filter(r => r[3] === "untested");
  const late = L.savedAt && S.openedAt && L.savedAt > S.openedAt;
  const head = fails.length ? `Rejected on ${fails.map(r => r[0]).join("; ")}` : !passes.length ? "No decision yet: set the limits" : open.length ? "Passes the limits set; others are open" : "Passes every limit this file can test";
  box.className = `verdict ${fails.length ? "bad" : passes.length && !open.length ? "good" : "warn"}`;
  box.replaceChildren(el("b", {}, head),
    el("div", {}, `Each criterion is judged on the bound of its 95 % interval, not on the estimate. Not tested by this file: ${untested.map(r => r[0]).join("; ")}.`),
    ...(open.length ? [el("p", {}, `Open (no limit, no method, or nothing to judge yet): ${open.map(r => r[0]).join("; ")}.`)] : []),
    ...(late ? [el("p", {}, `The limits were changed after this file was opened (${stamp(new Date(L.savedAt))}): a decision holds only on limits fixed before the data were seen.`)]
      : L.savedAt ? [el("p", {}, `Limits saved ${stamp(new Date(L.savedAt))}, before this file was opened.`)] : []));
  t.textContent = "";
  t.append(el("tr", {}, ...["Criterion", "Found", "Limit", ""].map(h => el("th", {}, h))));
  const mark = { pass: "✓ pass", fail: "✗ fail", open: "open", untested: "not tested" };
  for (const r of rows) t.append(el("tr", { class: r[3] === "fail" ? "warn" : "" }, el("td", {}, r[0]), el("td", {}, r[1], r[4] ? el("div", { class: "sub" }, r[4]) : ""), el("td", { class: "sub" }, r[2]), el("td", { class: r[3] === "pass" || r[3] === "fail" ? "" : "sub" }, mark[r[3]])));
  $("#acc-size-cards").replaceChildren(...sizing);
  note.textContent = "Limits are kept in this browser and saved with the method file (Acquisition record). " +
    (S.points ? "Between particles: the upper 95 % limit of the particle-to-particle CV (Paule-Mandel); one spot per particle, so it includes variation within particles at the scale of the spot. Spectra accepted: quantified particles over the particles located in the run. "
      : "Within particles: the highest particle's upper 95 % limit of the excess CV (robust CV beyond the pixels' own errors) from 500 resamples of 8 × 8 pixel blocks. Area covered: the lowest particle's lower bound. ") +
    "How many to measure: SE² = CV_b² / n + (CV_w² + CV_n²) / (n k), with k independent pixels a map (1 for one spectrum a particle); no number of pixels removes the between-particle term. To tell two samples apart: 2 ((z(0.975) + z(0.8)) CV / difference)² per sample for the mean; for a ratio of spreads, 1 + ((z(0.975) + z(0.8)) / ln ratio)² from the error of a log variance ratio (on the measured spread, noise included, so it is a little optimistic for the true spread).";
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
  renderD3g();
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
  if ($("#pf-scatter").value === "density") {
    // hexagons coloured by how many particles they hold (log scale): with thousands of particles,
    // one dot each piles up into a block, and a density shows where most of them are
    const px = [], py = [];
    idx.forEach((p, i) => { if (ys[i] >= ylo && ys[i] <= yhi && xs[i] >= xlo && xs[i] <= xhi) { px.push(PX(xs[i])); py.push(PY(ys[i])); } });
    const sz = Math.max(4, Math.min(9, 0.9 * Math.sqrt((W - L - Rr) * (H - Tt - B) / Math.max(1, px.length)))), cells = hexBins(px, py, sz);
    // sparse hexagons in the colour nearer the background, dense ones in the contrasting end, so a
    // lone outlier does not outweigh the crowd (light to dark on a light page, the other way on a dark one)
    const mx = cells.reduce((t, c) => Math.max(t, c.n), 1), V = lut("Viridis"), light = currentTheme() === "light";
    ctx.save(); ctx.beginPath(); ctx.rect(L, Tt, W - L - Rr, H - Tt - B); ctx.clip();
    for (const c of cells) {
      const u = 0.12 + 0.88 * Math.log1p(c.n) / Math.log1p(mx), j = Math.round((light ? 1 - u : u) * 255) * 3;
      ctx.fillStyle = `rgb(${V[j]},${V[j + 1]},${V[j + 2]})`; ctx.beginPath();
      for (let k = 0; k < 6; k++) { const a = Math.PI / 6 + k * Math.PI / 3, hx = c.x + sz * Math.cos(a), hy = c.y + sz * Math.sin(a); if (k) ctx.lineTo(hx, hy); else ctx.moveTo(hx, hy); }
      ctx.fill();
    }
    ctx.restore();
    ctx.font = font(10); ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillStyle = T.label;
    ctx.fillText(`hexagons: 1 to ${fmt(mx)} particles (log scale) · line: binned medians`, L + 6, Tt + 2);
  } else {
    ctx.globalAlpha = Math.max(0.15, Math.min(0.8, 300 / idx.length));
    idx.forEach((p, i) => {
      if (ys[i] < ylo || ys[i] > yhi || xs[i] < xlo || xs[i] > xhi) return;
      ctx.fillStyle = byCoat ? (A.coat.cat[p] < 0 ? T.hatchLine : CC[A.coat.cat[p]])
        : byPop ? (A.label[p] < 0 ? T.hatchLine : A.pops[A.label[p]].nd ? T.label : PARTICLE_COLOURS[A.label[p] % PARTICLE_COLOURS.length]) : cols[Math.max(0, cls[p]) % cols.length];
      ctx.fillRect(PX(xs[i]) - 1.2, PY(ys[i]) - 1.2, 2.4, 2.4);
    });
    ctx.globalAlpha = 1;
  }
  // binned medians: the trend without assuming a shape
  const nb = 10, bins = Array.from({ length: nb }, () => []);
  xs.forEach((v, i) => { const b = Math.floor((v - xlo) / ((xhi - xlo) || 1) * nb); if (b >= 0 && b < nb) bins[b].push(ys[i]); });
  ctx.beginPath();
  let started = false;
  bins.forEach((b, k) => {
    if (b.length < 8) return;
    b.sort((u, v) => u - v);
    const x = PX(xlo + (k + 0.5) / nb * (xhi - xlo)), y = PY(quantile(b, 0.5));
    if (started) ctx.lineTo(x, y); else { ctx.moveTo(x, y); started = true; }
  });
  ctx.strokeStyle = T.paper; ctx.lineWidth = 4.5; ctx.stroke();      // a halo, so the line shows over a dense core
  ctx.strokeStyle = T.strong; ctx.lineWidth = 2; ctx.stroke();
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
  // ParticleFinder's status of each particle (1 measured, 2 selected but not measured, 0 excluded by
  // the run's own filter) and objects on the seams of a stitched mosaic: only measured particles off
  // the seams count, as in LabSpec's results, unless the filter-excluded ones are asked for
  const status = pts.cls, seam = pts.seam, inclExcl = $("#pf-incl-excl").checked;
  const inRun = p => (!status || status[p] === 1 || (inclExcl && status[p] === 0)) && !(seam && seam[p]);
  const key = [S.ds.id, f.seconds, f.names.join(), stableJSON(VIEW_KEYS.map(k => c[k])), kd, kq, mode, cuts.join(), lambda, popBand, coatBand, inclExcl].join("|");
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
  const gate = { notMeasured: 0, excluded: 0, seams: 0, measured: 0 };
  for (let p = 0; p < n; p++) {
    if (status) { if (status[p] === 2) gate.notMeasured++; else if (status[p] === 0) gate.excluded++; else gate.measured++; }
    if (seam && seam[p]) gate.seams++;
  }
  for (let p = 0; p < n; p++) {
    if (!(se[p] > 0) || !inRun(p)) continue;
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

  // size: over the particles measured, off the seams, as the run's filter left them (and, to compare,
  // over every object in the table, as the routine's own histogram counts them)
  const inSize = []; for (let p = 0; p < n; p++) if (inRun(p)) inSize.push(p);
  const ecd = inSize.map(p => pts.ecd[p]), vw = inSize.map(p => (vol[p] > 0 ? vol[p] : 0)), es = Float64Array.from(ecd).filter(Number.isFinite).sort();
  const eAll = Float64Array.from(pts.ecd).filter(Number.isFinite).sort();
  const size = { num: [0.1, 0.5, 0.9].map(q => quantileSorted(es, q)), vol: [0.1, 0.5, 0.9].map(q => weightedQuantile(ecd, vw, q)),
                 all: [0.1, 0.5, 0.9].map(q => quantileSorted(eAll, q)), n: es.length, nAll: eAll.length };
  // the run's own filter: how many objects it removed below the smallest it kept and above the largest
  let filt = null;
  if (status && gate.excluded) {
    let kmin = Infinity, kmax = -Infinity; const dropped = [];
    for (let p = 0; p < n; p++) { if (seam && seam[p]) continue; if (status[p] === 0) dropped.push(pts.ecd[p]); else { kmin = Math.min(kmin, pts.ecd[p]); kmax = Math.max(kmax, pts.ecd[p]); } }
    filt = { n: dropped.length, below: dropped.filter(d => d < kmin).length, above: dropped.filter(d => d > kmax).length, kmin, kmax, of: n - gate.seams };
  }
  // fixed size classes: located, measured and quantified, so that coverage can be read by size
  const EDGES = [0, 1, 3, 10, 25, Infinity];
  const sizeBins = EDGES.slice(0, -1).map((lo, i) => {
    const hi = EDGES[i + 1], b = { lo, hi, located: 0, measured: 0, quantified: 0 };
    for (let p = 0; p < n; p++) {
      if (!(pts.ecd[p] >= lo && pts.ecd[p] < hi) || (seam && seam[p])) continue;
      if (!status || status[p] !== 0 || inclExcl) b.located++;
      if (inRun(p) && se[p] > 0) b.measured++;
      if (quant[p]) b.quantified++;
    }
    return b;
  });
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
      // the thinnest particles, each particle's error taken out (on the log scale when right-skewed)
      const thin = thinTail(use, sR, { log: !!lognormalPreferred(use) });
      // the coating index against size, convexity and elongation together, each scaled to its SD
      const mrow = Q.map((_, i) => i).filter(i => pts.ecd[Q[i]] > 0 && Number.isFinite(pts.convexity[Q[i]]) && Number.isFinite(pts.elongation[Q[i]]));
      const zs = a => { const m = a.reduce((t, v) => t + v, 0) / a.length, s = Math.sqrt(a.reduce((t, v) => t + (v - m) ** 2, 0) / (a.length - 1)); return s > 0 ? a.map(v => (v - m) / s) : null; };
      const preds = [zs(mrow.map(i => Math.log10(pts.ecd[Q[i]]))), zs(mrow.map(i => pts.convexity[Q[i]])), zs(mrow.map(i => pts.elongation[Q[i]]))];
      const morph = mrow.length >= 20 && preds.every(Boolean) ? olsHC3(preds, mrow.map(i => use[i])) : null;
      // coverage without a threshold, both populations free (anchored on a reference under Coverage)
      const mixFree = mixtureCoverage(R, sR);
      coat = { band: coatBand, n: Q.length, Q, R, sR, use, trend, w, zc, nC, raw, H, drift, rho, rhoCI, zdev, cat, below: cnt[0], within: cnt[1], above: cnt[2], thin, morph, mixFree,
               spectra: groupSpectra(p => cat[p], 3),
               ends: [median(trend.slice(0, tenth)), median(trend.slice(-tenth))], moran, rhoSize, rhoSizeCI: spearmanCI(rhoSize, Q.length) };
    }
  }

  // Checks on the measurement. Spectra coverage cannot judge: the coating band with a parameter on a fit
  // bound, or fitted with zero height (counted to bound the coverage, with G on a bound)
  const cOut = new Uint8Array(n);
  if (jC >= 0) { const cp = fc.peaks.find(q => q.name === f.names[jC]); for (let p = 0; p < n; p++) if (Number.isFinite(snr[p])) cOut[p] = f.intensity[p * m + jC] <= 0 || bandOnBound(f, cp, p, jC) ? 1 : 0; }
  // the coating band over G without a band fit, on the same spectra: a check on the fitted ratio
  const ff = jC >= 0 && jC !== jG && Q.length >= 10 ? fitFreeCheck(Q, jG, jC, jD, p => zOf(p, jC) >= kd, kd, kq) : null;
  // the run: where the measured particles lie against every one selected (a run that stops early keeps only
  // the start of its list), and whether the list follows position, which makes stopping early a bias
  let run = null;
  if (status) {
    const loc = [], meas = [];
    for (let p = 0; p < n; p++) if ((status[p] === 1 || status[p] === 2) && !(seam && seam[p])) { loc.push(p); if (status[p] === 1) meas.push(p); }
    if (loc.length >= 10) {
      const X = pts.x, Y = pts.y; let x0 = Infinity, x1 = -Infinity;
      for (const p of loc) { x0 = Math.min(x0, X[p]); x1 = Math.max(x1, X[p]); }
      const left = l => l.filter(p => X[p] < 0.5 * (x0 + x1)).length / l.length, order = loc.map((_, i) => i);
      run = { located: loc.length, measured: meas.length, ksX: ksTest(meas.map(p => X[p]), loc.map(p => X[p])), ksY: ksTest(meas.map(p => Y[p]), loc.map(p => Y[p])),
              leftMeas: left(meas), leftAll: left(loc), orderX: spearman(order, loc.map(p => X[p])), orderY: spearman(order, loc.map(p => Y[p])) };
    }
  }
  // located particles the run counts (the run's filter honoured unless included), off the seams
  let located = 0; for (let p = 0; p < n; p++) if ((!status || status[p] !== 0 || inclExcl) && !(seam && seam[p])) located++;
  // how the particles lie on the slide: nearest neighbours, Clark-Evans, touching and likely merged objects,
  // over the located particles and the field the mosaic images (or the particles' own extent)
  const dIdx = []; for (let p = 0; p < n; p++) if ((!status || status[p] !== 0 || inclExcl) && !(seam && seam[p]) && Number.isFinite(pts.x[p]) && Number.isFinite(pts.y[p])) dIdx.push(p);
  let fieldArea = S.ds.video ? Math.abs((S.ds.video.x[1] - S.ds.video.x[0]) * (S.ds.video.y[1] - S.ds.video.y[0])) : NaN;
  if (!(fieldArea > 0)) { const xs = dIdx.map(p => pts.x[p]), ys = dIdx.map(p => pts.y[p]); fieldArea = (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys)); }
  const disp = dIdx.length >= 3 && fieldArea > 0 ? dispersion(dIdx.map(p => pts.x[p]), dIdx.map(p => pts.y[p]), dIdx.map(p => pts.ecd[p]), dIdx.map(p => pts.area[p]), fieldArea) : null;

  return (S.pfa = { key, n, fitted, gate, inclExcl, cOut, ff, run, located, disp, filt, sizeBins, norm: { NS, K2, wn: ks.map(q => S.wn[q]) }, kd, kq, snr, zG, atBound, det, quant, Q, Qd, label, k: cats, how, pops, la, laQ, noD, lambda, size, classes, coat,
                    vwRatio: sw > 0 ? swr / sw : NaN, nwRatio: Qd.length ? sr / Qd.length : NaN, rho, spectra, jD, ratio, rlab, nNd, popBand });
}

function renderPfAnalysis() {
  const A = pfAnalysis(), lab = A.rlab;
  const popColour = l => (A.pops[l] && A.pops[l].nd ? T.label : PARTICLE_COLOURS[l % PARTICLE_COLOURS.length]);
  const card = (k, v, s) => el("div", { class: "card" }, el("div", { class: "k" }, k), el("div", { class: "v" }, v), el("div", { class: "s" }, s));
  const nDet = A.det.reduce((s, v) => s + v, 0), nQ = A.Q.length, nBound = A.atBound.reduce((s, v) => s + v, 0);
  const [dlo, dhi] = wilson(nDet, A.fitted);
  $("#pf-detect-cards").replaceChildren(
    card("Particles used", `${fmt(A.fitted)} of ${fmt(A.n)}`, [
      A.gate.notMeasured ? `${fmt(A.gate.notMeasured)} not measured` : "",
      A.gate.excluded ? `${fmt(A.gate.excluded)} excluded by the run's filter${A.inclExcl ? " (included)" : ""}` : "",
      A.gate.seams ? `${fmt(A.gate.seams)} on tile seams` : ""].filter(Boolean).join(" · ") || (A.n > A.fitted ? `${fmt(A.n - A.fitted)} without a spectrum` : "every particle measured")),
    card("Carbon detected", `${fmt(nDet)} of ${fmt(A.fitted)}`, `${pct(nDet / A.fitted)} (95 % CI ${pct(dlo)}–${pct(dhi)}) · G ≥ ${A.kd} SE`),
    card("Quantified", fmt(nQ), `G ≥ ${A.kq} × noise, inside its bounds`),
    card("Low signal", fmt(nDet - A.Q.filter(p => A.det[p]).length), "detected, not quantified"),
    card("G on a fit bound", fmt(nBound), "fitting noise: left out"));

  renderCoating(A, card);
  renderCoverage(card);
  renderChecks(card); renderAcq(); renderAccept(card);

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
    el("tr", {}, el("td", {}, "By volume"), ...A.size.vol.map(v => el("td", {}, fmt(v, 3)))),
    ...(A.size.nAll > A.size.n ? [el("tr", {}, el("td", { class: "sub" }, "Every object"), ...A.size.all.map(v => el("td", { class: "sub" }, fmt(v, 3))))] : []));
  // coverage by size class (the run's filter and the seams aside), so that no size limit hides in a total
  const bt = $("#pf-sizebin-table"), RC = S.covR && S.covR.mode === "pf" ? S.covR : null; bt.textContent = "";
  $("#pf-sizebin-h").textContent = "By size class";
  bt.append(el("tr", {}, ...["Diameter (µm)", "Located", "Measured", "Quantified", ...(RC ? ["Coated"] : [])].map(h => el("th", {}, h))));
  for (const sb of A.sizeBins) {
    if (!sb.located && !sb.measured) continue;
    const inB = p => S.points.ecd[p] >= sb.lo && S.points.ecd[p] < sb.hi;
    let nJ = 0, nC = 0; if (RC) { for (const p of RC.judged) if (inB(p)) { nJ++; if (RC.coated.has(p)) nC++; } }
    bt.append(el("tr", {}, el("td", {}, sb.hi === Infinity ? `≥ ${sb.lo}` : `${sb.lo}–${sb.hi}`), el("td", {}, fmt(sb.located)), el("td", {}, fmt(sb.measured)),
      el("td", {}, fmt(sb.quantified), el("div", { class: "sub" }, sb.measured ? pct(sb.quantified / sb.measured) : "")),
      ...(RC ? [el("td", {}, nJ ? pct(nC / nJ) : "–", el("div", { class: "sub" }, nJ ? `${fmt(nC)} of ${fmt(nJ)} · ≥ ${pct(binomLower(nJ, nJ - nC))}` : "none judged"))] : [])));
  }
  const ct = $("#pf-sizeclass-table"); ct.textContent = "";
  if (A.classes.length) {
    ct.append(el("tr", {}, ...["Diameter class (µm)", "n", `${lab} median`, "IQR", "La (nm)"].map(h => el("th", {}, h))));
    for (const cl of A.classes) ct.append(el("tr", {}, el("td", {}, `${fmt(cl.lo, 3)}–${fmt(cl.hi, 3)}`), el("td", {}, fmt(cl.n)),
      el("td", {}, fmt(cl.ratio, 3)), el("td", { class: "sub" }, `${fmt(cl.q1, 3)}–${fmt(cl.q3, 3)}`), el("td", {}, fmt(cl.la, 3))));
  }
  $("#pf-size-note").textContent = `D10, D50, D90: the ${fmt(A.size.n)} particles measured, off the tile seams and as the run's filter left them, by number and weighted by the ParticleFinder volume` +
    (A.size.nAll > A.size.n ? `; every object in the table (${fmt(A.size.nAll)}, as the routine's own histogram counts them) for comparison` : "") +
    `. Classes: quartiles of diameter among the quantified particles with ${A.popBand} detected. By size class: located (selected or measured, off the seams), measured, quantified${RC ? ", and the share judged coated above τ with its exact 95 % lower bound (Coverage)" : ""}.` +
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
function pfAxes(host, xr, yr, { logx = false, xlabel = "", ylabel = "", yticks = true } = {}) {
  const W = host.clientWidth, H = host.clientHeight; if (W < 40 || H < 40) return null;
  const ctx = hidpi(host.querySelector("canvas"), W, H);
  const L = 44, R = 10, Tp = 10, B = 34, X = v => L + ((logx ? Math.log10(v) : v) - xr[0]) / (xr[1] - xr[0]) * (W - L - R), Y = v => Tp + (1 - (v - yr[0]) / (yr[1] - yr[0])) * (H - Tp - B);
  ctx.font = font(11); ctx.textBaseline = "middle"; ctx.textAlign = "right";
  if (yticks) for (const t of niceTicks(yr[0], yr[1], 4)) { ctx.fillStyle = T.grid; ctx.fillRect(L, Y(t), W - L - R, 1); ctx.fillStyle = T.ink; ctx.fillText(fmt(t, 3), L - 4, Y(t)); }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const xt = logx ? [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000].filter(v => Math.log10(v) >= xr[0] && Math.log10(v) <= xr[1]) : niceTicks(xr[0], xr[1], 6);
  for (const t of xt) { ctx.fillStyle = T.grid; ctx.fillRect(X(t), Tp, 1, H - Tp - B); ctx.fillStyle = T.ink; ctx.fillText(fmt(t, 3), X(t), H - B + 4); }
  ctx.fillStyle = T.label; ctx.fillText(xlabel, (L + W - R) / 2, H - 15);
  if (ylabel) { ctx.save(); ctx.translate(12, (Tp + H - B) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle"; ctx.fillText(ylabel, 0, 0); ctx.restore(); }
  return { ctx, X, Y, W, H, L, R, Tp, B };
}

function drawPopSpectra() {
  const A = S.pfa; if (!A) return;
  const mode = $("#pf-spec-mode").value;
  if (mode !== "mean") { drawSpecHeat(A, mode === "diff"); return; }
  const byCoat = $("#pf-colour").value === "coat" && A.coat;
  $("#pf-spec-h").textContent = (byCoat ? `by coating group (${A.coat.band}/G against the noise band)` : "by population") + " · baseline removed, G = 1, band ± 1 SD";
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

// Every quantified particle's spectrum as one row (baseline removed, scaled to its G
// height), low coating index at the bottom: how the bands change from the least to the
// most coated particle, with odd spectra as stripes. Rows are averaged when there are more
// particles than pixel rows. Minus the median: each row less the median spectrum, on a
// diverging scale, so what sets particles apart stands out from what they share.
const diverge = t => {                                   // -1 blue, 0 near white, +1 red
  const a = [33, 102, 172], m = [247, 247, 247], b = [178, 24, 43], u = Math.abs(t), e = t < 0 ? a : b;
  return m.map((v, i) => Math.round(v + (e[i] - v) * u));
};
function drawSpecHeat(A, diff) {
  const nm = A.norm, K2 = nm ? nm.K2 : 0, C = A.coat;
  const key = C ? C.R : A.Q.map(p => (A.ratio ? A.ratio[p] : NaN)), band = C ? C.band : A.popBand;
  const order = A.Q.map((_, i) => i).filter(i => Number.isFinite(key[i])).sort((a, b) => key[a] - key[b]), n = order.length;
  if (!n || !K2) return;
  const ax = pfAxes(pfSpecHost, [nm.wn[0], nm.wn[K2 - 1]], [0, 1], { xlabel: "Raman shift (cm⁻¹)", ylabel: `${band}/G, low to high`, yticks: false });
  if (!ax) return;
  const { ctx, X, W, H, L, R, Tp, B } = ax, pw = Math.max(1, Math.round(W - L - R)), ph = Math.max(1, Math.round(H - Tp - B));
  // each pixel row: the mean of the particles that fall on it
  const img = new Float32Array(ph * K2);
  for (let r = 0; r < ph; r++) {
    const a = Math.min(n - 1, Math.floor((ph - 1 - r) * n / ph)), b = Math.max(a + 1, Math.floor((ph - r) * n / ph));
    for (let k = a; k < b; k++) { const o = order[k] * K2; for (let i = 0; i < K2; i++) img[r * K2 + i] += nm.NS[o + i] / (b - a); }
  }
  if (diff) {                                              // the median spectrum, channel by channel
    const col = new Float32Array(n);
    for (let i = 0; i < K2; i++) {
      for (let k = 0; k < n; k++) col[k] = nm.NS[order[k] * K2 + i];
      const med = quantile(Float32Array.from(col).sort(), 0.5);
      for (let r = 0; r < ph; r++) img[r * K2 + i] -= med;
    }
  }
  // colour range: 0 to the 99th percentile on a square-root scale (weak bands show), or
  // a symmetric range from the 1st and 99th percentiles of the differences
  const all = Float32Array.from(img).sort();
  const top = (diff ? Math.max(Math.abs(quantile(all, 0.01)), Math.abs(quantile(all, 0.99))) : quantile(all, 0.99)) || 1;
  const V = lut("Viridis"), rgb = v => {
    if (diff) return diverge(Math.max(-1, Math.min(1, v / top)));
    const j = Math.round(Math.sqrt(Math.max(0, Math.min(1, v / top))) * 255) * 3; return [V[j], V[j + 1], V[j + 2]];
  };
  const off = document.createElement("canvas"); off.width = pw; off.height = ph;
  const og = off.getContext("2d"), id = og.createImageData(pw, ph);
  for (let r = 0; r < ph; r++) for (let c = 0; c < pw; c++) {
    const q = rgb(img[r * K2 + Math.min(K2 - 1, Math.floor(c * K2 / pw))]), o = (r * pw + c) * 4;
    id.data[o] = q[0]; id.data[o + 1] = q[1]; id.data[o + 2] = q[2]; id.data[o + 3] = 255;
  }
  og.putImageData(id, 0, 0);
  ctx.imageSmoothingEnabled = false; ctx.drawImage(off, L, Tp, pw, ph);
  // the coating index at the quartiles of the order
  ctx.font = font(11); ctx.fillStyle = T.ink; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const f of [0, 0.25, 0.5, 0.75, 1]) ctx.fillText(fmt(key[order[Math.round(f * (n - 1))]], 2), L - 4, Tp + (1 - f) * ph);
  // where the coating band and G sit
  ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.font = font(10, "bold");
  for (const name of [band, "G"]) {
    const pk = S.cfg.peaks.find(q => q.name === name); if (!pk || pk.pos < nm.wn[0] || pk.pos > nm.wn[K2 - 1]) continue;
    ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,.6)"; ctx.strokeText(name, X(pk.pos), Tp + 2); ctx.fillStyle = "#fff"; ctx.fillText(name, X(pk.pos), Tp + 2);
  }
  // colour bar, inside at the right
  const cx = L + pw - 12, ch = Math.min(70, ph * 0.5);
  for (let k = 0; k < ch; k++) { const t = 1 - k / ch; ctx.fillStyle = `rgb(${diff ? diverge(2 * t - 1) : rgb(top * t * t)})`; ctx.fillRect(cx, Tp + 6 + k, 8, 1); }
  ctx.strokeStyle = "rgba(255,255,255,.7)"; ctx.lineWidth = 1; ctx.strokeRect(cx - 0.5, Tp + 5.5, 9, ch + 1);
  ctx.font = font(10); ctx.textAlign = "right"; ctx.textBaseline = "middle";
  const ticks = diff ? [[top, `+${fmt(top, 2)}`], [0, "0"], [-top, `−${fmt(top, 2)}`]] : [[top, fmt(top, 2)], [top / 4, fmt(top / 4, 2)], [0, "0"]];
  ticks.forEach(([v, t]) => {
    const y = Tp + 6 + (diff ? (1 - (v / top + 1) / 2) : 1 - Math.sqrt(v / top)) * ch;
    ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,.6)"; ctx.strokeText(t, cx - 3, y); ctx.fillStyle = "#fff"; ctx.fillText(t, cx - 3, y);
  });
  $("#pf-spec-h").textContent = `${fmt(n)} particles by ${band}/G · baseline removed, G = 1` + (diff ? " · less the median spectrum, blue below, red above" : " · square-root colour scale") + (n > ph ? ` · ${fmt(n / ph, 2)} particles per row` : "");
}

function drawPfDist() {
  const A = S.pfa; if (!A) return;
  const kind = $("#pf-dist").value;
  if (kind === "funnel" || kind === "cater" || kind === "order" || kind === "dev") { drawCoatPlot(A, kind); return; }
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

// =========================================================================
// Coating overview: the verdict at a glance, above the numbers
// =========================================================================
// A headline with its status (an icon and words, never colour alone), a meter of the real share of the
// spread, three tiles with their intervals drawn, where the spread comes from (one stacked bar), and the
// distribution: measured values, what noise alone would give, and the true values with the noise taken
// out. Every number in it is also in the cards and notes below.
const OV_ICON = { good: "✓", warn: "!", bad: "✕" };
// text on a fill: dark or white by the fill's luminance
function inkOn(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec((hex || "").trim()); if (!m) return "#ffffff";
  const lin = i => { const v = parseInt(m[1].slice(i, i + 2), 16) / 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * lin(0) + 0.7152 * lin(2) + 0.0722 * lin(4) > 0.36 ? "#0b1226" : "#ffffff";
}
// a value and its 95 % interval on a scale from 0 to max: the interval a band, the value a ringed dot
function ciGlyph(est, lo, hi, max, colour) {
  const f = v => 100 * Math.max(0, Math.min(1, v / max)), P = v => `${f(v).toFixed(2)}%`;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("role", "img"); svg.setAttribute("aria-label", `${fmt(est, 3)}, 95 % interval ${fmt(lo, 3)} to ${fmt(hi, 3)}`);
  svg.innerHTML = `<line x1="0" y1="7" x2="100%" y2="7" style="stroke: var(--line-2); stroke-width: 1"/>` +
    (Number.isFinite(lo) && Number.isFinite(hi) ? `<rect x="${P(lo)}" y="3.5" width="${Math.max(0, f(hi) - f(lo)).toFixed(2)}%" height="7" rx="3.5" style="fill: ${colour}; opacity: 0.28"/>` : "") +
    `<circle cx="${P(est)}" cy="7" r="4.5" style="fill: ${colour}; stroke: var(--input); stroke-width: 2"/>`;
  return svg;
}

function ovRender(host, o) {
  host.replaceChildren();
  if (!o) return;
  host.append(el("div", { class: `ov-head ${o.level}` }, el("span", { class: "ic", "aria-hidden": "true" }, OV_ICON[o.level] || "·"), el("span", {}, o.head)));
  if (o.sub) host.append(el("div", { class: "ov-sub" }, o.sub));
  if (o.meter) {
    const m = o.meter, fill = el("div", { class: `ov-fill ${o.level === "good" ? "" : o.level}`, style: "width:0%" });
    host.append(el("div", { class: "ov-meter", role: "img", "aria-label": `${m.title}: ${m.label}` },
      el("div", { class: "lbl" }, el("span", {}, m.title), el("b", {}, m.label)),
      el("div", { class: "ov-track" }, fill, el("div", { class: "ov-zone", style: "left:25%" }), el("div", { class: "ov-zone", style: "left:75%" })),
      el("div", { class: "ov-scale" }, el("span", {}, "nearly even"), el("span", {}, "moderately patchy"), el("span", {}, "strongly"))));
    requestAnimationFrame(() => { fill.style.width = `${Math.max(1.2, 100 * Math.min(1, m.share))}%`; });
  }
  host.append(el("div", { class: "ov-tiles" }, ...o.tiles.map(t => el("div", { class: "ov-tile" },
    el("div", { class: "k" }, t.k), el("div", { class: "v" }, t.v), t.ci ? ciGlyph(...t.ci, T.viz1) : "", el("div", { class: "s" }, t.s)))));
  if (o.split && o.split.some(q => q.share > 0)) {
    const parts = o.split.filter(q => q.share > 0), width = Math.max(100, host.clientWidth - 24 - 2 * (parts.length - 1));
    const segs = parts.map(q => el("span", { style: `flex-grow:0.0001; background:${q.colour}; color:${inkOn(q.colour)}`, title: `${q.name}: ${pct(q.share)} of the variance` },
      q.share * width >= 48 ? pct(q.share) : ""));
    host.append(el("div", { class: "ov-h" }, o.splitTitle),
      el("div", { class: "ov-bar", role: "img", "aria-label": `${o.splitTitle}: ${parts.map(q => `${q.name} ${pct(q.share)}`).join(", ")}` }, ...segs),
      el("div", { class: "ov-legend" }, ...parts.map(q => el("span", {}, el("span", { class: "sw", style: `background:${q.colour}` }), `${q.name} · ${pct(q.share)}`))));
    requestAnimationFrame(() => segs.forEach((s, i) => { s.style.flexGrow = String(parts[i].share); }));
  }
  for (const pl of o.plots || []) {
    const box = el("div", { class: "ov-plot", style: pl.height ? `height:${pl.height}px` : "" }, el("canvas"), el("div", { class: "ov-tip" }));
    host.append(el("div", { class: "ov-h" }, pl.title), box);
    if (pl.legend) host.append(el("div", { class: "ov-legend" }, ...pl.legend.map(([c, n, kind]) => el("span", {}, el("span", { class: "sw", style: kind === "line" ? `background:${c}; height:3px; border-radius:2px; vertical-align:3px` : `background:${c}` }), n))));
    // drawn once laid out, and again whenever the box changes width (a narrower window, a panel opened)
    let lastW = -1;
    new ResizeObserver(() => { const w = box.clientWidth; if (w && w !== lastW) { lastW = w; pl.draw(box); } }).observe(box);
  }
}

// Histogram of values with smooth curves over it (densities scaled to counts), a shaded lower tail under
// one of them and labelled vertical lines; a tooltip per bin.
function drawDistPlot(host, { values, curves = [], tail = null, lines = [], xlabel = "", unit = "particles" }) {
  const v = Float64Array.from(values).filter(Number.isFinite).sort(); if (v.length < 5) return;
  let lo = quantileSorted(v, 0.005), hi = quantileSorted(v, 0.995);
  for (const ln of lines) if (Number.isFinite(ln.x)) { lo = Math.min(lo, ln.x); hi = Math.max(hi, ln.x); }
  const pad = 0.04 * (hi - lo || 1); lo -= pad; hi += pad;
  const iqr = quantileSorted(v, 0.75) - quantileSorted(v, 0.25), fd = 2 * iqr / Math.cbrt(v.length);
  const nb = Math.max(12, Math.min(48, Math.round((hi - lo) / (fd || (hi - lo) / 24)))), bw = (hi - lo) / nb, cnt = new Float64Array(nb);
  for (const x of v) { const b = Math.floor((x - lo) / bw); if (b >= 0 && b < nb) cnt[b]++; }
  const xs = Array.from({ length: 160 }, (_, i) => lo + (i + 0.5) * (hi - lo) / 160);
  const cur = curves.map(c => ({ ...c, y: xs.map(x => c.f(x) * v.length * bw) }));
  const ymax = Math.max(...cnt, ...cur.flatMap(c => c.y)) * 1.12 || 1;
  const ax = pfAxes(host, [lo, hi], [0, ymax], { xlabel, ylabel: unit }); if (!ax) return;
  const { ctx, X, Y } = ax;
  // the bars: neutral, since the curves carry the meaning; 2 px of surface between them
  ctx.fillStyle = T.label; ctx.globalAlpha = 0.3;
  for (let b = 0; b < nb; b++) {
    if (!cnt[b]) continue;
    const x0 = X(lo + b * bw) + 1, x1 = X(lo + (b + 1) * bw) - 1, y0 = Y(cnt[b]), r = Math.min(2, (x1 - x0) / 2);
    ctx.beginPath(); ctx.moveTo(x0, Y(0)); ctx.lineTo(x0, y0 + r); ctx.quadraticCurveTo(x0, y0, x0 + r, y0); ctx.lineTo(x1 - r, y0); ctx.quadraticCurveTo(x1, y0, x1, y0 + r); ctx.lineTo(x1, Y(0)); ctx.fill();
  }
  ctx.globalAlpha = 1;
  // the shaded tail under its curve, then the curves (a wash under the main one)
  if (tail) {
    const c = cur.find(q => q.name === tail.curve);
    if (c) {
      ctx.beginPath(); ctx.moveTo(X(xs[0]), Y(0));
      xs.forEach((x, i) => { if (x <= tail.x) ctx.lineTo(X(x), Y(c.y[i])); });
      ctx.lineTo(X(Math.min(tail.x, xs[xs.length - 1])), Y(0)); ctx.closePath();
      ctx.fillStyle = tail.colour; ctx.globalAlpha = 0.3; ctx.fill(); ctx.globalAlpha = 1;
    }
  }
  for (const c of cur) {
    if (c.wash) { ctx.beginPath(); ctx.moveTo(X(xs[0]), Y(0)); xs.forEach((x, i) => ctx.lineTo(X(x), Y(c.y[i]))); ctx.lineTo(X(xs[xs.length - 1]), Y(0)); ctx.closePath(); ctx.fillStyle = c.colour; ctx.globalAlpha = 0.1; ctx.fill(); ctx.globalAlpha = 1; }
    ctx.beginPath(); xs.forEach((x, i) => (i ? ctx.lineTo(X(x), Y(c.y[i])) : ctx.moveTo(X(x), Y(c.y[i]))));
    ctx.strokeStyle = c.colour; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.stroke();
  }
  // labelled lines, the labels in text colour on a small surface backing
  ctx.font = font(10.5); ctx.textBaseline = "top";
  lines.forEach((ln, k) => {
    if (!Number.isFinite(ln.x)) return;
    const x = X(ln.x); ctx.strokeStyle = ln.colour; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x, ax.Tp); ctx.lineTo(x, Y(0)); ctx.stroke();
    const w = ctx.measureText(ln.label).width, right = x + 4 + w < ax.W - ax.R, tx = right ? x + 4 : x - 4 - w, ty = ax.Tp + 2 + 14 * k;
    ctx.fillStyle = T.paper; ctx.globalAlpha = 0.85; ctx.fillRect(tx - 2, ty - 1, w + 4, 13); ctx.globalAlpha = 1;
    ctx.fillStyle = T.ink; ctx.textAlign = "left"; ctx.fillText(ln.label, tx, ty);
  });
  // a tooltip per bin
  const tip = host.querySelector(".ov-tip"), cv = host.querySelector("canvas");
  cv.onmousemove = e => {
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, b = Math.floor(((mx - ax.L) / (ax.W - ax.L - ax.R)) * nb);
    if (b < 0 || b >= nb) { tip.style.display = "none"; return; }
    tip.textContent = `${fmt(lo + b * bw, 3)} to ${fmt(lo + (b + 1) * bw, 3)}: ${fmt(cnt[b])} ${unit} (${pct(cnt[b] / v.length)})`;
    tip.style.left = `${Math.min(Math.max(mx, 70), ax.W - 70)}px`; tip.style.top = `${Y(cnt[b])}px`; tip.style.display = "block";
  };
  cv.onmouseleave = () => { tip.style.display = "none"; };
}

// One row per particle: its within-particle CV with its 95 % interval, patchy ones (p < 0.05) apart
function drawPartStrip(host, rows) {
  const max = Math.max(0.05, ...rows.map(r => (Number.isFinite(r.hi) ? r.hi : r.est))) * 1.08;
  const ax = pfAxes(host, [0, 100 * max], [0, rows.length], { xlabel: "Within-particle CV (%), noise taken out", yticks: false }); if (!ax) return;
  const { ctx, X, Y } = ax;
  ctx.font = font(11); ctx.textBaseline = "middle"; ctx.textAlign = "right";
  rows.forEach((r, i) => {
    const y = Y(rows.length - i - 0.5), c = r.patchy ? T.viz2 : T.viz1;
    ctx.fillStyle = T.ink; ctx.fillText(`#${r.id}`, ax.L - 6, y);
    if (Number.isFinite(r.lo) && Number.isFinite(r.hi)) { ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.lineCap = "round"; ctx.beginPath(); ctx.moveTo(X(100 * r.lo), y); ctx.lineTo(X(100 * r.hi), y); ctx.stroke(); }
    ctx.beginPath(); ctx.arc(X(100 * r.est), y, 4.5, 0, 2 * Math.PI); ctx.fillStyle = c; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = T.paper; ctx.stroke();
  });
  const tip = host.querySelector(".ov-tip"), cv = host.querySelector("canvas");
  cv.onmousemove = e => {
    const rr = cv.getBoundingClientRect(), my = e.clientY - rr.top, i = rows.length - 1 - Math.floor((Y(0) - my) / ((Y(0) - Y(rows.length)) / rows.length));
    const r = rows[i]; if (!r) { tip.style.display = "none"; return; }
    tip.textContent = `particle ${r.id}: CV ${pct(r.est)} (${pct(r.lo)}–${pct(r.hi)}) · ${r.patchy ? "patchy" : "even within the noise"} · ${fmt(r.n)} pixels`;
    tip.style.left = `${Math.min(Math.max(X(100 * r.est), 120), ax.W - 120)}px`; tip.style.top = `${Y(rows.length - i - 0.5) - 6}px`; tip.style.display = "block";
  };
  cv.onmouseleave = () => { tip.style.display = "none"; };
}

// the notes of a verdict folded away: the overview carries its headline
function foldNotes(box) {
  const notes = [...box.children].filter(c => c.tagName === "P");
  box.className = "verdict";
  box.replaceChildren(...(notes.length ? [el("details", { class: "ov-more" }, el("summary", {}, `What this means · ${notes.length} note${notes.length > 1 ? "s" : ""}`), ...notes)] : []));
}
const npdfAt = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));

function renderCoating(A, card) {
  const C = A.coat, box = $("#pf-verdict");
  if (!C) {
    box.className = "verdict"; box.textContent = ""; $("#pf-ov").replaceChildren();
    $("#pf-coat-cards").replaceChildren(); $("#pf-more-cards").replaceChildren(); $("#pf-more-cards-h").textContent = "";
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

  // beyond the CV: how unequal the coating index is between particles, against what their noise alone gives
  const gObs = gini(C.use), gNoise = noiseGini(H.mu, C.sR), logn = lognormalPreferred(C.use), mo = moments(C.use);
  $("#pf-more-cards-h").textContent = "Beyond the CV, between particles";
  $("#pf-more-cards").replaceChildren(
    card("Inequality (Gini)", fmt(gObs, 3), `noise alone gives ${fmt(gNoise, 3)} · excess ${fmt(gObs - gNoise, 2)}`),
    card("Distribution", logn == null ? "–" : logn ? "right-skewed" : "symmetric", `${logn == null ? "too few particles" : logn ? "lognormal fits better than normal" : "normal fits as well as lognormal"} · log-scale CV ${pct(logCV(C.use))}`),
    card("Spread without a mean", `QCD ${fmt(qcd(C.use), 3)}`, `q90/q10 ${fmt(qRatio(C.use), 3)} · the lowest tenth holds ${pct(lowShare(C.use))} of the total (10 % if even)`),
    card("Shape", `skew ${fmt(mo.skew, 2)}`, `excess kurtosis ${fmt(mo.kurt, 2)} · evenness ${fmt(evenness(C.use, 32), 2)} (Shannon, 32 bins; 1 spreads evenly)`));
  if (logn) box.append(el("p", {}, `${b}/G is right-skewed between particles (a lognormal fits better than a normal), so a limit belongs on its logarithm or its quantiles, not as a band about the mean.`));

  // the thinnest particles: the lower tail of the true coating index, each particle's error taken out
  const Tn = C.thin, L = parseFloat($("#acc-lsl").value);
  $("#pf-thin-h").textContent = Tn ? "The thinnest particles" : "";
  const bl = Tn && Number.isFinite(L) ? Tn.below(L) : null;
  $("#pf-thin-cards").replaceChildren(...(Tn ? [
    card(`Thinnest 5 %: ${b}/G`, `≤ ${fmt(Tn.value, 3)}`, `95 % CI ${fmt(Tn.ci[0], 3)}–${fmt(Tn.ci[1], 3)} · noise taken out${Tn.log ? " · lognormal" : ""}`),
    card("The next particle", `${fmt(Tn.pred[0], 3)} to ${fmt(Tn.pred[1], 3)}`, "95 % prediction interval of its true value"),
    bl ? card(`Below ${fmt(L, 3)}`, pct(bl.share), `95 % CI ${pct(bl.ci[0])}–${pct(bl.ci[1])} · Cpk ${fmt(bl.cpk, 2)} (${fmt(bl.cpkCI[0], 2)}–${fmt(bl.cpkCI[1], 2)})`)
       : card("Below a lower limit", "–", "set it under Acceptance")] : []));
  if (Tn) {
    const plain = quantileSorted(Float64Array.from(C.use).sort(), 0.05);
    box.append(el("p", {}, `The thinnest 5 % of particles carry ${b}/G of ${fmt(Tn.value, 3)} or less once each particle's measurement error is taken out (95 % CI ${fmt(Tn.ci[0], 3)}–${fmt(Tn.ci[1], 3)}); the measured values alone put it at ${fmt(plain, 3)}, ${plain < Tn.value ? "lower, because noise widens the spread" : "about the same: the noise is small beside the real spread"}.` +
      (bl ? ` ${pct(bl.share)} of particles lie below the limit of ${fmt(L, 3)} (95 % CI ${pct(bl.ci[0])}–${pct(bl.ci[1])}), an equivalent Cpk of ${fmt(bl.cpk, 2)}.` : "")));
  }
  // coating against morphology
  const M = C.morph, mt = $("#pf-morph-table"); mt.textContent = "";
  $("#pf-morph-h").textContent = M ? "Coating against morphology" : "";
  if (M) {
    const tq = tQuantile(0.975, M.df);
    mt.append(el("tr", {}, ...["+1 SD of", `${b}/G`, "95 % CI", "p", "Holm p"].map(h => el("th", {}, h))));
    ["Diameter (log)", "Convexity", "Elongation"].forEach((name, a) => {
      const k = a + 1, sig = M.holm[k] < 0.05;
      mt.append(el("tr", { class: sig ? "on" : "" }, el("td", {}, name), el("td", {}, `${M.beta[k] >= 0 ? "+" : ""}${fmt(M.beta[k], 3)}`, el("div", { class: "sub" }, `${M.beta[k] >= 0 ? "+" : ""}${pct(M.beta[k] / H.mu)} of the mean`)),
        el("td", { class: "sub" }, `${fmt(M.beta[k] - tq * M.se[k], 3)} to ${fmt(M.beta[k] + tq * M.se[k], 3)}`), el("td", {}, pFmt(M.p[k])), el("td", {}, pFmt(M.holm[k]))));
    });
    const dirs = [1, 2, 3].filter(k => M.holm[k] < 0.05).map(k => `${["larger", "more convex", "more elongated"][k - 1]} particles carry ${M.beta[k] > 0 ? "more" : "less"}`);
    box.append(el("p", {}, dirs.length ? `Against morphology, ${dirs.join(", and ")} coating index, each with the other two held (Holm-adjusted p < 0.05; R² ${fmt(M.r2, 2)}).` : `The coating index does not follow particle size, convexity or elongation once the three are taken together (no Holm-adjusted p < 0.05; R² ${fmt(M.r2, 2)}).`));
  }
  // the overview at the top, and the verdict's notes folded under it
  {
    const noise2 = H.noise ** 2, real2 = H.tau ** 2, drift2 = C.drift ? Math.max(0, C.raw.sd ** 2 - H.sd ** 2) : 0, tot = noise2 + real2 + drift2;
    const trueCurve = Tn && Tn.tau > 1e-9 ? (Tn.log ? (x => (x > 0 ? npdfAt(Math.log(x), Math.log(Tn.mu), Tn.tau) / x : 0)) : (x => npdfAt(x, Tn.mu, Tn.tau))) : null;
    const sRs = C.sR.filter(e => e > 0), noiseCurve = x => { let s = 0; for (const e of sRs) s += npdfAt(x, H.mu, e); return s / sRs.length; };
    const hiCI = H.tauHi / H.mu;
    ovRender($("#pf-ov"), {
      level, head, sub: body,
      meter: { share: H.p >= 0.05 ? 0 : H.I2, title: "Real share of the spread between particles (I²)", label: H.p >= 0.05 ? `${pct(H.I2)} · within the noise` : pct(H.I2) },
      tiles: [
        { k: `Mean ${b}/G`, v: fmt(H.mu, 3), s: `${fmt(C.n)} particles · median ${fmt(median(C.use), 3)}` },
        { k: "Spread (CV)", v: pct(cv), ci: [cv, H.tauLo / H.mu, hiCI, Math.max(0.2, 1.3 * hiCI)], s: `particle to particle · 95 % CI ${pct(H.tauLo / H.mu)}–${pct(hiCI)}` },
        Tn ? { k: "Thinnest 5 %", v: `≤ ${fmt(Tn.value, 3)}`, ci: [Tn.value, Tn.ci[0], Tn.ci[1], 1.6 * H.mu],
               s: bl ? `${pct(bl.share)} below ${fmt(L, 3)} · Cpk ${fmt(bl.cpk, 2)}` : `95 % CI ${fmt(Tn.ci[0], 3)}–${fmt(Tn.ci[1], 3)}` }
           : { k: `${b} detected`, v: pct(C.nC / C.n), s: `${fmt(C.nC)} of ${fmt(C.n)} particles` }],
      splitTitle: "Where the spread between particles comes from (share of the variance)",
      split: [{ name: "measurement noise", share: noise2 / tot, colour: T.muted }, { name: "real, between particles", share: real2 / tot, colour: T.viz1 },
              ...(drift2 > 0 ? [{ name: "drift along the order, removed", share: drift2 / tot, colour: T.viz2 }] : [])],
      plots: [{ title: `${b}/G over the particles`,
        draw: h => drawDistPlot(h, { values: C.use, xlabel: `${b}/G${C.drift ? ", drift removed" : ""}`, unit: "particles",
          curves: [{ name: "noise", f: noiseCurve, colour: T.muted }, ...(trueCurve ? [{ name: "true", f: trueCurve, colour: T.viz1, wash: true }] : [])],
          tail: trueCurve ? { curve: "true", x: Tn.value, colour: T.warn } : null,
          lines: [...(Tn ? [{ x: Tn.value, colour: T.warn, label: "thinnest 5 %" }] : []), ...(Number.isFinite(L) ? [{ x: L, colour: T.bad, label: `limit ${fmt(L, 3)}` }] : [])] }),
        legend: [[T.label, "measured", "box"], [T.muted, "noise alone (one coating for all)", "line"], ...(trueCurve ? [[T.viz1, "true values, noise taken out", "line"]] : []), ...(Tn ? [[T.warn, "thinnest 5 %", "box"]] : [])] }],
    });
    foldNotes(box);
  }
  $("#pf-morph-note").textContent = M ? `Least squares of ${b}/G (drift removed) on the three together, each scaled to its standard deviation (diameter on a log scale), with heteroscedasticity-robust (HC3) errors, t p-values on ${M.df} df and Holm's adjustment over the three. Particles that cluster on the slide (Moran's I above) are not independent, which this does not allow for. Thinnest particles: the particles' true values modelled as ${Tn && Tn.log ? "lognormal" : "normal"} about a mean with each particle's error taken out (Paule-Mandel), intervals from 400 resamples of the particles; the equivalent Cpk is z(1 − share below the limit) / 3.` : "";

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
  if (kind === "cater") {
    // every particle with its 95 % interval, in order of the coating index: an interval that
    // misses the band of the real spread is a particle that differs beyond its noise
    $("#pf-dist-note").textContent = `each particle with its 95 % interval${C.drift ? " · drift removed" : ""}`;
    const idx = C.use.map((_, i) => i).sort((a, b) => C.use[a] - C.use[b]), n = idx.length;
    const ylo = quantileSorted(qs(idx.map(i => C.use[i] - 1.96 * C.sR[i])), 0.01), yhi = quantileSorted(qs(idx.map(i => C.use[i] + 1.96 * C.sR[i])), 0.99);
    const ax = pfAxes(pfDistHost, [0.5, n + 0.5], [ylo, yhi], { xlabel: `Particles, in order of ${b}/G`, ylabel: `${b}/G` }); if (!ax) return;
    const { ctx, X, Y } = ax, pw = ax.W - ax.L - ax.R;
    ctx.save(); ctx.beginPath(); ctx.rect(ax.L, ax.Tp, pw, ax.H - ax.Tp - ax.B); ctx.clip();
    // the real spread between particles about their mean (± 1.96 tau), and the mean
    ctx.globalAlpha = 0.16; ctx.fillStyle = T.strong;
    ctx.fillRect(ax.L, Y(H.mu + 1.96 * H.tau), pw, Y(H.mu - 1.96 * H.tau) - Y(H.mu + 1.96 * H.tau));
    ctx.globalAlpha = 1; ctx.fillStyle = T.strong; ctx.fillRect(ax.L, Y(H.mu) - 0.5, pw, 1.5);
    ctx.globalAlpha = Math.max(0.15, Math.min(0.9, 600 / n)); ctx.lineWidth = Math.max(0.6, Math.min(2, 0.8 * pw / n));
    idx.forEach((i, k) => {
      const x = X(k + 1); ctx.strokeStyle = CC[C.cat[C.Q[i]]];
      ctx.beginPath(); ctx.moveTo(x, Y(C.use[i] - 1.96 * C.sR[i])); ctx.lineTo(x, Y(C.use[i] + 1.96 * C.sR[i])); ctx.stroke();
    });
    ctx.globalAlpha = 1; ctx.fillStyle = T.strong;
    const dot = Math.max(1.2, Math.min(3, 0.6 * pw / n));
    idx.forEach((i, k) => ctx.fillRect(X(k + 1) - dot / 2, Y(C.use[i]) - dot / 2, dot, dot));
    ctx.restore();
    ctx.font = font(11, "bold"); ctx.textAlign = "left"; ctx.textBaseline = "top";
    [[`above · ${pct(C.above / C.n)}`, CC[2]], [`within · ${pct(C.within / C.n)}`, CC[1]], [`below · ${pct(C.below / C.n)}`, CC[0]]].forEach(([t, col], k) => { ctx.fillStyle = col; ctx.fillText(t, ax.L + 8, ax.Tp + 4 + k * 15); });
    ctx.fillStyle = T.label; ctx.font = font(10); ctx.fillText("line: the mean · band: the real spread between particles (± 1.96 τ)", ax.L + 8, ax.Tp + 50);
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

for (const id of ["#pf-kd", "#pf-kq", "#pf-incl-excl", "#pf-coat-band", "#pf-colour", "#pf-pop-ratio", "#pf-pop-mode", "#pf-cut1", "#pf-cut2", "#pf-laser"]) $(id).addEventListener("change", () => { if (S.points && S.fit) renderPF(); });
$("#pf-dist").onchange = () => drawPfDist();
$("#pf-spec-mode").onchange = () => drawPopSpectra();
$("#pf-scatter").onchange = () => drawScatter();
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

// The Particles tab's results for this file, as plain numbers for the report. The tab's analyses
// run first (with its current settings) in case it has not been opened on this file, and the
// calibration's fits are awaited.
async function reportAnalysis() {
  if (!S.fit || S.fit.resultsOnly || !S.cube) return null;
  particlesInit();
  // a map works out its band-height errors after drawing when there are many pixels
  if (!S.points) for (let i = 0; i < 150 && !(S.segCoat && S.segCoat.seg === S.seg); i++) await sleep(200);
  const ch = d3gChoice(), cur = fileOf(S.ds.id), refF = $("#d3g-ref").value ? fileOf($("#d3g-ref").value) : null;
  try { await d3gResult(cur, ch); if (refF) await d3gResult(refF, ch); } catch { /* d3gSummary says why */ }
  const level = node => ["good", "warn", "bad"].find(c => node.classList.contains(c)) || null;
  const verdict = sel => { const b = $(`${sel} b`), box = $(sel); return { head: b ? b.textContent : "", level: level(box) }; };
  // the coating verdict's headline sits in the overview above the notes
  const overview = (ov, sel) => { const h = $(`${ov} .ov-head`); return h ? { head: h.lastElementChild.textContent, level: level(h) } : verdict(sel); };
  const A = { kind: S.points ? "pf" : "map" };

  const Z = d3gSummary();
  if (Z.res) {
    const sh = Z.ch.step.shapes, set = Z.ch.cal.settings;
    A.calib = { name: `${Z.ch.cal.name} · ${Z.ch.step.name}`, unit: Z.unit, n: Z.res.n, measured: Z.res.measured, median: Z.M.median, lo: Z.M.lo, hi: Z.M.hi,
      dp: Z.dp, rn: Z.res.rn, cmp: Z.cmp, refMed: Z.refMed, ref: Z.cmp ? { name: Z.refName } : null,
      bands: D3G_BANDS.map(b => ({ name: b, pos: sh[b].pos, fwhm: sh[b].fwhm, shape: set.bands[b].shape,
        from: /^median spectrum/.test(sh[b].note || "") ? "median spectrum" : sh[b].support > 1 ? `median of ${fmt(sh[b].support)} spectra` : sh[b].note || "" })) };
  } else A.calib = { missing: Z.pending ? "Still fitting the heights: refresh the report in a moment." : Z.missing || "Not available." };

  // coverage against the uncoated reference: how many particles, or how much of the area, carry the coating
  const R = S.covR, refName = $("#cov-ref").value ? (fileOf($("#cov-ref").value) || {}).name : "";
  if (R && R.mode === "pf") {
    A.cov = { mode: "pf", band: R.band, tau: R.F.tau, gmin: R.gmin, refName, x: R.x, n: R.n, pool: R.pool, e: R.e, c: R.c, lower: R.lower };
  } else if (R && R.mode === "map") {
    const pa = Math.abs((S.x[1] - S.x[0]) * (S.y[1] - S.y[0])), all = new Map(S.seg.parts.map(p => [p.id, p.pixels.length]));
    const parts = R.parts.map(q => ({ id: q.id, area: (all.get(q.id) || 0) * pa, judged: q.n * pa, covered: q.c * q.n * pa, c: q.c, lo: q.lo, hi: q.hi, bound: q.bound }));
    const sum = k => parts.reduce((t, q) => t + q[k], 0);
    A.cov = { mode: "map", band: R.band, tau: R.F.tau, gmin: R.gmin, refName, parts, lowest: R.lowest,
      total: sum("area"), judged: sum("judged"), covered: sum("covered"), areaShare: sum("covered") / sum("judged") };
  } else A.cov = { reason: $("#cov-note").textContent || "" };

  if (S.points && S.pfa && S.pfa.coat) {
    const C = S.pfa.coat, H = C.H;
    A.uni = { mode: "pf", n: C.n, nC: C.nC, cv: H.tau / H.mu, I2: H.I2, drift: !!C.drift, rhoSize: C.rhoSize, rhoSizeCI: C.rhoSizeCI,
      thin: C.thin ? { value: C.thin.value } : null, ...overview("#pf-ov", "#pf-verdict") };
  } else if (!S.points && S.segCoat && S.segCoat.C) {
    const G = S.segCoat, C = G.C, mu = C.mu;
    A.uni = { mode: "map", W: { cv: C.within.tau / mu }, B: C.between ? { cv: C.between.tau / mu } : null, det: G.nDet / G.n, ...overview("#seg-ov", "#seg-verdict") };
  }
  return A;
}

let reportRun = 0;
async function makeReport() {
  if (!S.fit) { $("#report-paper").replaceChildren(el("p", {}, "Open and fit a map first.")); return; }
  const run = ++reportRun;
  $("#report-paper").replaceChildren(el("p", { class: "note" }, "Gathering the analysis…"));
  let analysis = null;
  try { analysis = await reportAnalysis(); } catch (e) { toast(`Report: ${e.message}`, "error"); }
  if (run !== reportRun) return;                       // a newer refresh has started
  buildReport($("#report-paper"), {
    title: $("#rp-title").value, ds: S.ds, cfg: S.cfg, fit: S.fit, nx: S.nx, ny: S.ny, x: S.x, y: S.y, points: !!S.points,
    figs: reportFigures(), quality: $("#quality"), stats: statRows(), analysis,
    pinned: S.pinned, pstats: S.points ? null : S.pstats, splat: splatState().result, splatImg: sview && splatState().field ? sview.snapshot() : null,
  });
}
$("#rp-refresh").onclick = makeReport;
$("#rp-title").onchange = makeReport;
$("#rp-print").onclick = async () => { await makeReport(); setTimeout(() => window.print(), 150); };

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
