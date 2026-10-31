// Printable report from the current state (Print -> Save as PDF): the results at a glance, how
// many particles (or how much area) carry the coating against the uncoated reference, the
// calibrated D3/G and the analysis in a few plain questions, then the maps, spectra, quality and
// configuration. Charts are inline SVG so they print sharply.

import { el, fmt } from "./util.js";

const img = (src, cap) => el("figure", {}, el("img", { src }), cap ? el("figcaption", {}, cap) : null);

// chart colours for white paper: the app's light-theme set (checked for lightness, chroma,
// colour-blind separation and contrast on white), status colours with an icon and a word
const COL = { ink: "#142033", faint: "#8a98b0", grid: "#dfe6f1", track: "#e8eef7", coated: "#0a86b8", uncoated: "#d9700b", other: "#7346d1" };
const STATUS = { good: ["✓", "good"], warn: ["!", "check"], bad: ["✕", "attention"] };

const NS = "http://www.w3.org/2000/svg";
function sv(tag, attrs = {}, ...kids) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null && v !== false) e.setAttribute(k, v);
  for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(k));
  return e;
}
const pct = (v, d) => (Number.isFinite(v) ? `${(100 * v).toFixed(d ?? (Math.abs(v) < 0.1 ? 1 : 0))} %` : "–");
const fx = (v, dp) => (Number.isFinite(v) ? `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(dp)}` : "–");
const sgn = (v, dp) => (Number.isFinite(v) ? `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(dp)}` : "–");
const area = v => (Number.isFinite(v) ? `${fmt(Math.round(v))} µm²` : "–");
const pill = level => { const [g, w] = STATUS[level]; return el("span", { class: `pill ${level}` }, el("b", {}, g), ` ${w}`); };
// a tile's value: a number large, a verdict in words smaller so it fits the tile
const tile = (k, v, s, level, foot) => el("div", { class: `tile${level ? ` ${level}` : ""}` }, el("div", { class: "k" }, k, level ? pill(level) : null),
  el("div", { class: /[a-z]{3}/i.test(v) ? "v txt" : "v" }, v), el("div", { class: "s" }, s), foot ? el("div", { class: "f" }, foot) : null);
const sec = (n, title, ...kids) => el("section", { class: "rp-sec" }, el("h2", {}, el("span", { class: "num" }, String(n)), title), ...kids);

// --------------------------------------------------------------------------
// charts
// --------------------------------------------------------------------------
// one stacked bar of counts that add up to the whole, a 2 px paper gap between segments
function composition(parts, label) {
  const W = 600, L = 8, R = W - 8, y = 6, h = 18, tot = parts.reduce((t, p) => t + p.n, 0) || 1;
  const g = sv("svg", { viewBox: `0 0 ${W} 34`, class: "chart", role: "img", "aria-label": label });
  let x = L;
  for (const p of parts) {
    const w = (p.n / tot) * (R - L); if (w <= 0) continue;
    g.append(sv("rect", { x: x + (x > L ? 1 : 0), y, width: Math.max(0, w - (x > L ? 2 : 0)), height: h, rx: 3, fill: p.colour, ...(p.hatch ? { fill: "url(#rp-hatch)" } : {}) },
      sv("title", {}, `${p.name}: ${fmt(p.n)} (${pct(p.n / tot, 1)})`)));
    if (w > 46) g.append(sv("text", { x: x + w / 2, y: y + h / 2 + 4, "text-anchor": "middle", class: p.hatch ? "in dark" : "in" }, fmt(p.n)));
    x += w;
  }
  g.prepend(sv("defs", {}, sv("pattern", { id: "rp-hatch", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" },
    sv("rect", { width: 6, height: 6, fill: COL.track }), sv("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: COL.faint, "stroke-width": 2 }))));
  return el("div", {}, g, el("div", { class: "legend" }, ...parts.filter(p => p.n > 0).map(p =>
    el("span", {}, el("i", { class: p.hatch ? "hatch" : "", style: p.hatch ? "" : `background:${p.colour}` }), `${p.name} ${fmt(p.n)}`))));
}

// covered share of each particle of a map: the bar, its range with the left-out pixels counted
// both ways, and the 95 % lower bound
function particleBars(rows) {
  const W = 600, L = 150, R = W - 60, rowH = 24, X = v => L + Math.max(0, Math.min(1, v)) * (R - L), H = rows.length * rowH + 22;
  const g = sv("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": "Covered area of each particle" });
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    g.append(sv("line", { x1: X(t), x2: X(t), y1: 0, y2: rows.length * rowH, stroke: COL.grid, "stroke-width": 1 }));
    g.append(sv("text", { x: X(t), y: H - 6, "text-anchor": "middle", class: "tk" }, pct(t, 0)));
  }
  rows.forEach((r, i) => {
    const y = i * rowH + 4, h = 14;
    g.append(sv("text", { x: L - 8, y: y + 11, "text-anchor": "end", class: "lbl" }, `Particle ${r.id} · ${area(r.area)}`));
    g.append(sv("rect", { x: L, y, width: R - L, height: h, rx: 4, fill: COL.track }));
    g.append(sv("rect", { x: L, y, width: Math.max(0, X(r.c) - L), height: h, rx: 4, fill: COL.coated }, sv("title", {}, `Particle ${r.id}: ${pct(r.c, 1)} covered, 95 % lower bound ${pct(r.bound, 1)}`)));
    if (r.hi > r.lo + 1e-9) g.append(sv("line", { x1: X(r.lo), x2: X(r.hi), y1: y + h / 2, y2: y + h / 2, stroke: COL.ink, "stroke-width": 1.5 }));
    if (Number.isFinite(r.bound)) g.append(sv("line", { x1: X(r.bound), x2: X(r.bound), y1: y - 2, y2: y + h + 2, stroke: COL.ink, "stroke-width": 2 }));
    g.append(sv("text", { x: R + 6, y: y + 11, class: "val" }, pct(r.c)));
  });
  return g;
}

// --------------------------------------------------------------------------
// the sections that need the analysis
// --------------------------------------------------------------------------
function glance(A) {
  const C = A.calib, V = A.cov, U = A.uni, out = [];
  out.push(C && !C.missing ? tile("D3/G (amorphous carbon)", fx(C.median, C.dp), `median · 95 % CI ${fx(C.lo, C.dp)}–${fx(C.hi, C.dp)} · ${fmt(C.n)} carbon ${C.unit}`, null, `calibrated: ${C.name}`)
    : tile("D3/G (amorphous carbon)", "–", C && C.missing ? C.missing : "not run"));
  out.push(C && C.cmp ? tile("Against the uncoated reference", `${C.cmp.ratio.toFixed(2)} ×`, `${sgn(C.cmp.diff, C.dp)} (95 % CI ${fx(C.cmp.lo, C.dp)} to ${fx(C.cmp.hi, C.dp)})`, null, C.ref ? C.ref.name : "")
    : tile("Against the uncoated reference", "–", "no uncoated reference chosen"));
  if (V && V.mode === "pf") out.push(tile("Particles coated", `${fmt(V.x)} of ${fmt(V.n)}`, `${pct(V.c, 1)} have more D3 than the uncoated reference`, null,
    `at least ${pct(V.lower, 1)} (95 % confidence)${V.e ? ` · ${fmt(V.e)} more could not be judged` : ""}`));
  else if (V && V.mode === "map") out.push(tile("Area coated", pct(V.areaShare, 1), `${area(V.covered)} of ${area(V.judged)} have more D3 than the uncoated reference`, null,
    `every particle at least ${pct(V.lowest, 1)} (95 % confidence)`));
  else out.push(tile(A.kind === "pf" ? "Particles coated" : "Area coated", "–", "needs an uncoated reference (Particles › Coverage)"));
  if (U) {
    const [h1] = (U.head || "–").split("; ");               // a map's verdict has two clauses: the first is enough here
    out.push(tile("Coating uniformity", h1, U.mode === "pf" ? `particles differ by ${pct(U.cv)} (CV)` : `places within a particle differ by ${pct(U.W.cv)} (CV)`, U.level));
  } else out.push(tile("Coating uniformity", "–", A.kind === "map" ? "find the particles under Particles" : "too few particles"));
  return el("div", { class: "tiles" }, ...out);
}

function coverageSection(A) {
  const V = A.cov, kids = [];
  if (!V || V.reason) {
    kids.push(el("div", { class: "callout" }, el("b", {}, "Coverage not computed. "),
      (V && V.reason) || "Choose an uncoated reference under Particles › Coverage (an open file, or Choose a file…), then refresh this report."));
    return kids;
  }
  const level = `${V.band}/G above ${fx(V.tau, 3)}${V.refName ? `, a level only 1 % of the uncoated reference's spectra (${V.refName}) exceed` : " (the level entered by hand)"}`;
  if (V.mode === "pf") {
    kids.push(el("p", { class: "lead" }, el("b", {}, `${fmt(V.x)} of the ${fmt(V.n)} particles judged (${pct(V.c, 1)}) have more D3 than the uncoated reference`),
      `: ${level}. ` + (V.e ? `${fmt(V.e)} more particles could not be judged (their fit sat on a bound); counting them as uncoated, at least ${pct(V.lower, 1)} of particles are coated (95 % confidence).`
        : `At least ${pct(V.lower, 1)} of particles are coated (95 % confidence).`)));
    kids.push(composition([
      { name: "coated", n: V.x, colour: COL.coated },
      { name: "not coated", n: V.n - V.x, colour: COL.uncoated },
      { name: "could not be judged", n: V.e, colour: COL.other },
      { name: "too weak to judge", n: V.pool - V.n, colour: COL.track, hatch: true }], "Particles by outcome"));
  } else {
    const cs = V.parts.map(r => r.c);
    kids.push(el("p", { class: "lead" }, el("b", {}, `${area(V.covered)} of the ${area(V.judged)} of particle area judged (${pct(V.areaShare, 1)}) has more D3 than the uncoated reference`),
      `: ${level}. By particle: ${cs.length > 1 ? `${pct(Math.min(...cs), 0)} to ${pct(Math.max(...cs), 0)}` : pct(cs[0], 0)}; every particle at least ${pct(V.lowest, 1)} (95 % confidence).`));
    kids.push(particleBars(V.parts.slice(0, 14)));
    kids.push(el("table", { class: "num" },
      el("tr", {}, ...["Particle", "Area", "Judged", "Coated", "Coated share", "At least (95 %)"].map(h => el("th", {}, h))),
      ...V.parts.map(r => el("tr", {}, el("td", {}, String(r.id)), el("td", {}, area(r.area)), el("td", {}, area(r.judged)), el("td", {}, area(r.covered)), el("td", {}, pct(r.c, 1)), el("td", {}, pct(r.bound, 1)))),
      el("tr", { class: "total" }, el("td", {}, "All"), el("td", {}, area(V.total)), el("td", {}, area(V.judged)), el("td", {}, area(V.covered)), el("td", {}, pct(V.areaShare, 1)), el("td", {}, ""))));
  }
  kids.push(el("p", { class: "note" }, `Spectra whose G is below ${fmt(V.gmin, 3)} counts are too weak to tell coated from uncoated and are not judged.` +
    (V.mode === "map" ? " Only pixels inside the particles count; the line on each bar shows its range, the tick its 95 % lower bound." : "")));
  return kids;
}

function calibrationSection(A) {
  const C = A.calib;
  if (!C || C.missing) return [el("div", { class: "callout" }, el("b", {}, "Calibrated D3/G not available. "), (C && C.missing) || "Open the Particles tab once to run it.")];
  return [el("p", { class: "lead" }, el("b", {}, `D3/G ${fx(C.median, C.dp)} (95 % CI ${fx(C.lo, C.dp)}–${fx(C.hi, C.dp)})`),
      ` over ${fmt(C.n)} carbon ${C.unit} of ${fmt(C.measured)}, with the band shapes of ${C.name} held so the value compares between samples.` +
      (C.cmp ? ` That is ${C.cmp.ratio.toFixed(2)} × the uncoated reference (${fx(C.refMed, C.dp)}), a difference of ${sgn(C.cmp.diff, C.dp)} (95 % CI ${fx(C.cmp.lo, C.dp)} to ${fx(C.cmp.hi, C.dp)}): ` +
        (C.cmp.lo > 0 ? "more amorphous carbon than the reference." : C.cmp.hi < 0 ? "less than the reference." : "not told apart from the reference.") : " No uncoated reference was chosen to compare with.")),
    el("table", { class: "num compact" }, el("tr", {}, ...["Band", "Position (cm⁻¹)", "FWHM (cm⁻¹)", "Shape", "From"].map((h, i) => el("th", { class: i > 2 ? "l" : "" }, h))),
      ...C.bands.map(b => el("tr", {}, el("td", {}, b.name), el("td", {}, b.pos.toFixed(1)), el("td", {}, b.fwhm.toFixed(1)), el("td", { class: "l" }, b.shape), el("td", { class: "l" }, b.from)))),
    el("p", { class: "note" }, `Median over spectra whose G stands at least 10 × their noise; the interval resamples whole blocks of neighbouring spectra. The fixed shapes leave a residual of ${C.rn.toFixed(2)} × the noise.`)];
}

// the analysis in a few plain questions
function statsSummary(A) {
  const rows = [], U = A.uni, V = A.cov, C = A.calib, checks = [];
  const row = (q, a, d) => rows.push(el("tr", {}, el("td", {}, q), el("td", {}, a), el("td", {}, d)));
  if (C && !C.missing) {
    row("How much amorphous carbon (D3/G)?", fx(C.median, C.dp), `median, 95 % CI ${fx(C.lo, C.dp)}–${fx(C.hi, C.dp)}`);
    if (C.cmp) row("More than the uncoated reference?", C.cmp.lo > 0 ? `Yes, ${C.cmp.ratio.toFixed(2)} ×` : C.cmp.hi < 0 ? "No, less" : "Not clearly",
      `${sgn(C.cmp.diff, C.dp)} (95 % CI ${fx(C.cmp.lo, C.dp)} to ${fx(C.cmp.hi, C.dp)})`);
    if (C.rn > 2) checks.push(`the fixed band shapes fit these spectra less closely (residual ${C.rn.toFixed(1)} × the noise)`);
  }
  if (V && V.mode === "pf") {
    row("How many particles are coated?", `${fmt(V.x)} of ${fmt(V.n)} (${pct(V.c, 1)})`, `at least ${pct(V.lower, 1)} (95 % confidence), compared with the uncoated reference`);
    if (V.e > 0.1 * (V.n + V.e)) checks.push(`${fmt(V.e)} particles could not be judged`);
  } else if (V && V.mode === "map") row("How much of the area is coated?", pct(V.areaShare, 1), `${area(V.covered)} of ${area(V.judged)}, compared with the uncoated reference`);
  else row(A.kind === "pf" ? "How many particles are coated?" : "How much of the area is coated?", "–", "choose an uncoated reference under Particles › Coverage");
  if (U && U.mode === "pf") {
    row("Is the coating the same on every particle?", U.head || "–", `particles differ by ${pct(U.cv)} (CV); ${pct(U.I2)} of the differences are real, the rest is noise`);
    if (U.thin) row("The thinnest particles", `5 % below ${fx(U.thin.value, 3)}`, V && V.tau ? (U.thin.value > V.tau ? `still above the uncoated level (${fx(V.tau, 3)})` : `at or below the uncoated level (${fx(V.tau, 3)})`) : "D3/G per particle, noise taken out");
    if (Number.isFinite(U.rhoSize)) row("Does it depend on particle size?", U.rhoSizeCI[0] > 0 ? "Yes: larger carry more" : U.rhoSizeCI[1] < 0 ? "Yes: larger carry less" : "No", `rank correlation ${fx(U.rhoSize, 2)}`);
    if (U.drift) checks.push("the signal drifted during the run (taken out before these numbers)");
    if (U.nC / U.n < 0.2) checks.push(`D3 is clearly seen in only ${pct(U.nC / U.n)} of particles`);
  } else if (U && U.mode === "map") {
    const [h1] = (U.head || "–").split("; ");
    row("Is the coating even within a particle?", h1, `places within a particle differ by ${pct(U.W.cv)} (CV)`);
    row("Do the particles differ from each other?", !U.B ? "Too few particles" : U.B.cv < U.W.cv ? "Less than places within one" : "More than places within one", U.B ? `by ${pct(U.B.cv)} (CV)` : "needs 3 or more");
    if (U.det < 0.2) checks.push(`D3 is clearly seen in only ${pct(U.det)} of pixels`);
  }
  row("Anything to check?", checks.length ? "Yes" : "No", checks.length ? checks.join("; ") : "nothing flagged");
  return [el("table", { class: "stats-sum" }, el("tr", {}, ...["Question", "Answer", "Details"].map(h => el("th", {}, h))), ...rows),
    el("p", { class: "note" }, "CV: how much the true values differ, as a share of their mean, with the measurement noise taken out. All intervals are 95 %. The Particles tab has every test behind these answers.")];
}

// --------------------------------------------------------------------------
export function buildReport(host, R) {
  const now = new Date(), c = R.cfg, A = R.analysis, pf = !!(A ? A.kind === "pf" : R.points);
  const step = Math.abs(R.x[1] - R.x[0]);
  host.replaceChildren();
  host.append(el("header", { class: "rp-head" },
    el("div", { class: "kicker" }, "Carbo Metrics · Raman carbon report"),
    el("h1", {}, R.title || "Raman carbon analysis"),
    el("div", { class: "sub" }, `${R.ds.name} · ${pf ? `${fmt(R.nx)} ParticleFinder particles` : `${R.nx} × ${R.ny} points (${fmt(R.nx * R.ny)}), step ${fmt(step, 3)} µm`}`),
    el("div", { class: "meta" }, `Fit ${c.range_lo}–${c.range_hi} cm⁻¹, baseline ${c.baseline_order < 0 ? "none" : "poly order " + c.baseline_order}${c.fit_separately ? " (separately)" : ""}, ratio ${c.ratio_num}/${c.ratio_den} · ${now.toLocaleString()}`),
    R.ds.path ? el("div", { class: "meta" }, R.ds.path) : null));

  let n = 1;
  if (A) {
    host.append(sec(n++, "At a glance", glance(A)));
    host.append(sec(n++, pf ? "Particles covered by the coating" : "Area covered by the coating", ...coverageSection(A)));
    host.append(sec(n++, "Calibrated D3/G", ...calibrationSection(A)));
    host.append(sec(n++, "Statistical analysis: summary", ...statsSummary(A)));
  }

  host.append(sec(n++, "Maps", el("div", { class: "figs four" }, ...R.figs.maps.map(m => img(m.url, m.title)))));
  host.append(sec(n++, "Spectrum, distribution and optical image", el("div", { class: "figs" },
    img(R.figs.cursor, `Cursor spectrum at point ${R.pinned != null ? R.pinned + 1 : "–"} with the fitted bands, baseline (dashed) and residual.`),
    img(R.figs.hist, "Distribution over valid points of the selected parameter."),
    img(R.figs.video, "White-light image with the map footprint (dashed) and the selected map overlaid."))));

  const qRows = [...R.quality.querySelectorAll(".qcell")].map(q => [q.querySelector(".k").textContent, q.querySelector(".v").textContent]);
  const qt = el("table", {}, el("tr", {}, el("th", {}, "Quality"), el("th", {}, "")), ...qRows.map(([k, v]) => el("tr", {}, el("td", {}, k), el("td", {}, v))));
  const st = el("table", {}, el("tr", {}, el("th", {}, "Parameter"), el("th", {}, "Mean"), el("th", {}, "St Dev"), el("th", {}, "Median"), el("th", {}, "IQR")),
    ...R.stats.map(r => el("tr", {}, el("td", {}, r.label), el("td", {}, fmt(r.st.mean)), el("td", {}, fmt(r.st.sd)), el("td", {}, fmt(r.st.median)), el("td", {}, `${fmt(r.st.q1, 3)}–${fmt(r.st.q3, 3)}`))));
  host.append(sec(n++, "Fit quality and statistics", el("div", { class: "two" }, qt, st)));

  const lim = (r, nm) => `${nm}: ${r.use_lo ? "≥ " + r.lo : ""}${r.use_lo && r.use_hi ? ", " : ""}${r.use_hi ? "≤ " + r.hi : ""}${!r.use_lo && !r.use_hi ? "off" : ""}`;
  host.append(sec(n++, "Fitting configuration", el("table", {},
    el("tr", {}, ...["Band", "Pos.", "Min", "Max", "Width", "Min", "Max", "Type", "Fitted"].map(h => el("th", {}, h))),
    ...c.peaks.map(p => el("tr", {}, el("td", {}, p.name), ...[p.pos, p.pos_min, p.pos_max, p.width, p.width_min, p.width_max].map(v => el("td", {}, v)), el("td", {}, p.shape), el("td", {}, p.fit ? "yes" : "held")))),
    el("p", { class: "note" }, `Valid-point limits: ${lim(c.intensity, "intensity of the stronger ratio band")}; ${lim(c.ratio_i, "ratio(I)")}; ${lim(c.ratio_a, "ratio(Ar)")}. Areas are integrals over the fit range. Std. error = √(SSR / (N − 3 × bands)); red. χ² uses Poisson variance.`)));

  if (R.pstats) {
    const ps = R.pstats;
    host.append(sec(n++, "Particles",
      el("p", { class: "note" }, `${ps.parts.length} particles cover ${(ps.cover * 100).toFixed(1)} % of the map. ${ps.r.name}: pixel level ${fmt(ps.pixel.mean)} ± ${fmt(ps.pixel.sd, 3)} (n = ${ps.pixel.n} pixels); particle level ${fmt(ps.particle.mean)} ± ${fmt(ps.particle.sd, 3)} (n = ${ps.particle.n} particles). Compare samples on particle means; neighbouring pixels are not independent.`),
      el("table", {}, el("tr", {}, ...["#", "Pixels", "Area µm²", "Median", "IQR"].map(h => el("th", {}, h))),
        ...ps.parts.slice(0, 12).map(p => el("tr", {}, el("td", {}, p.id), el("td", {}, p.n), el("td", {}, fmt(p.area, 3)), el("td", {}, fmt(p.st.median)), el("td", {}, `${fmt(p.st.q1, 3)}–${fmt(p.st.q3, 3)}`))))));
  }

  if (R.splat && R.splatImg) {
    const rp = R.splat.report || {};
    host.append(sec(n++, "Spectral Gaussian splat field",
      el("div", { class: "two" }, img(R.splatImg, "3-D Gaussian field over the white-light image; height = Raman shift."),
        el("p", { class: "note" }, `${R.splat.n} Gaussians fitted to the baseline-corrected cube on the particles (${(R.splat.mask_fraction * 100).toFixed(0)} % of the map). Held-out channels: error ÷ the map's own noise = ${fmt(rp.resid_over_noise, 3)} (1 = at the noise floor), PSNR ${fmt(rp.psnr_eval_dB, 3)} dB; ${fmt(rp.compression_ratio_full, 3)}× fewer numbers than the cube.`))));
  }
  host.append(el("footer", { class: "rp-foot" }, "Generated by Carbo Metrics. Made by Siddhartha Pahari. All rights reserved. Per-pixel ratios with a near-zero denominator are heavy-tailed; the median is reported beside the mean for that reason."));
}

// --------------------------------------------------------------------------
// the DLC coating report: the fit's results at a glance, the quality test against the limits,
// how even each parameter is, hydrogen, the maps and a spectrum, then the method
// --------------------------------------------------------------------------
export function buildDlcReport(host, A) {
  const { R, F, settings: s, params: P } = A, st = F.stats, Q = F.Q, now = new Date(), unit = A.pf ? "particles" : "spectra";
  const lab = k => P[k].label, un = k => (P[k].unit ? ` ${P[k].unit}` : "");
  const med = (k, d = 3) => (st[k].n ? fmt(st[k].median, d) : "–"), iqr = k => (st[k].n ? `IQR ${fmt(st[k].q1, 3)}–${fmt(st[k].q3, 3)}` : "none fitted");
  const p1 = v => pct(v, 1);
  host.replaceChildren();
  host.append(el("header", { class: "rp-head" },
    el("div", { class: "kicker" }, "Carbo Metrics · DLC coating report"),
    el("h1", {}, A.title || "DLC coating analysis"),
    el("div", { class: "sub" }, `${A.ds.name} · ${A.pf ? `${fmt(R.P)} ParticleFinder particles` : R.P === 1 ? "one spectrum" : `${R.nx} × ${R.ny} points (${fmt(R.P)})`}`),
    el("div", { class: "meta" }, `D (Lorentzian) + G (Breit-Wigner-Fano) over a straight background, ${R.range[0].toFixed(0)}–${R.range[1].toFixed(0)} cm⁻¹ · ${now.toLocaleString()}`),
    A.ds.path ? el("div", { class: "meta" }, A.ds.path) : null));
  let n = 1;

  // at a glance
  const qcLevel = !Q.use.length || !Q.n ? null : !Q.nFail ? "good" : Q.share >= 0.95 ? "warn" : "bad";
  const key3 = ["idig", "gpos", "gfwhm"].filter(k => F.hom[k]);
  const worst = key3.length ? key3.reduce((a, b) => (F.hom[b].I2 > F.hom[a].I2 ? b : a)) : null;
  const homLevel = worst ? (F.hom[worst].I2 < 0.25 ? "good" : F.hom[worst].I2 < 0.75 ? "warn" : "bad") : null;
  const left = F.measured - F.n;
  host.append(sec(n++, "At a glance", el("div", { class: "tiles" },
    tile("I(D)/I(G)", med("idig"), `median · ${iqr("idig")}`),
    tile("G position", `${med("gpos", 5)} cm⁻¹`, `median · ${iqr("gpos")}`),
    tile("FWHM(G)", `${med("gfwhm")} cm⁻¹`, `median · ${iqr("gfwhm")}`),
    A.graphite ? tile("Hydrogen", "–", "does not apply: graphite-like spectra")
      : tile("Hydrogen", st.h.n ? `${med("h")} at.%` : "–", st.h.n ? `median of ${fmt(st.h.n)} · ${iqr("h")}` : "not defined: the background does not rise"),
    A.graphite ? tile("sp² cluster size La", "–", "does not apply: graphite-like spectra")
      : tile("sp² cluster size La", `${med("la")} nm`, `median · from I(D)/I(G), C′ ${s.laC} Å⁻²`),
    tile("Quality test", !Q.use.length ? "–" : !Q.n ? "Nothing tested" : !Q.nFail ? "Pass" : qcLevel === "warn" ? "Mostly inside" : "Fail",
      Q.use.length ? (Q.n ? `${p1(Q.share)} of ${fmt(Q.n)} ${unit} inside the limits` : "no fitted spectrum to test") : "no limits set", qcLevel),
    tile("Homogeneity", !worst ? "–" : homLevel === "good" ? "Homogeneous" : homLevel === "warn" ? "Partly" : "Inhomogeneous",
      worst ? `${lab(worst)}: ${p1(F.hom[worst].I2)} of the spread is real` : "too few fitted spectra", homLevel),
    tile("Spectra fitted", `${fmt(F.n)} of ${fmt(F.measured)}`, left ? `${fmt(left)} left out (weak G, G on a bound or no fit)` : "every measured spectrum")),
    A.graphite ? el("div", { class: "callout" }, el("b", {}, "Not DLC. "), A.graphite) : null));

  // the quality test
  const lim = A.limits.filter(L => (L.use_lo && Number.isFinite(L.lo)) || (L.use_hi && Number.isFinite(L.hi)));
  const qcKids = [];
  if (!lim.length) qcKids.push(el("div", { class: "callout" }, el("b", {}, "No limits set. "), "Enter a minimum, a maximum or both under DLC › Quality test, then refresh this report."));
  else {
    qcKids.push(el("p", { class: "lead" }, el("b", {}, !Q.n ? "No fitted spectrum could be tested." : !Q.nFail ? `All ${fmt(Q.n)} ${unit} tested are inside the limits.`
      : `${fmt(Q.nFail)} of ${fmt(Q.n)} ${unit} tested are outside the limits.`), Q.n ? ` ${p1(Q.share)} pass (95 % CI ${p1(Q.ci[0])}–${p1(Q.ci[1])}).` : ""));
    qcKids.push(el("table", {}, el("tr", {}, ...["Parameter", "Minimum", "Maximum", "Outside"].map(h => el("th", {}, h))),
      ...lim.map(L => el("tr", {}, el("td", {}, lab(L.key) + un(L.key)), el("td", {}, L.use_lo && Number.isFinite(L.lo) ? fmt(L.lo, 5) : "–"),
        el("td", {}, L.use_hi && Number.isFinite(L.hi) ? fmt(L.hi, 5) : "–"), el("td", {}, fmt(Q.byKey[L.key]))))));
  }
  qcKids.push(el("table", { class: "num" }, el("tr", {}, ...["Parameter", "Median", "Mean", "SD", "Range", "Inside (95 % CI)", "Ppk"].map(h => el("th", {}, h))),
    ...A.qcKeys.map(k => {
      const t = st[k];
      if (!t.n) return el("tr", {}, el("td", {}, lab(k) + un(k)), el("td", { colspan: 6 }, "none fitted"));
      return el("tr", {}, el("td", {}, lab(k) + un(k)), el("td", {}, fmt(t.median, 4)), el("td", {}, fmt(t.mean, 4)), el("td", {}, fmt(t.sd, 3)),
        el("td", {}, `${fmt(t.min, 4)}–${fmt(t.max, 4)}`), el("td", {}, t.inSpec != null ? `${p1(t.inSpec)} (${p1(t.inSpecCI[0])}–${p1(t.inSpecCI[1])})` : "–"),
        el("td", {}, t.ppk != null ? fmt(t.ppk, 3) : "–"));
    })));
  qcKids.push(el("p", { class: "note" }, `Every fitted spectrum is tested against every limit set; spectra not fitted are not tested${Q.undef ? `, nor are ${fmt(Q.undef)} whose tested value is undefined (hydrogen where the background does not rise)` : ""}. Pass: all inside; check: at least 95 % inside; fail: fewer. Ppk = min(maximum − mean, mean − minimum) / (3 SD). Neighbouring map points share part of their signal, so the intervals are on the narrow side.`));
  host.append(sec(n++, "Quality test", ...qcKids));

  // homogeneity
  const tol = k => { const L = A.limits.find(q => q.key === k); return L && L.use_lo && L.use_hi && Number.isFinite(L.lo) && Number.isFinite(L.hi) ? L.hi - L.lo : NaN; };
  host.append(sec(n++, "Homogeneity",
    el("table", {}, el("tr", {}, ...["Parameter", "CV", "True SD (95 % CI)", "I²", "Moran's I", "Of the tolerance"].map(h => el("th", {}, h))),
      ...A.homKeys.map(k => {
        const H = F.hom[k];
        if (!H) return el("tr", {}, el("td", {}, lab(k) + un(k)), el("td", { colspan: 5 }, "too few"));
        return el("tr", {}, el("td", {}, lab(k) + un(k)), el("td", {}, p1(H.cv)), el("td", {}, `${fmt(H.tau, 3)} (${fmt(H.tauLo, 3)}–${fmt(H.tauHi, 3)})`),
          el("td", {}, p1(H.I2)), el("td", {}, fmt(H.moran, 2)), el("td", {}, Number.isFinite(tol(k)) ? p1(6 * H.tau / tol(k)) : "–"));
      })),
    el("p", { class: "note" }, "Each parameter is set against its own fit errors: if the coating were the same everywhere, the values would scatter by those errors alone. I² is the share of the spread beyond them (below 25 % homogeneous, above 75 % inhomogeneous; Higgins et al. 2003), the true SD that real spread with the errors taken out (Paule-Mandel), Moran's I how alike neighbouring points are (patches or a gradient near 1, point-to-point variation near 0). With many spectra I² is high for even a small real variation: read it with the true SD and its share of the tolerance (6 × true SD over maximum − minimum).")));

  // hydrogen
  const H = F.H, nH = H.inside + H.below + H.above;
  host.append(sec(n++, "Hydrogen content",
    A.graphite ? el("div", { class: "callout" }, el("b", {}, "Does not apply. "), A.graphite) : null,
    el("p", { class: "lead" }, st.h.n ? el("b", {}, `${med("h")} at.% (median of ${fmt(st.h.n)} ${unit}; ${iqr("h")})`) : el("b", {}, "Not defined: the background does not rise in any fitted spectrum."),
      st.h.n ? `, ${fmt(H.inside)} of ${fmt(nH)} inside the calibration's ${s.hRange[0]}–${s.hRange[1]} at.% (${fmt(H.below)} below, ${fmt(H.above)} above); ${fmt(H.undef)} fitted ${unit} have a falling or flat background.` : ""),
    el("p", { class: "note" }, `H [at.%] = ${s.hA} + ${s.hB} log₁₀(m/I(G) [µm]), m the slope of the straight background and I(G) the G height (Casiraghi et al. 2005, at 514.5 nm). It reads the photoluminescence hydrogen brings to a-C:H, so it holds for hydrogenated DLC with about ${s.hRange[0]} to ${s.hRange[1]} at.% H; hydrogen-free DLC (a-C, ta-C) has none to read, and a fluorescent substrate or contamination raises m too. Median m/I(G) ${med("mig")} µm.`)));

  // maps and a spectrum
  host.append(sec(n++, "Maps", el("div", { class: "figs four" }, ...A.figs.maps.map(m => img(m.url, m.title)))));
  host.append(sec(n++, "Spectrum and distribution", el("div", { class: "figs" },
    img(A.figs.spectrum, A.pinned != null ? `${A.pf ? "Particle" : "Point"} ${A.pinned + 1}: the spectrum with the fitted D, G and background (dashed), and the residual.`
      : `The mean of the ${fmt(R.mean.n)} measured spectra with the fitted D, G and background (dashed), and the residual. Every spectrum's fit starts from these band shapes.`),
    A.figs.hist ? img(A.figs.hist, `${A.figs.histLabel} over the fitted ${unit}; limits in red.`) : null)));

  // method
  host.append(sec(n++, "Method",
    el("table", {}, el("tr", {}, ...["Band", "Line shape", "Position (cm⁻¹)", "Width (cm⁻¹)"].map(h => el("th", {}, h))),
      el("tr", {}, el("td", {}, "D"), el("td", {}, "Lorentzian"), el("td", {}, `${R.settings.D.pos_min}–${R.settings.D.pos_max}`), el("td", {}, `${R.settings.D.width_min}–${R.settings.D.width_max} (FWHM)`)),
      el("tr", {}, el("td", {}, "G"), el("td", {}, `Breit-Wigner-Fano, 1/Q ${R.settings.q[0]} to ${R.settings.q[1]}`), el("td", {}, `${R.settings.G.pos_min}–${R.settings.G.pos_max} (maximum)`),
        el("td", {}, `${R.settings.G.width_min}–${R.settings.G.width_max} (Γ)`))),
    el("p", { class: "note" }, `Each spectrum is cut to ${R.range[0].toFixed(0)}–${R.range[1].toFixed(0)} cm⁻¹ and cleared of cosmic rays. The mean spectrum is fitted first; every spectrum then starts from its band shapes, with its background and heights solved by least squares, and D, G and a straight background are fitted together by bounded Levenberg-Marquardt. ` +
      `G: I(ω) = I₀ (1 + 2(ω − ω₀)/(QΓ))² / (1 + (2(ω − ω₀)/Γ)²); its position is the maximum, ω₀ + Γ/(2Q), and FWHM(G) = Γ (1 + 1/Q²) / (1 − 1/Q²). I(D)/I(G) is the ratio of the band heights, A(D)/A(G) of their areas over the fit range. ` +
      `A spectrum counts as fitted when its G height is at least ${s.snrMin} × its noise and G's position and width are inside their bounds. Standard errors come from each fit's covariance at its residual. ` +
      `La = √(I(D)/I(G) / C′), C′(514 nm) = ${s.laC} Å⁻², for amorphous carbon with clusters below about ${s.laMax} nm (approximate at 532 nm). The sp³ fraction and hardness are not given: at one laser line they need a calibration.`),
    el("p", { class: "note" }, "References: A. C. Ferrari and J. Robertson, Interpretation of Raman spectra of disordered and amorphous carbon, Phys. Rev. B 61, 14095 (2000). C. Casiraghi, A. C. Ferrari and J. Robertson, Raman spectroscopy of hydrogenated amorphous carbons, Phys. Rev. B 72, 085401 (2005). J. P. T. Higgins, S. G. Thompson, J. J. Deeks and D. G. Altman, Measuring inconsistency in meta-analyses, BMJ 327, 557 (2003).")));
  host.append(el("footer", { class: "rp-foot" }, "Generated by Carbo Metrics. Made by Siddhartha Pahari. All rights reserved."));
}

// --------------------------------------------------------------------------
// Layers report: the layers as a table (number, from, to, thickness, material, comment), the
// thickness limits, the figures, the checks to read before trusting them, and the method
// --------------------------------------------------------------------------
const LAYER_COLOURS = ["#0a86b8", "#d9700b", "#7346d1", "#a8820a", "#d02f68", "#16935a", "#3a465c", "#2f62d0"];
export function buildLayersReport(host, A) {
  const { R, L, Q, settings: s } = A, now = new Date(), p1 = v => pct(v, 1);
  const um = v => (Number.isFinite(v) ? (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3) : "–");      // µm to the nanometre
  const colour = a => (a < 0 ? COL.faint : LAYER_COLOURS[a % LAYER_COLOURS.length]);
  const sw = a => el("span", { style: `display:inline-block;width:3mm;height:3mm;border-radius:0.6mm;margin-right:1.5mm;vertical-align:-0.4mm;background:${colour(a)}` });
  const open = L.layers.some(x => x.open), mats = L.layers.filter(x => x.mat >= 0);
  const thinnest = mats.reduce((a, x) => (!a || x.t.mean < a.t.mean ? x : a), null);
  const res = Float64Array.from(R.comps.resid).sort(), resMed = res.length ? res[Math.floor(res.length / 2)] : NaN;
  host.replaceChildren();
  host.append(el("header", { class: "rp-head" },
    el("div", { class: "kicker" }, "Carbo Metrics · Layers report"),
    el("h1", {}, A.title || "Multilayer analysis"),
    el("div", { class: "sub" }, `${A.ds.name} · ${A.line ? `a line of ${fmt(R.prep.P)} points` : `${R.geom.nx} × ${R.geom.ny} points`} · ${s.kind === "depth" ? "a confocal depth profile" : "across a cut edge"}`),
    el("div", { class: "meta" }, `${R.comps.k} material${R.comps.k > 1 ? "s" : ""} ${R.comps.mode === "refs" ? "from reference spectra" : "found in the spectra"}, ${R.prep.range[0].toFixed(0)}–${R.prep.range[1].toFixed(0)} cm⁻¹ · ${now.toLocaleString()}`),
    A.ds.path ? el("div", { class: "meta" }, A.ds.path) : null));
  let n = 1;

  // at a glance
  const limLevel = Q.set ? Q.level : null;
  host.append(sec(n++, "At a glance", el("div", { class: "tiles" },
    tile("Layers", String(L.layers.length), L.layers.map(x => A.name(x.mat)).join(" / ") || "none found"),
    tile("Total thickness", L.layers.length ? `${open ? "≥ " : ""}${um(L.total.mean)} µm` : "–", A.line ? "along the scan" : `mean of ${fmt(L.nCons)} lines${L.total.n > 1 ? `, SD ${fmt(L.total.sd, 2)} µm` : ""}`),
    tile("Thinnest layer", thinnest ? `${um(thinnest.t.mean)} µm` : "–", thinnest ? `layer ${thinnest.i + 1}, ${A.name(thinnest.mat)}${thinnest.thin ? ", below the resolution" : ""}` : "", thinnest && thinnest.thin ? "warn" : null),
    tile("Thickness limits", !Q.set ? "–" : limLevel === "good" ? "Pass" : limLevel === "warn" ? "Mostly inside" : limLevel === "bad" ? "Fail" : "Not tested",
      Q.set ? `${Q.set} layer${Q.set > 1 ? "s" : ""} with limits` : "no limits set", limLevel),
    A.line ? tile("Points", fmt(L.m), `${fmt(L.step, 3)} µm apart`)
      : tile("Lines sharing the stack", `${fmt(L.nCons)} of ${fmt(L.nLines)}`, L.tilt ? `layers tilted ${fmt(L.tilt.deg, 2)}° in the map${s.tilt ? ", taken out" : ", not taken out"}` : "tilt not measured", L.nCons < 0.8 * L.nLines ? "warn" : "good"),
    tile("Resolution", Number.isFinite(L.resolution) ? `${fmt(L.resolution, 3)} µm` : "–", "16–84 % width of the steps between layers", thinnest && thinnest.thin ? "warn" : null),
    tile("Materials", String(R.comps.k), R.comps.mode === "refs" ? "unmixed against references" : "found in the spectra"),
    tile("Residual", `${fmt(resMed, 3)} ×`, "the noise, median over the spectra: about 1 when the materials explain them", resMed > 2 ? "warn" : "good"))));

  // the table
  const mark = { good: "✓ pass", warn: "! check", bad: "✕ fail" };
  const head = ["#", "From", "To", "Thickness", ...(A.line ? [] : ["SD", "Range"]), "Material", "Steps", "Limits", "Comment"];
  host.append(sec(n++, "Layers",
    el("table", { class: "num" }, el("tr", {}, ...head.map(h => el("th", {}, h))),
      ...L.layers.map((x, i) => {
        const q = Q.rows[i];
        return el("tr", {}, el("td", {}, String(i + 1)), el("td", {}, um(x.from)), el("td", {}, um(x.to)),
          el("td", {}, `${x.open ? "≥ " : ""}${um(x.t.mean)}${x.thin ? " ⚠" : ""}`), ...(A.line ? [] : [el("td", {}, x.t.n > 1 ? um(x.t.sd) : "–"), el("td", {}, x.t.n > 1 ? `${um(x.t.min)}–${um(x.t.max)}` : "–")]),
          el("td", {}, sw(x.mat), A.name(x.mat)), el("td", {}, `${um(x.wFrom)} / ${um(x.wTo)}`),
          el("td", {}, q ? (q.level ? mark[q.level] : "untested") : "–"), el("td", {}, A.comments[i] || ""));
      }),
      el("tr", {}, el("td", {}, ""), el("td", { colspan: 2 }, "Total"), el("td", {}, L.layers.length ? `${open ? "≥ " : ""}${um(L.total.mean)}` : "–"),
        ...(A.line ? [] : [el("td", {}, L.total.n > 1 ? um(L.total.sd) : "–"), el("td", {}, L.total.n > 1 ? `${um(L.total.min)}–${um(L.total.max)}` : "–")]), el("td", { colspan: 4 }, ""))),
    el("p", { class: "note" }, `In µm along the ${s.kind === "depth" ? `depth (the stage's travel × ${fmt(s.depthScale, 3)})` : "scan"}. From and To: ${A.line ? "" : "the median over the lines of "}the boundaries; thickness ${A.line ? "" : "the mean over the lines sharing the stack, "}${!A.line && L.tilt && s.tilt ? `with the layers' tilt of ${fmt(L.tilt.deg, 2)}° taken out (so it can be less than To − From), ` : ""}` +
      "steps the 16–84 % widths of the boundaries at From / To. ≥: the layer runs past the scan's edge, so its thickness is a lower bound. ⚠: thinner than about 1.65 × the resolution, so it never reaches its pure level and reads too thick.")));

  // limits
  if (Q.set) {
    host.append(sec(n++, "Thickness limits",
      el("table", {}, el("tr", {}, ...["Layer", "Minimum (µm)", "Maximum (µm)", A.line ? "Thickness (µm)" : "Lines inside (95 % CI)", "Verdict"].map(h => el("th", {}, h))),
        ...Q.rows.map((q, i) => (q ? el("tr", {}, el("td", {}, sw(L.layers[i].mat), `${i + 1} · ${A.name(L.layers[i].mat)}`), el("td", {}, q.lo != null ? fmt(q.lo, 4) : "–"), el("td", {}, q.hi != null ? fmt(q.hi, 4) : "–"),
          el("td", {}, A.line ? um(L.layers[i].t.mean) : q.n ? `${fmt(q.pass)} of ${fmt(q.n)} (${p1(q.ci[0])}–${p1(q.ci[1])})` : "–"),
          el("td", {}, q.level ? mark[q.level] : `untested${q.undet ? ": runs past the scan's edge" : ""}`)) : null)).filter(Boolean)),
      el("p", { class: "note" }, `${A.line ? "" : "Every line sharing the stack is tested: pass when all are inside, check when at least 95 % are, fail below that. "}A layer that runs past the scan's edge has a lower bound for a thickness: it passes a minimum when that bound is above it, and cannot be tested against a maximum.`)));
  }

  // figures
  if (A.figs.maps.length) host.append(sec(n++, "Maps", el("div", { class: "figs" }, ...A.figs.maps.map(m => img(m.url, m.title)))));
  host.append(sec(n++, A.line ? "Profile and materials" : "Across the layers, and the materials",
    A.figs.profile ? img(A.figs.profile, `Each material's amount, scaled to its pure level (1), across the layers${A.line ? "" : ", averaged over the lines sharing the stack"}; the layers shaded and numbered, the boundaries dashed, one half dotted.`) : null,
    A.figs.spectra ? img(A.figs.spectra, R.comps.mode === "refs" ? "The reference spectra the spectra were unmixed against, each scaled to 1 and offset." : "The materials' spectra found in the data, each scaled to 1 and offset; a matched reference dashed.") : null));

  // checks
  host.append(sec(n++, "Checks", el("ul", { class: "note" }, ...A.checks.map(c => el("li", {}, c.warn ? el("b", {}, "Check: ") : null, c.text)))));

  // method
  const kStep = R.comps.steps && R.comps.steps[R.comps.steps.length - 1];
  host.append(sec(n++, "Method",
    el("p", { class: "note" }, `Each spectrum is cut to ${R.prep.range[0].toFixed(0)}–${R.prep.range[1].toFixed(0)} cm⁻¹, cleared of cosmic rays against its neighbouring spectra (a channel more than 8 × its noise above the same channel in every neighbour, over at most two channels), binned to ${R.prep.nb} channels of ${R.prep.w} and cleared of ${R.prep.degree > 0 ? `a degree-${R.prep.degree} polynomial background by modified polyfit, raised to the middle of the noise` : "no background (none taken off)"}. ` +
      (R.comps.mode === "refs" ? `Every spectrum is unmixed against the ${R.comps.k} reference spectra (${A.refs.join(", ")}), treated the same way, by non-negative least squares. `
        : `The materials are the purest spectra, picked one after another by successive projections on what each spectrum has left over its noise (each pick averaged with the spectra most like it), ${R.settings.k ? `${R.comps.k} as set` : `until no more than a handful of spectra have more than ${R.settings.stop} × their noise left`}${kStep ? ` (${fmt(kStep.over)} of ${fmt(R.prep.n)} with ${R.comps.k})` : ""}; they are refined by non-negative matrix factorisation (hierarchical alternating least squares) and every spectrum is unmixed against them by non-negative least squares. ${A.refs.length ? `Each takes the name of the reference it matches best at HQI ${s.minHQI} or above (100 r²). ` : ""}`) +
      `Each material's amount is scaled to its plateau, the median over the lines of its highest amount (smoothed over three points), so 1 is the pure material. Along every line across the layers each point goes to the material with the largest scaled amount, or to none below one half, and a run shorter than ${s.minPts} point${s.minPts > 1 ? "s" : ""} inside the stack goes to its neighbour. ` +
      "A boundary is where the two materials' scaled amounts cross: for a step blurred by the laser spot both are one half there, whatever their Raman cross-sections. An outer edge is where the layer's amount falls to one half. The width of a step is the distance over which it goes from 84 % to 16 %, the 3-point smoothing taken out; a layer thinner than about 1.65 × the median width never reaches 95 % of its pure level. " +
      (A.line ? "" : `On a map, the stack most lines share is the sample's; its thicknesses are the mean over those lines, and the boundaries' drift along the lines (one slope for all, by least squares) is the layers' tilt in the map${s.tilt ? ", taken out of every thickness by its cosine" : ""}. `) +
      (s.kind === "depth" ? `Depths are the stage's travel × ${fmt(s.depthScale, 3)}: focused into a film with a dry objective, the focus goes deeper than the stage moves, and the depth resolution worsens with depth (Everall 2000).` : "")),
    el("p", { class: "note" }, "References: C. A. Lieber and A. Mahadevan-Jansen, Automated method for subtraction of fluorescence from biological Raman spectra, Appl. Spectrosc. 57, 1363 (2003). J. Zhao, H. Lui, D. I. McLean and H. Zeng, Automated autofluorescence background subtraction algorithm for biomedical Raman spectroscopy, Appl. Spectrosc. 61, 1225 (2007). N. Gillis and S. A. Vavasis, Fast and robust recursive algorithms for separable nonnegative matrix factorization, IEEE Trans. Pattern Anal. Mach. Intell. 36, 698 (2014). A. Cichocki and A.-H. Phan, Fast local algorithms for large scale nonnegative matrix and tensor factorizations, IEICE Trans. Fundam. E92-A, 708 (2009). N. J. Everall, Confocal Raman microscopy: why the depth resolution and spatial accuracy can be much worse than you think, Appl. Spectrosc. 54, 1515 (2000).")));
  host.append(el("footer", { class: "rp-foot" }, "Generated by Carbo Metrics. Made by Siddhartha Pahari. All rights reserved."));
}

// --------------------------------------------------------------------------
// MVA report: the method and its components, the images, the loadings and a diagnostic, the
// comparison with other methods, the checks and the method
// --------------------------------------------------------------------------
export function buildMvaReport(host, A) {
  const { R, prep, compare } = A, now = new Date(), p1 = v => pct(v, 1), clusters = !!R.labels, unit = R.method === "pca" ? "component" : clusters ? "class" : "component";
  const sw = a => el("span", { style: `display:inline-block;width:3mm;height:3mm;border-radius:0.6mm;margin-right:1.5mm;vertical-align:-0.4mm;background:${LAYER_COLOURS[a % LAYER_COLOURS.length]}` });
  const res = Float64Array.from(R.resid).sort(), resMed = res[Math.floor(res.length / 2)];
  host.replaceChildren();
  host.append(el("header", { class: "rp-head" },
    el("div", { class: "kicker" }, "Carbo Metrics · MVA report"),
    el("h1", {}, A.title || "Multivariate analysis"),
    el("div", { class: "sub" }, `${A.ds.name} · ${A.method} · ${fmt(R.n)} ${A.pf ? "particles" : "spectra"}`),
    el("div", { class: "meta" }, `${prep.range[0].toFixed(0)}–${prep.range[1].toFixed(0)} cm⁻¹ in ${R.nb} channels, ${R.settings.norm === "none" ? "unscaled" : `each spectrum scaled to unit ${R.settings.norm === "area" ? "area" : "length"}`} · ${now.toLocaleString()}`),
    A.ds.path ? el("div", { class: "meta" }, A.ds.path) : null));
  let n = 1;

  // at a glance
  const sil = clusters ? (R.sil.find(x => x.k === R.k) || {}).s : NaN, best = compare.length ? compare.reduce((a, c) => (c.mean > a.mean ? c : a)) : null;
  const named = R.match ? R.match.filter(m => m.best >= 0).length : null;
  host.append(sec(n++, "At a glance", el("div", { class: "tiles" },
    tile("Method", A.method, { pca: "principal components", mcr: "non-negative components", kmeans: "k-means classes", hca: "Ward's classes", refs: "CLS against references" }[R.method]),
    tile(clusters ? "Classes" : "Components", String(R.k), R.settings.k ? "as set" : R.method === "refs" ? "one per reference" : clusters ? "highest mean silhouette" : "what the noise leaves"),
    R.method === "pca" ? tile("Variance kept", p1(R.pca.eig.slice(0, R.k).reduce((t, v) => t + v, 0) / R.pca.total), `by ${R.k} of the components; ${R.pca.suggested} suggested`)
      : clusters ? tile("Silhouette", fmt(sil, 2), sil >= 0.5 ? "the classes are distinct" : sil >= 0.25 ? "the classes overlap" : "no real structure", sil >= 0.5 ? "good" : sil >= 0.25 ? "warn" : "bad")
      : tile("Explained", p1(res.filter(v => v <= 2).length / res.length), `spectra left within 2 × their noise by the ${R.k} ${unit}s`),
    tile("Residual", `${fmt(resMed, 3)} ×`, "the noise, median over the spectra: about 1 when the fit explains them", resMed > 2 ? "warn" : "good"),
    tile("Named by reference", named == null ? "–" : `${named} of ${R.k}`, named == null ? (R.method === "pca" ? "PCA loadings are not spectra" : R.refNames ? "the references themselves" : "no references loaded") : `HQI ${A.minHQI} or above`),
    tile("Agreement", best ? fmt(best.mean, 3) : "–", best ? `mean |r| with ${best.method}` : "no other method run on this file"),
    tile("Spectra", fmt(R.n), A.pf ? "one per particle" : "measured points"),
    tile("Time", `${R.seconds.toFixed(1)} s`, "for this run"))));

  // components
  host.append(sec(n++, clusters ? "Classes" : "Components",
    el("table", {}, el("tr", {}, ...["#", "Name", R.method === "pca" ? "Explains" : clusters ? "Size" : "Share", "Best reference (HQI)"].map(h => el("th", {}, h))),
      ...Array.from({ length: R.k }, (_, a) => el("tr", {}, el("td", {}, String(a + 1)), el("td", {}, sw(a), A.name(a)), el("td", {}, A.detail(a)), el("td", {}, A.ref(a))))),
    el("p", { class: "note" }, R.method === "pca" ? "Each loading is a direction of change about the mean spectrum; its image is each spectrum's score along it." : "Each image is the component's share of the fitted spectrum (CLS, non-negative least squares).")));

  // figures
  host.append(sec(n++, "Images", el("div", { class: "figs four" }, ...A.figs.maps.map(m => img(m.url, m.title)))));
  host.append(sec(n++, "Loadings and diagnostics",
    A.figs.load ? img(A.figs.load, R.kind === "signed" ? "The loadings, each scaled to 1 and offset (dotted: zero)." : "The loadings, each scaled to 1 and offset; a matched reference dashed.") : null,
    A.figs.diag ? img(A.figs.diag, A.figs.diagName) : null));

  // comparison
  if (compare.length) {
    host.append(sec(n++, "Comparison with other methods",
      el("table", {}, el("tr", {}, ...["Method", "Mean |r|", "Pairs (|r|)"].map(h => el("th", {}, h))),
        ...compare.map(c => el("tr", {}, el("td", {}, c.method), el("td", {}, fmt(c.mean, 3)), el("td", {}, c.pairs.map(p => `${A.name(p.a)} ↔ ${c.name(p.b)} ${fmt(Math.abs(p.r), 2)}`).join(" · "))))),
      el("p", { class: "note" }, "Each loading paired with the most alike of the other method's (correlation over the channels, sign ignored), the closest pairs first. PCA's loadings are directions about the mean, not spectra, so they agree with the others only loosely.")));
  }

  // checks and method
  host.append(sec(n++, "Checks", el("ul", { class: "note" }, ...A.checks.map(c => el("li", {}, c.warn ? el("b", {}, "Check: ") : null, c.text)))));
  host.append(sec(n++, "Method",
    el("p", { class: "note" }, `Each spectrum is cut to the range, cleared of cosmic rays against its neighbouring spectra, binned to ${R.nb} channels and cleared of ${prep.degree > 0 ? `a degree-${prep.degree} polynomial background (modified polyfit, raised to the middle of the noise)` : "no background"}; ${R.settings.norm === "none" ? "it is not scaled" : `it is then scaled to unit ${R.settings.norm === "area" ? "area" : "length"} (a spectrum with less than 4 × its noise is scaled as if it had that much)`}. ` +
      { pca: "PCA: eigenvectors of the covariance of the mean-centred spectra (from up to 3 000 of them), each signed so its largest value is positive; the number suggested is the smallest that leaves no more than a handful of spectra with over twice their noise unexplained. The images are the scores.",
        mcr: "MCR: the purest spectra by successive projections, until no more than a handful of spectra have over twice their noise left, refined by non-negative matrix factorisation (hierarchical alternating least squares). Images by CLS: non-negative least squares on the components.",
        kmeans: "K-means on the first principal components (k-means++ starts, the best of three); k the highest mean silhouette of 2 to 8 unless set. The loadings are the classes' mean spectra; images by CLS on them.",
        hca: "HCA: Ward's linkage by the nearest-neighbour chain on the first principal components (on up to 1 000 evenly spread spectra, the others going to the nearest class); k the highest mean silhouette of 2 to 8 unless set. The loadings are the classes' mean spectra; images by CLS on them.",
        refs: `CLS: every spectrum fitted by non-negative least squares to the reference spectra (${A.refs.join(", ")}), treated like the spectra.` }[R.method] +
      " A loading is named after the reference it matches best at HQI = 100 r² of the minimum or above."),
    el("p", { class: "note" }, "References: I. T. Jolliffe and J. Cadima, Principal component analysis: a review and recent developments, Phil. Trans. R. Soc. A 374, 20150202 (2016). N. Gillis and S. A. Vavasis, Fast and robust recursive algorithms for separable nonnegative matrix factorization, IEEE Trans. Pattern Anal. Mach. Intell. 36, 698 (2014). A. Cichocki and A.-H. Phan, Fast local algorithms for large scale nonnegative matrix and tensor factorizations, IEICE Trans. Fundam. E92-A, 708 (2009). D. Arthur and S. Vassilvitskii, k-means++: the advantages of careful seeding, Proc. 18th ACM-SIAM Symposium on Discrete Algorithms, 1027 (2007). J. H. Ward, Hierarchical grouping to optimize an objective function, J. Am. Stat. Assoc. 58, 236 (1963). F. Murtagh, A survey of recent advances in hierarchical clustering algorithms, Comput. J. 26, 354 (1983). P. J. Rousseeuw, Silhouettes: a graphical aid to the interpretation and validation of cluster analysis, J. Comput. Appl. Math. 20, 53 (1987). C. A. Lieber and A. Mahadevan-Jansen, Appl. Spectrosc. 57, 1363 (2003). J. Zhao, H. Lui, D. I. McLean and H. Zeng, Appl. Spectrosc. 61, 1225 (2007).")));
  host.append(el("footer", { class: "rp-foot" }, "Generated by Carbo Metrics. Made by Siddhartha Pahari. All rights reserved."));
}
