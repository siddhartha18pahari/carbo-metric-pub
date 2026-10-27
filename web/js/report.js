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
