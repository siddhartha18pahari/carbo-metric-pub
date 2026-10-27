// Printable report from the current state (Print -> Save as PDF): the results at a glance, the
// coating's coverage, the calibrated D3/G and a summary of every statistical test, then the
// maps, spectra, quality and configuration. Charts are inline SVG so they print sharply.

import { el, fmt } from "./util.js";

const img = (src, cap) => el("figure", {}, el("img", { src }), cap ? el("figcaption", {}, cap) : null);

// chart colours for white paper: the app's light-theme set (checked for lightness, chroma,
// colour-blind separation and contrast on white), status colours with an icon and a word
const COL = { ink: "#142033", muted: "#4b5d7a", faint: "#8a98b0", grid: "#dfe6f1", track: "#e8eef7", paper: "#ffffff",
              coated: "#0a86b8", uncoated: "#d9700b", other: "#7346d1", good: "#13834f", warn: "#a96500", bad: "#c9334a" };
const STATUS = { good: ["✓", "good"], warn: ["!", "check"], bad: ["✕", "not met"], info: ["i", "note"] };

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
const pv = p => (!Number.isFinite(p) ? "–" : p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(3)}`);
const area = v => (Number.isFinite(v) ? `${fmt(Math.round(v))} µm²` : "–");
const pill = level => { const [g, w] = STATUS[level] || STATUS.info; return el("span", { class: `pill ${level || "info"}` }, el("b", {}, g), ` ${w}`); };
// a tile's value: a number large, a verdict in words smaller so it fits the tile
const tile = (k, v, s, level, foot) => el("div", { class: `tile${level ? ` ${level}` : ""}` }, el("div", { class: "k" }, k, level ? pill(level) : null),
  el("div", { class: /[a-z]{3}/i.test(v) ? "v txt" : "v" }, v), el("div", { class: "s" }, s), foot ? el("div", { class: "f" }, foot) : null);
const sec = (n, title, ...kids) => el("section", { class: "rp-sec" }, el("h2", {}, el("span", { class: "num" }, String(n)), title), ...kids);

// round ticks for an axis from 0, the last at or beyond max
function ticks(max, n = 4) {
  const raw = max / n, mag = 10 ** Math.floor(Math.log10(raw || 1)), step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || raw;
  const out = [0]; while (out[out.length - 1] < max - 1e-12) out.push(+(out.length * step).toPrecision(12));
  return out;
}

// --------------------------------------------------------------------------
// charts
// --------------------------------------------------------------------------
// a share 0..1 on a track, the measured value filled, its 95 % lower bound and a target marked
function meter({ value, lower, target, targetLabel, label }) {
  const W = 600, L = 8, R = W - 8, y = 26, h = 16, X = v => L + Math.max(0, Math.min(1, v)) * (R - L);
  const g = sv("svg", { viewBox: `0 0 ${W} 78`, class: "chart", role: "img", "aria-label": label });
  g.append(sv("rect", { x: L, y, width: R - L, height: h, rx: 4, fill: COL.track }));
  g.append(sv("rect", { x: L, y, width: Math.max(0, X(value) - L), height: h, rx: 4, fill: COL.coated }, sv("title", {}, `${label}: ${pct(value, 1)}`)));
  for (const t of [0, 0.25, 0.5, 0.75, 1]) g.append(sv("text", { x: X(t), y: y + h + 14, "text-anchor": t === 0 ? "start" : t === 1 ? "end" : "middle", class: "tk" }, pct(t, 0)));
  if (Number.isFinite(lower)) {
    g.append(sv("line", { x1: X(lower), x2: X(lower), y1: y - 6, y2: y + h + 2, stroke: COL.ink, "stroke-width": 2 }));
    g.append(sv("text", { x: X(lower), y: y - 9, "text-anchor": lower > 0.85 ? "end" : "middle", class: "lb" }, `95 % lower bound ${pct(lower, 1)}`));
  }
  if (Number.isFinite(target)) {
    g.append(sv("line", { x1: X(target), x2: X(target), y1: y - 2, y2: y + h + 6, stroke: COL.bad, "stroke-width": 1.5, "stroke-dasharray": "3 2" }));
    g.append(sv("text", { x: X(target), y: y + h + 28, "text-anchor": target > 0.85 ? "end" : "middle", class: "lb warn" }, targetLabel));
  }
  return g;
}

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

// medians with their 95 % intervals on one axis: the uncoated reference and this sample
function dotWhisker(rows, label) {
  const W = 600, L = 170, R = W - 20, rowH = 30, max = Math.max(...rows.map(r => (Number.isFinite(r.hi) ? r.hi : r.v))) * 1.15 || 1;
  const tk = ticks(max), top = tk[tk.length - 1], X = v => L + (v / top) * (R - L), H = rows.length * rowH + 26;
  const g = sv("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": label });
  for (const t of tk) {
    g.append(sv("line", { x1: X(t), x2: X(t), y1: 0, y2: rows.length * rowH, stroke: COL.grid, "stroke-width": 1 }));
    g.append(sv("text", { x: X(t), y: H - 8, "text-anchor": "middle", class: "tk" }, String(t)));
  }
  rows.forEach((r, i) => {
    const y = i * rowH + rowH / 2;
    g.append(sv("text", { x: L - 10, y: y + 4, "text-anchor": "end", class: "lbl" }, r.name));
    if (Number.isFinite(r.lo) && Number.isFinite(r.hi)) g.append(sv("line", { x1: X(r.lo), x2: X(r.hi), y1: y, y2: y, stroke: r.colour, "stroke-width": 2, "stroke-linecap": "round" }));
    g.append(sv("circle", { cx: X(r.v), cy: y, r: 5, fill: r.colour, stroke: COL.paper, "stroke-width": 2 }, sv("title", {}, `${r.name}: ${r.text}`)));
    g.append(sv("text", { x: Math.min(X(Number.isFinite(r.hi) ? r.hi : r.v) + 8, R - 2), y: y - 8, "text-anchor": X(Number.isFinite(r.hi) ? r.hi : r.v) + 8 > R - 60 ? "end" : "start", class: "val" }, r.text));
  });
  return g;
}

// the sample's place between the uncoated reference (0) and glassy carbon (1)
function scaleGauge(index, lo, hi, top) {
  const W = 600, L = 20, R = W - 20, a = Math.min(0, lo, index), b = Math.max(1, hi, index), X = v => L + ((v - a) / (b - a)) * (R - L), y = 30, h = 12;
  const g = sv("svg", { viewBox: `0 0 ${W} 92`, class: "chart", role: "img", "aria-label": "Place between the uncoated reference and glassy carbon" });
  g.append(sv("rect", { x: X(0), y, width: X(1) - X(0), height: h, rx: 4, fill: COL.track }));
  g.append(sv("rect", { x: X(Math.min(0, index)), y, width: Math.abs(X(index) - X(0)), height: h, rx: 4, fill: COL.coated }));
  if (Number.isFinite(lo) && Number.isFinite(hi)) g.append(sv("line", { x1: X(lo), x2: X(hi), y1: y + h / 2, y2: y + h / 2, stroke: COL.ink, "stroke-width": 2 }));
  g.append(sv("circle", { cx: X(index), cy: y + h / 2, r: 6, fill: COL.coated, stroke: COL.paper, "stroke-width": 2 }, sv("title", {}, `${index.toFixed(3)} (95 % CI ${lo.toFixed(3)} to ${hi.toFixed(3)})`)));
  g.append(sv("text", { x: X(index), y: y - 8, "text-anchor": X(index) < L + 60 ? "start" : X(index) > R - 60 ? "end" : "middle", class: "val" }, `this sample ${index.toFixed(3)}`));
  for (const t of [0, 0.25, 0.5, 0.75, 1]) g.append(sv("text", { x: X(t), y: y + h + 16, "text-anchor": t === 0 ? "start" : t === 1 ? "end" : "middle", class: "tk" }, String(t)));
  for (const [v, t] of [[0, "uncoated reference"], [1, `glassy carbon, I(D3)/I(G) ${top.toFixed(3)}`]])
    g.append(sv("text", { x: X(v), y: y + h + 32, "text-anchor": v === 0 ? "start" : "end", class: "lbl" }, t));
  return g;
}

// --------------------------------------------------------------------------
// the sections that need the analysis
// --------------------------------------------------------------------------
function glance(A) {
  const C = A.calib, V = A.cov, U = A.uni, out = [];
  out.push(C && !C.missing ? tile("Calibrated I(D3)/I(G)", fx(C.median, C.dp), `median · 95 % CI ${fx(C.lo, C.dp)}–${fx(C.hi, C.dp)} · ${fmt(C.n)} carbon ${C.unit}`, null, C.name)
    : tile("Calibrated I(D3)/I(G)", "–", C && C.missing ? C.missing : "not run"));
  out.push(C && C.cmp ? tile("Against the uncoated reference", `${C.cmp.ratio.toFixed(2)} ×`,
    `${sgn(C.cmp.diff, C.dp)} (95 % CI ${fx(C.cmp.lo, C.dp)} to ${fx(C.cmp.hi, C.dp)})`, null, Number.isFinite(C.cmp.index) ? `uncoated 0 → glassy carbon 1: ${C.cmp.index.toFixed(3)}` : C.ref ? C.ref.name : "")
    : tile("Against the uncoated reference", "–", "no uncoated reference chosen"));
  if (V && V.mode === "pf") out.push(tile("Particles coated", `${fmt(V.x)} of ${fmt(V.n)}`, `${pct(V.c, 1)} of those judged · 95 % lower bound ${pct(V.lower, 1)}`, V.pass ? "good" : V.lower >= 0.9 ? "warn" : "bad",
    `${V.e ? `${fmt(V.e)} more not judgeable, counted uncoated in the bound · ` : ""}certified ≥ ${pct(V.certified, 1)}`));
  else if (V && V.mode === "map") out.push(tile("Area covered by coating", pct(V.areaShare, 1), `${area(V.covered)} of the ${area(V.judged)} judged`, V.lowest >= V.spec ? "good" : V.mean >= V.spec ? "warn" : "bad", `lowest particle's 95 % bound ${pct(V.lowest, 1)}`));
  else out.push(tile(A.kind === "pf" ? "Particles coated" : "Area covered by coating", "–", "needs an uncoated reference or a threshold τ (Particles › Coverage)"));
  if (U) {
    const [h1, ...rest] = (U.head || "–").split("; ");      // a map's verdict has two clauses: the second goes below
    out.push(tile("Coating uniformity", h1, `${rest.length ? `${rest.join("; ")} · ` : ""}${U.mode === "pf" ? `I² ${pct(U.I2)} · particle-to-particle CV ${pct(U.cv)}` : `inside particles: I² ${pct(U.W.I2)} · CV ${pct(U.W.cv)}`}`, U.level));
  }
  else out.push(tile("Coating uniformity", "–", A.kind === "map" ? "find the particles under Particles" : "too few quantified particles"));
  return el("div", { class: "tiles" }, ...out);
}

function coverageSection(A) {
  const V = A.cov, kids = [];
  if (!V || V.reason) {
    kids.push(el("div", { class: "callout" }, el("b", {}, "Coverage not computed. "),
      (V && V.reason) || "Set an uncoated reference measured in the same session, or a threshold τ, under Particles › Coverage, then refresh this report."));
    return kids;
  }
  const how = `A spectrum counts as coated when its ${V.band}/G exceeds τ = ${fx(V.tau, 3)}${V.refName ? `, the 99th percentile of the uncoated reference ${V.refName}` : " (entered by hand)"}. Only spectra with G of at least ${fmt(V.gmin, 3)} counts are judged: below that a coating of ${V.band}/G ${fx(V.rc, 3)} cannot be told from no coating with the chosen power.`;
  if (V.mode === "pf") {
    kids.push(el("p", { class: "lead" }, el("b", {}, `${fmt(V.x)} of the ${fmt(V.n)} particles judged carry the coating (${pct(V.c, 1)}).`),
      V.e ? ` ${fmt(V.e)} more particles above the signal floor could not be judged (their ${V.band} or G fit sat on a bound, or ${V.band} came out at zero height), so coverage lies between ${pct(V.lo, 1)} (all of them uncoated) and ${pct(V.hi, 1)} (all coated).` : "",
      ` The exact 95 % lower bound, counting those as uncoated, is ${pct(V.lower, 1)}, which certifies at least ${pct(V.certified, 1)} of particles coated once detection power (${pct(V.piBar, 1)}) and 1 % false detections are allowed for. ` +
      (V.pass ? `That meets the ${pct(V.spec, 1)} certification level.` : `That does not certify ${pct(V.spec, 1)}${V.needed ? `: ${fmt(V.needed)} particles above the floor, all found coated, would` : ""}.`)));
    kids.push(meter({ value: V.c, lower: V.lower, target: V.need, targetLabel: `the lower bound needed to certify ${pct(V.spec, 1)}`, label: "Share of judged particles coated" }));
    kids.push(composition([
      { name: "coated", n: V.x, colour: COL.coated },
      { name: "uncoated", n: V.n - V.x, colour: COL.uncoated },
      { name: "not judgeable (fit on a bound)", n: V.e, colour: COL.other },
      { name: "below the signal floor", n: V.pool - V.n, colour: COL.track, hatch: true }], "Particles by outcome"));
    if (V.mix) kids.push(el("p", { class: "note" }, V.mix.mode === "reference"
      ? `Without any threshold, as a mixture of the uncoated reference's own distribution and one coated population over all ${fmt(V.quantified)} quantified particles: ${pct(V.mix.uncoated, 1)} uncoated (95 % CI ${pct(V.mix.ci[0], 1)}–${pct(V.mix.ci[1], 1)}).`
      : V.mix.two ? `Without any threshold, the quantified particles split into two populations, a low group holding ${pct(V.mix.uncoated, 1)} (95 % CI ${pct(V.mix.ci[0], 1)}–${pct(V.mix.ci[1], 1)}); whether it is uncoated needs a reference.`
        : "Without any threshold, the quantified particles form one population: no group of uncoated particles stands apart."));
  } else {
    kids.push(el("p", { class: "lead" }, el("b", {}, `${area(V.covered)} of the ${area(V.judged)} of particle area bright enough to judge is covered by the coating (${pct(V.areaShare, 1)}).`),
      ` Per particle the covered share averages ${pct(V.mean, 1)}, and the lowest particle's 95 % lower bound is ${pct(V.lowest, 1)}. The ${fmt(V.parts.length)} particles analysed cover ${area(V.total)} of the map; ${pct(V.judged / V.total, 0)} of that area is bright enough to judge.`));
    kids.push(particleBars(V.parts.slice(0, 14)));
    kids.push(el("table", { class: "num" },
      el("tr", {}, ...["Particle", "Area", "Judged", "Covered", "Covered share", "95 % lower bound"].map(h => el("th", {}, h))),
      ...V.parts.map(r => el("tr", {}, el("td", {}, String(r.id)), el("td", {}, area(r.area)), el("td", {}, area(r.judged)), el("td", {}, area(r.covered)), el("td", {}, pct(r.c, 1)), el("td", {}, pct(r.bound, 1)))),
      el("tr", { class: "total" }, el("td", {}, "All"), el("td", {}, area(V.total)), el("td", {}, area(V.judged)), el("td", {}, area(V.covered)), el("td", {}, pct(V.areaShare, 1)), el("td", {}, `lowest ${pct(V.lowest, 1)}`))));
  }
  kids.push(el("p", { class: "note" }, how + (V.mode === "map" ? " Each lower bound is the lower of an exact bound that counts the pixels as the independent values they are worth and a resampling of whole 8 × 8 pixel blocks; the range on each bar counts pixels whose fit sat on a bound both ways." : "")));
  return kids;
}

function calibrationSection(A) {
  const C = A.calib;
  if (!C || C.missing) return [el("div", { class: "callout" }, el("b", {}, "Calibrated D3/G not available. "), (C && C.missing) || "Open the Particles tab once to run it.")];
  const rows = [];
  if (C.ref) rows.push({ name: `Uncoated: ${C.ref.name.length > 22 ? C.ref.name.slice(0, 21) + "…" : C.ref.name}`, v: C.ref.median, lo: C.ref.lo, hi: C.ref.hi, colour: COL.uncoated,
                         text: Number.isFinite(C.ref.lo) ? `${fx(C.ref.median, C.dp)} (${fx(C.ref.lo, C.dp)}–${fx(C.ref.hi, C.dp)})` : fx(C.ref.median, C.dp) });
  rows.push({ name: "This sample", v: C.median, lo: C.lo, hi: C.hi, colour: COL.coated, text: `${fx(C.median, C.dp)} (${fx(C.lo, C.dp)}–${fx(C.hi, C.dp)})` });
  const kids = [el("p", { class: "lead" }, el("b", {}, `I(D3)/I(G) ${fx(C.median, C.dp)} (95 % CI ${fx(C.lo, C.dp)}–${fx(C.hi, C.dp)})`),
    ` over ${fmt(C.n)} carbon ${C.unit} of ${fmt(C.measured)}, with the band shapes of ${C.name} held and only the heights fitted.` +
    (C.cmp ? ` That is ${C.cmp.ratio.toFixed(2)} × the uncoated reference (${fx(C.refMed, C.dp)}): a difference of ${sgn(C.cmp.diff, C.dp)} (95 % CI ${fx(C.cmp.lo, C.dp)} to ${fx(C.cmp.hi, C.dp)}), ` +
      (C.cmp.lo > 0 ? "more D3 than the reference beyond the spread of the measurement." : C.cmp.hi < 0 ? "less D3 than the reference beyond the spread of the measurement." : "not told apart from the reference at this spread.") : " No uncoated reference was chosen to compare with.")),
    el("div", { class: "fig-h" }, "Median I(D3)/I(G) with its 95 % interval"), dotWhisker(rows, "Median I(D3)/I(G) with 95 % intervals")];
  if (C.cmp && Number.isFinite(C.cmp.index)) kids.push(el("div", { class: "fig-h" }, "Between the uncoated reference (0) and glassy carbon, the amorphous reference (1)"),
    scaleGauge(C.cmp.index, C.cmp.indexLo, C.cmp.indexHi, C.top),
    el("p", { class: "note" }, `(R − R_uncoated) / (R_glassy − R_uncoated) on I(D3)/I(G) with the same held shapes: ${C.cmp.index.toFixed(3)} (95 % CI ${C.cmp.indexLo.toFixed(3)} to ${C.cmp.indexHi.toFixed(3)}). It places the sample's D3/G between the two references; it is not a mass or thickness fraction.`));
  kids.push(el("table", { class: "num compact" }, el("tr", {}, ...["Band", "Position (cm⁻¹)", "FWHM (cm⁻¹)", "Shape", "From"].map((h, i) => el("th", { class: i > 2 ? "l" : "" }, h))),
    ...C.bands.map(b => el("tr", {}, el("td", {}, b.name), el("td", {}, b.pos.toFixed(1)), el("td", {}, b.fwhm.toFixed(1)), el("td", { class: "l" }, b.shape), el("td", { class: "l" }, b.from)))));
  kids.push(el("p", { class: "note" }, `Median over spectra whose G stands at least 10 × their noise; the intervals resample whole blocks of neighbouring spectra (8 × 8 pixels, or 150 µm squares of particles). A(D3)/A(G) ${fx(C.area, 3)}; residual of the held shapes ${C.rn.toFixed(2)} × the noise${C.rn > 2 ? " (above 2: the bands here differ from the calibration's)" : ""}.`));
  return kids;
}

// one row per question the analysis answers, its result and what it means
function statsSummary(A) {
  const rows = [], U = A.uni, V = A.cov, C = A.calib;
  const row = (q, r, m, level) => rows.push(el("tr", {}, el("td", {}, q), el("td", {}, r), el("td", {}, m), el("td", {}, level ? pill(level) : "")));
  if (C && !C.missing) {
    row("Calibrated I(D3)/I(G)", `${fx(C.median, C.dp)} (95 % CI ${fx(C.lo, C.dp)}–${fx(C.hi, C.dp)}), n ${fmt(C.n)}`, "Median with band shapes held, comparable between samples");
    if (C.cmp) row("More D3 than uncoated?", `${sgn(C.cmp.diff, C.dp)} (${fx(C.cmp.lo, C.dp)} to ${fx(C.cmp.hi, C.dp)}), ${C.cmp.ratio.toFixed(2)} ×`,
      C.cmp.lo > 0 ? "Yes: the interval lies above zero" : C.cmp.hi < 0 ? "No: less than the reference" : "Not told apart at this spread", C.cmp.lo > 0 ? "good" : "warn");
    if (C.cmp && Number.isFinite(C.cmp.index)) row("Uncoated 0 → glassy carbon 1", `${fx(C.cmp.index, 3)} (${fx(C.cmp.indexLo, 3)} to ${fx(C.cmp.indexHi, 3)})`, "Place of D3/G between the two references");
    row("Do the held shapes fit?", `residual ${C.rn.toFixed(2)} × noise`, C.rn > 2 ? "Above 2: bands differ from the calibration's" : "Yes, within about the noise", C.rn > 2 ? "warn" : "good");
  }
  if (U && U.mode === "pf") {
    row("Same coating on every particle?", `Cochran's Q ${pv(U.p)}, I² ${pct(U.I2)}`, U.head || "", U.level);
    row("How much do particles differ?", `CV ${pct(U.cv)} (95 % CI ${pct(U.cvLo)}–${pct(U.cvHi)})`, "Spread of the true values, noise taken out");
    row(`Is ${U.band} detected?`, `${fmt(U.nC)} of ${fmt(U.n)} (${pct(U.nC / U.n)})`, U.nC / U.n < 0.2 ? "Few: results rest mostly on noise" : "Height ≥ 3 standard errors");
    if (Number.isFinite(U.rho)) row("Drift along the run?", `ρ ${fx(U.rho, 2)}`, U.drift ? `Yes, ${fx(U.drift.ends[0], 3)} → ${fx(U.drift.ends[1], 3)}; removed first` : "None found", U.drift ? "warn" : "good");
    if (U.moran) row("Neighbouring particles alike?", `Moran's I ${fx(U.moran.I, 2)}, ${pv(U.moran.p)}`, U.moran.p < 0.05 && U.moran.I > 0.05 ? "Some variation is across the slide" : "No spatial clustering");
    if (Number.isFinite(U.rhoSize)) row("Does coating follow size?", `ρ ${fx(U.rhoSize, 2)} (${fx(U.rhoSizeCI[0], 2)} to ${fx(U.rhoSizeCI[1], 2)})`,
      U.rhoSizeCI[0] > 0 ? "Larger particles carry more" : U.rhoSizeCI[1] < 0 ? "Larger particles carry less" : "No trend with size");
    if (U.thin) row("Thinnest 5 % of particles", `≤ ${fx(U.thin.value, 3)} (${fx(U.thin.ci[0], 3)}–${fx(U.thin.ci[1], 3)})`, "True values, each particle's noise taken out");
  } else if (U && U.mode === "map") {
    row("Even within particles?", `I² ${pct(U.W.I2)}, ${pv(U.W.p)}`, U.head || "", U.level);
    row("Patchiness inside a particle", `CV ${pct(U.W.cv)} (95 % CI ${pct(U.W.cvLo)}–${pct(U.W.cvHi)})`, "Pixel to pixel, noise taken out");
    if (U.B) row("Particles differ?", `CV ${pct(U.B.cv)} (${pct(U.B.cvLo)}–${pct(U.B.cvHi)})`, Number.isFinite(U.shareWithin) ? `${pct(U.shareWithin)} of the real variation lies within particles` : "");
    row(`Is ${U.band} detected?`, `${pct(U.det)} of pixels`, U.det < 0.2 ? "Few: results rest mostly on noise" : "Height ≥ 3 standard errors");
    if (U.particles > 1) row("Patchy particles", `${fmt(U.patchy)} of ${fmt(U.particles)}`, `${fx(0.05 * U.particles, 1)} expected by chance`);
    if (U.rim) row("Rim against the interior", `${U.rim.ratio.toFixed(2)} × ${U.band}/G, G ${U.rim.g.toFixed(2)} ×`, U.rimLeft ? "Rim left out: the spot lies partly off the particle" : "Rim included");
    if (U.pairs && U.pairs.length > 1) { const a = U.pairs[0], b = U.pairs[U.pairs.length - 1]; row("Patches in space?", `${a.ratio.toFixed(2)} × at ${fx(a.um, 2)} µm, ${b.ratio.toFixed(2)} × at ${fx(b.um, 2)} µm`, b.ratio > a.ratio + 0.2 ? "Yes: differences grow with distance" : "Not organised in space"); }
  }
  if (V && V.mode === "pf") row("Particles coated", `${fmt(V.x)} of ${fmt(V.n)} (${pct(V.c, 1)}), lower bound ${pct(V.lower, 1)}`, V.pass ? `Certifies ${pct(V.spec, 1)}` : `Does not certify ${pct(V.spec, 1)}`, V.pass ? "good" : "bad");
  else if (V && V.mode === "map") row("Area covered", `${pct(V.areaShare, 1)} (${area(V.covered)} of ${area(V.judged)})`, `Lowest particle's 95 % bound ${pct(V.lowest, 1)}`, V.lowest >= V.spec ? "good" : V.mean >= V.spec ? "warn" : "bad");
  if (A.accept && A.accept.head) row("Acceptance against the method's limits", A.accept.head, "Limits fixed under Particles › Acceptance", A.accept.level);
  if (!rows.length) return [el("p", { class: "note" }, "No analysis yet: open the Particles tab once on this file.")];
  return [el("table", { class: "stats-sum" }, el("tr", {}, ...["Question", "Result", "What it means", ""].map(h => el("th", {}, h))), ...rows),
    el("p", { class: "note" }, A.kind === "pf"
      ? "Uniformity: Cochran's Q tests one coating for all particles against each particle's own error; I² is the share of the spread beyond the errors; the CV is Paule-Mandel's with a Q-profile interval. Drift: Spearman ρ along the particle order, removed with a running median before the tests. Clustering: Moran's I over 6 nearest neighbours. Thinnest 5 %: random effects with each particle's error taken out, interval from 400 resamples."
      : "Uniformity inside particles: Cochran's Q over each particle's pixels against their own errors, pooled; I² is the share of the spread beyond the errors; CVs are Paule-Mandel with Q-profile intervals. Between particles: the same test on particle means. Pixel pairs: the mean squared difference of pixel pairs at a distance over what their errors allow (1 when nothing changes).")];
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
