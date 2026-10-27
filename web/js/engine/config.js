// Fitting configuration for the browser: the default and INI load/save, in the
// same format as scarbon/config.py, so a file saved by either version opens in
// the other.

import { checkConfig } from "./core.js";

export const DEFAULT_CONFIG = {
  baseline_order: 2, fit_separately: false, range_lo: 1000, range_hi: 1800,
  intensity: { lo: 0, hi: 5000, use_lo: false, use_hi: true },
  ratio_i: { lo: 0, hi: 10000, use_lo: false, use_hi: false },
  ratio_a: { lo: 0, hi: 10000, use_lo: false, use_hi: false },
  peaks: [
    { name: "G", pos: 1580, pos_min: 1570, pos_max: 1590, width: 20, width_min: 10, width_max: 35, shape: "Lorentz", fit: true },
    { name: "D1", pos: 1350, pos_min: 1340, pos_max: 1360, width: 45, width_min: 15, width_max: 90, shape: "Lorentz", fit: true },
    { name: "D2", pos: 1620, pos_min: 1610, pos_max: 1630, width: 25, width_min: 10, width_max: 60, shape: "Lorentz", fit: true },
    { name: "D3", pos: 1500, pos_min: 1490, pos_max: 1510, width: 100, width_min: 40, width_max: 200, shape: "Gauss", fit: true },
    { name: "D4", pos: 1200, pos_min: 1190, pos_max: 1210, width: 50, width_min: 20, width_max: 120, shape: "Lorentz", fit: true },
  ],
  skip_fitting: false, auto_shapes: true, smooth_window: 0, smooth_order: 2, spatial_kernel: 0,
  ratio_num: "D3", ratio_den: "G", max_iter: 60,
};

const py = v => (typeof v === "boolean" ? (v ? "True" : "False") : String(v));

export function toIni(cfg) {
  const out = ["; Carbo Metrics fitting configuration"];
  const sec = (name, obj) => { out.push(`[${name}]`); for (const [k, v] of Object.entries(obj)) out.push(`${k} = ${py(v)}`); out.push(""); };
  sec("processing", {
    baseline_order: cfg.baseline_order, fit_separately: cfg.fit_separately,
    range_lo: cfg.range_lo, range_hi: cfg.range_hi, skip_fitting: cfg.skip_fitting,
    auto_shapes: cfg.auto_shapes !== false, smooth_window: cfg.smooth_window || 0, smooth_order: cfg.smooth_order ?? 2,
    spatial_kernel: cfg.spatial_kernel || 0, max_iter: cfg.max_iter, ratio: `${cfg.ratio_num}/${cfg.ratio_den}`,
  });
  for (const key of ["intensity", "ratio_i", "ratio_a"]) sec(`constraint.${key}`, cfg[key]);
  cfg.peaks.forEach((p, i) => sec(`peak.${i + 1}`, p));
  return out.join("\n");
}

export function fromIni(text) {
  const secs = {};
  let cur = null;
  for (const raw of text.split(/\r?\n/)) {
    const ln = raw.trim();
    if (!ln || ln.startsWith(";") || ln.startsWith("#")) continue;
    const m = ln.match(/^\[(.+)\]$/);
    if (m) { cur = secs[m[1].trim()] = {}; continue; }
    const kv = ln.match(/^([^=:]+)[=:](.*)$/);
    if (!kv || !cur) throw new Error(`unreadable line: ${ln.slice(0, 60)}`);
    cur[kv[1].trim().toLowerCase()] = kv[2].trim();
  }
  if (!secs.processing) throw new Error("no [processing] section");
  const bool = (v, d) => (v === undefined ? d : /^(true|yes|on|1)$/i.test(v));
  const num = (v, d) => (v === undefined ? d : Number(v));
  const pr = secs.processing, base = structuredClone(DEFAULT_CONFIG);
  const [num_, den_] = (pr.ratio || "D3/G").split("/");
  const rng = key => {
    const s = secs[`constraint.${key}`];
    return s ? { lo: num(s.lo), hi: num(s.hi), use_lo: bool(s.use_lo), use_hi: bool(s.use_hi) } : base[key];
  };
  const peaks = Object.keys(secs).filter(s => s.startsWith("peak.")).sort((a, b) => +a.split(".")[1] - +b.split(".")[1])
    .map(s => {
      const p = secs[s];
      return { name: p.name, pos: num(p.pos), pos_min: num(p.pos_min), pos_max: num(p.pos_max), width: num(p.width),
               width_min: num(p.width_min), width_max: num(p.width_max), shape: p.shape || "Lorentz", fit: bool(p.fit, true) };
    });
  return checkConfig({
    baseline_order: num(pr.baseline_order, 2), fit_separately: bool(pr.fit_separately, false),
    range_lo: num(pr.range_lo, 1000), range_hi: num(pr.range_hi, 1800),
    intensity: rng("intensity"), ratio_i: rng("ratio_i"), ratio_a: rng("ratio_a"),
    peaks: peaks.length ? peaks : base.peaks,
    skip_fitting: bool(pr.skip_fitting, false), auto_shapes: bool(pr.auto_shapes, true),
    smooth_window: num(pr.smooth_window, 0), smooth_order: num(pr.smooth_order, 2), spatial_kernel: num(pr.spatial_kernel, 0),
    ratio_num: num_.trim(), ratio_den: den_.trim(), max_iter: num(pr.max_iter, 60),
  });
}
