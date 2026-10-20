// Canvas charts: cursor spectrum with fit components, histogram, and the
// all-spectra density view.

import { lut } from "./colormaps.js";
import { el, fmt, hidpi, niceTicks } from "./util.js";
import { T, font, PEAK_COLOURS } from "./theme.js";

export { PEAK_COLOURS };

class Base {
  constructor(host) {
    this.host = host;
    this.canvas = el("canvas");
    host.append(this.canvas);
    new ResizeObserver(() => this.render()).observe(host);
  }
  frame(pad = { l: 46, r: 12, t: 10, b: 30 }) {
    const W = this.host.clientWidth, H = this.host.clientHeight;
    if (W < 30 || H < 30) return null;
    const ctx = hidpi(this.canvas, W, H);
    return { ctx, W, H, x: pad.l, y: pad.t, w: W - pad.l - pad.r, h: H - pad.t - pad.b };
  }
  axes(F, xr, yr, { xl = "", yl = "", grid = true } = {}) {
    const { ctx } = F;
    const X = v => F.x + (v - xr[0]) / (xr[1] - xr[0]) * F.w;
    const Y = v => F.y + F.h - (v - yr[0]) / (yr[1] - yr[0]) * F.h;
    ctx.font = font(11); ctx.fillStyle = T.ink;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of niceTicks(xr[0], xr[1], Math.max(2, Math.floor(F.w / 70)))) {
      if (grid) { ctx.fillStyle = T.grid; ctx.fillRect(X(t), F.y, 1, F.h); ctx.fillStyle = T.ink; }
      ctx.fillText(fmt(t, 4), X(t), F.y + F.h + 4);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const t of niceTicks(yr[0], yr[1], Math.max(2, Math.floor(F.h / 34)))) {
      if (grid) { ctx.fillStyle = T.grid; ctx.fillRect(F.x, Y(t), F.w, 1); ctx.fillStyle = T.ink; }
      ctx.fillText(fmt(t, 3), F.x - 5, Y(t));
    }
    ctx.strokeStyle = T.frame; ctx.strokeRect(F.x - .5, F.y - .5, F.w + 1, F.h + 1);
    ctx.fillStyle = T.label; ctx.font = font(11); ctx.textAlign = "center";
    if (xl) { ctx.textBaseline = "bottom"; ctx.fillText(xl, F.x + F.w / 2, F.y + F.h + 30); }
    if (yl) { ctx.save(); ctx.translate(10, F.y + F.h / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle"; ctx.fillText(yl, 0, 0); ctx.restore(); }
    return { X, Y };
  }
}

// ---------------------------------------------------------------------------
export class SpectrumPlot extends Base {
  constructor(host) {
    super(host);
    this.view = null;
    this.canvas.addEventListener("mousemove", e => { this.mouse = e.offsetX; this.render(); });
    this.canvas.addEventListener("mouseleave", () => { this.mouse = null; this.render(); });
    this.canvas.addEventListener("mousedown", e => { this.drag = [e.offsetX, e.offsetX]; });
    this.canvas.addEventListener("mouseup", e => {
      if (this.drag && Math.abs(this.drag[0] - e.offsetX) > 8 && this.F) {
        const inv = px => this.xr[0] + (px - this.F.x) / this.F.w * (this.xr[1] - this.xr[0]);
        const a = inv(Math.min(this.drag[0], e.offsetX)), b = inv(Math.max(this.drag[0], e.offsetX));
        this.view = [a, b];
      }
      this.drag = null; this.render();
    });
    this.canvas.addEventListener("dblclick", () => { this.view = null; this.render(); });
    this.canvas.title = "Drag to zoom, double-click to reset";
  }
  setData(d) { this.d = d; this.render(); }
  render() {
    const F = this.frame({ l: 50, r: 12, t: 12, b: 34 }); if (!F) return;
    this.F = F;
    const d = this.d, { ctx } = F;
    if (!d) { ctx.fillStyle = T.label; ctx.font = font(13); ctx.textAlign = "center"; ctx.fillText(this.emptyText || "Hover or click a map to see its spectrum", F.x + F.w / 2, F.y + F.h / 2); return; }
    const wn = d.wn, full = [wn[0], wn[wn.length - 1]];
    const xr = this.view || full; this.xr = xr;
    const inX = i => wn[i] >= xr[0] && wn[i] <= xr[1];
    const resH = d.fit && !d.modelOnly ? Math.min(46, F.h * 0.2) : 0;
    const main = { ...F, h: F.h - resH - (resH ? 6 : 0) };
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < wn.length; i++) if (inX(i)) { lo = Math.min(lo, d.raw[i]); hi = Math.max(hi, d.raw[i]); }
    if (d.raw0) for (let i = 0; i < wn.length; i++) if (inX(i)) { lo = Math.min(lo, d.raw0[i]); hi = Math.max(hi, d.raw0[i]); }
    if (d.fit) for (let i = 0; i < wn.length; i++) if (inX(i)) hi = Math.max(hi, d.fit.total[i]);
    const pad = (hi - lo) * 0.08 || 1; const yr = [Math.min(0, lo - pad), hi + pad];
    const { X, Y } = this.axes(main, xr, yr, { yl: "Intensity (counts)" });
    ctx.save(); ctx.beginPath(); ctx.rect(main.x, main.y, main.w, main.h); ctx.clip();
    // spectral range: dim outside, green band on top like the instrument
    if (d.range) {
      ctx.fillStyle = T.shade;
      ctx.fillRect(main.x, main.y, X(d.range[0]) - main.x, main.h);
      ctx.fillRect(X(d.range[1]), main.y, main.x + main.w - X(d.range[1]), main.h);
      ctx.fillStyle = T.range; ctx.fillRect(X(d.range[0]), main.y, X(d.range[1]) - X(d.range[0]), 4);
    }
    const line = (ys, col, w = 1, dash = []) => {
      ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash); ctx.beginPath();
      let s = false;
      for (let i = 0; i < wn.length; i++) { if (!inX(i) && !(i && inX(i - 1)) && !(i + 1 < wn.length && inX(i + 1))) continue; const px = X(wn[i]), py = Y(ys[i]); s ? ctx.lineTo(px, py) : ctx.moveTo(px, py); s = true; }
      ctx.stroke(); ctx.setLineDash([]);
    };
    if (d.fit) {
      d.fit.comps.forEach((c, j) => {
        const col = PEAK_COLOURS[j % PEAK_COLOURS.length];
        ctx.beginPath();
        for (let i = 0; i < wn.length; i++) ctx.lineTo(X(wn[i]), Y(d.fit.baseline[i] + c.y[i]));
        for (let i = wn.length - 1; i >= 0; i--) ctx.lineTo(X(wn[i]), Y(d.fit.baseline[i]));
        ctx.fillStyle = col + "33"; ctx.fill();
        const top = d.fit.baseline.map((b, i) => b + c.y[i]); line(top, col, 1.2);
      });
      line(d.fit.baseline, T.baseline, 1, [4, 3]);
    }
    if (d.raw0) line(d.raw0, T.raw0, 1);   // the raw spectrum behind its smoothed version
    if (!d.modelOnly) line(d.raw, T.raw, 1);
    if (d.fit) line(d.fit.total, T.fit, 1.8);
    // peak labels at fitted centres, rotated like LabSpec
    if (d.fit) {
      ctx.font = font(11);
      d.fit.comps.forEach((c, j) => {
        if (!(c.A > 0) || c.pos < xr[0] || c.pos > xr[1]) return;
        let k = 0; while (k < wn.length - 1 && wn[k] < c.pos) k++;
        const px = X(c.pos), py = Y(d.fit.total[k]) - 6;
        ctx.save(); ctx.translate(px, py); ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = PEAK_COLOURS[j % PEAK_COLOURS.length]; ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText(`${c.name} ${c.pos.toFixed(0)}`, 0, 0); ctx.restore();
      });
    }
    if (this.drag) { ctx.fillStyle = "rgba(63,224,255,.12)"; ctx.fillRect(Math.min(...this.drag), main.y, Math.abs(this.drag[1] - this.drag[0]), main.h); }
    if (d.modelOnly) {
      ctx.fillStyle = T.label; ctx.font = font(11); ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(d.caption || "Rebuilt from the fitted values: this file holds no spectra", main.x + 8, main.y + 8);
    }
    ctx.restore();
    // residual strip
    if (d.fit && !d.modelOnly) {
      const R = { x: F.x, y: main.y + main.h + 6, w: F.w, h: resH };
      let m = 1e-9; const res = d.raw.map((v, i) => v - d.fit.total[i]);
      for (let i = 0; i < wn.length; i++) if (inX(i) && wn[i] >= d.range[0] && wn[i] <= d.range[1]) m = Math.max(m, Math.abs(res[i]));
      const RY = v => R.y + R.h / 2 - v / m * R.h / 2;
      ctx.fillStyle = T.residBg; ctx.fillRect(R.x, R.y, R.w, R.h);
      ctx.fillStyle = T.frame; ctx.fillRect(R.x, R.y + R.h / 2, R.w, 1);
      ctx.strokeStyle = T.resid; ctx.lineWidth = 1; ctx.beginPath();
      let s = false;
      for (let i = 0; i < wn.length; i++) { if (!inX(i) || wn[i] < d.range[0] || wn[i] > d.range[1]) continue; const px = X(wn[i]); s ? ctx.lineTo(px, RY(res[i])) : ctx.moveTo(px, RY(res[i])); s = true; }
      ctx.stroke();
      ctx.fillStyle = T.label; ctx.font = font(10); ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(`residual ±${fmt(m, 2)}`, R.x + 4, R.y + 2);
    }
    ctx.fillStyle = T.label; ctx.font = font(11); ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText("Raman shift (cm⁻¹)", F.x + F.w / 2, F.H ?? F.y + F.h + 33);
    if (this.mouse != null && this.mouse > F.x && this.mouse < F.x + F.w) {
      const v = xr[0] + (this.mouse - F.x) / F.w * (xr[1] - xr[0]);
      let k = 0; while (k < wn.length - 1 && wn[k] < v) k++;
      ctx.fillStyle = T.hair; ctx.fillRect(this.mouse, F.y, 1, F.h);
      const txt = d.modelOnly ? `${wn[k].toFixed(1)} cm⁻¹   bands ${fmt(d.fit.total[k])}`
        : `${wn[k].toFixed(1)} cm⁻¹   data ${fmt(d.raw[k])}` + (d.fit ? `   fit ${fmt(d.fit.total[k])}` : "");
      ctx.font = font(11); const tw = ctx.measureText(txt).width + 10;
      const tx = Math.min(this.mouse + 8, F.x + F.w - tw);
      ctx.fillStyle = T.tipBg; ctx.fillRect(tx, F.y + 8, tw, 18);
      ctx.fillStyle = T.tipInk; ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(txt, tx + 5, F.y + 17);
    }
  }
}

// ---------------------------------------------------------------------------
export class HistogramPlot extends Base {
  setData(d) { this.d = d; this.render(); }
  render() {
    const F = this.frame({ l: 46, r: 12, t: 12, b: 34 }); if (!F) return;
    const d = this.d, { ctx } = F; if (!d || !d.values.length) return;
    const [lo, hi] = d.range;
    const n = d.bins || Math.max(8, Math.min(60, Math.round(Math.sqrt(d.values.length) / 1.5)));
    const h = new Float64Array(n); let out = 0;
    for (const v of d.values) {
      if (v < lo || v > hi) { out++; continue; }
      h[Math.min(n - 1, Math.floor((v - lo) / (hi - lo) * n))]++;
    }
    const tot = d.values.length; let m = 0; for (const c of h) m = Math.max(m, c / tot * 100);
    const { X, Y } = this.axes(F, [lo, hi], [0, m * 1.1], { yl: "Points (%)" });
    const L = lut(d.cmap || "LabSpec");
    for (let b = 0; b < n; b++) {
      const a = lo + b / n * (hi - lo), z = lo + (b + 1) / n * (hi - lo);
      const t = Math.round((b + .5) / n * 255) * 3;
      const top = Y(h[b] / tot * 100);
      const g = ctx.createLinearGradient(0, top, 0, F.y + F.h);
      g.addColorStop(0, `rgba(${L[t]},${L[t + 1]},${L[t + 2]},.95)`); g.addColorStop(1, `rgba(${L[t]},${L[t + 1]},${L[t + 2]},.35)`);
      ctx.fillStyle = g; ctx.fillRect(X(a) + .5, top, Math.max(1, X(z) - X(a) - 1), F.y + F.h - top);
    }
    const vline = (v, col, label, dash) => {
      if (!(v >= lo && v <= hi)) return;
      ctx.strokeStyle = col; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(X(v), F.y); ctx.lineTo(X(v), F.y + F.h); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = col; ctx.font = font(11); ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(label, X(v) + 3, F.y + (dash.length ? 14 : 2));
    };
    vline(d.median, T.strong, `median ${fmt(d.median)}`, []);
    vline(d.mean, T.mean, `mean ${fmt(d.mean)}`, [4, 3]);
    ctx.fillStyle = T.label; ctx.font = font(11); ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    // the longest caption that fits the axis
    const cap = [d.label + (out ? `   (${out} outside the axis)` : ""), d.label + (out ? `  (${out} off axis)` : ""), d.label]
      .find(t => ctx.measureText(t).width <= F.w + F.x - 4) || d.label;
    ctx.fillText(cap, F.x + F.w / 2, F.y + F.h + 33);
  }
}

// ---------------------------------------------------------------------------
export class DensityPlot extends Base {
  setData(d) { this.d = d; this._img = null; this.render(); }
  _build() {
    const { cube, k, wn } = this.d, P = cube.length / k;
    let hi = 0;
    const sample = [];
    for (let i = 0; i < cube.length; i += 97) if (Number.isFinite(cube[i])) sample.push(cube[i]);
    sample.sort((a, b) => a - b);
    hi = this.d.hi ?? sample[Math.floor(sample.length * 0.999)] * 1.1;
    let lo = Math.min(0, sample[0]);
    const B = 120, H = new Float32Array(B * k), mean = new Float64Array(k);
    let measured = 0;
    for (let p = 0; p < P; p++) {
      const o = p * k;
      if (!Number.isFinite(cube[o])) continue;          // a point never measured
      measured++;
      for (let c = 0; c < k; c++) {
        const v = cube[o + c]; mean[c] += v;
        const b = Math.floor((v - lo) / (hi - lo) * B);
        if (b >= 0 && b < B) H[b * k + c]++;
      }
    }
    for (let c = 0; c < k; c++) mean[c] /= Math.max(1, measured);
    let m = 0; for (const v of H) m = Math.max(m, v);
    const off = document.createElement("canvas"); off.width = k; off.height = B;
    const g = off.getContext("2d"), img = g.createImageData(k, B), L = lut("Ice");
    for (let b = 0; b < B; b++) for (let c = 0; c < k; c++) {
      const v = H[b * k + c], t = v ? Math.round(Math.log1p(v) / Math.log1p(m) * 255) * 3 : 0, o = ((B - 1 - b) * k + c) * 4;
      img.data[o] = L[t]; img.data[o + 1] = L[t + 1]; img.data[o + 2] = L[t + 2]; img.data[o + 3] = v ? 255 : 0;
    }
    g.putImageData(img, 0, 0);
    this._img = { off, lo, hi, mean, wn };
  }
  render() {
    const F = this.frame({ l: 40, r: 8, t: 8, b: 26 }); if (!F || !this.d) return;
    if (!this._img) this._build();
    const { ctx } = F, I = this._img;
    const xr = [I.wn[0], I.wn[I.wn.length - 1]];
    ctx.fillStyle = T.bg; ctx.fillRect(F.x, F.y, F.w, F.h);
    ctx.imageSmoothingEnabled = true; ctx.drawImage(I.off, F.x, F.y, F.w, F.h);
    const { X, Y } = this.axes(F, xr, [I.lo, I.hi], { grid: false });
    const r = this.d.range;
    if (r) {
      ctx.fillStyle = T.shade; ctx.fillRect(F.x, F.y, X(r[0]) - F.x, F.h); ctx.fillRect(X(r[1]), F.y, F.x + F.w - X(r[1]), F.h);
      ctx.fillStyle = T.range; ctx.fillRect(X(r[0]), F.y, X(r[1]) - X(r[0]), 3);
    }
    (this.d.peaks || []).forEach((p, j) => {
      ctx.fillStyle = PEAK_COLOURS[j % PEAK_COLOURS.length]; ctx.fillRect(X(p.pos), F.y + F.h - 8, 1.5, 8);
    });
    ctx.strokeStyle = T.strong; ctx.lineWidth = 1.2; ctx.beginPath();
    I.mean.forEach((v, i) => { const px = X(I.wn[i]), py = Y(v); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    ctx.stroke();
  }
}
