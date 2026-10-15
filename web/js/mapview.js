// A Raman map panel: isotropic um axes, colour bar, scale bar, hover read-out,
// linked crosshair, optional map filter and interpolation (display only) and overlays.
// Stage convention as on the instrument: X to the right, Y increasing DOWN.

import { cssGradient, lut, COLORMAPS } from "./colormaps.js";
import { el, fmt, hidpi, niceTicks, quantile } from "./util.js";

const HATCH = (() => {
  const c = document.createElement("canvas"); c.width = c.height = 8;
  const g = c.getContext("2d");
  g.fillStyle = "#0a1630"; g.fillRect(0, 0, 8, 8);
  g.strokeStyle = "rgba(140,170,220,.22)"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, 8); g.lineTo(8, 0); g.stroke();
  return c;
})();

export class MapView {
  constructor(host, opts = {}) {
    this.host = host;
    this.opts = opts;
    this.cmap = opts.cmap || "LabSpec";
    this.scale = "robust";
    this.vmin = 0; this.vmax = 1;
    this.manual = [null, null];
    this.smooth = false;
    this.filter = "none";            // display filter: none, gauss3, gauss5, median3, median5
    this._shownCache = null;
    this.cursor = null; this.pinned = null;
    this.values = null; this.valid = null;
    this.overlays = [];
    this.label = ""; this.unit = "";

    host.classList.add("mapview");
    this.titleEl = el("button", { class: "map-title", title: "Choose what this map shows" });
    this.titleEl.addEventListener("click", e => opts.onTitle && opts.onTitle(this, e));
    this.readout = el("span", { class: "map-readout" });
    const tools = el("div", { class: "map-tools" },
      el("button", { class: "icon", title: "Display options", onclick: e => this.menu(e) }, "⚙"),
      el("button", { class: "icon", title: "Save as PNG", onclick: () => this.save() }, "⤓"),
      el("button", { class: "icon", title: "Enlarge", onclick: () => this.toggleMax() }, "⤢"));
    this.head = el("div", { class: "map-head" }, this.titleEl, this.readout, tools);
    this.canvas = el("canvas", { class: "map-canvas" });
    this.body = el("div", { class: "map-body" }, this.canvas);
    host.append(this.head, this.body);
    this.off = document.createElement("canvas");

    this.canvas.addEventListener("mousemove", e => this._hover(e));
    this.canvas.addEventListener("mouseleave", () => { this.readout.textContent = ""; opts.onHover && opts.onHover(null); });
    this.canvas.addEventListener("click", e => { const p = this._pick(e); if (p != null && opts.onPick) opts.onPick(p); });
    new ResizeObserver(() => this.render()).observe(this.body);
  }

  setGeometry({ nx, ny, x, y }) {
    this.nx = nx; this.ny = ny; this.x = x; this.y = y;
    this.off.width = nx; this.off.height = ny;
    this._dirty = true;
    this.dx = nx > 1 ? Math.abs(x[1] - x[0]) : 1;
    this.dy = ny > 1 ? Math.abs(y[1] - y[0]) : 1;
    this.xAsc = nx < 2 || x[nx - 1] > x[0];
    this.yAsc = ny < 2 || y[ny - 1] > y[0];
    this.x0 = Math.min(x[0], x[nx - 1]) - this.dx / 2; this.x1 = Math.max(x[0], x[nx - 1]) + this.dx / 2;
    this.y0 = Math.min(y[0], y[ny - 1]) - this.dy / 2; this.y1 = Math.max(y[0], y[ny - 1]) + this.dy / 2;
  }

  setTitle(html) { this.titleEl.innerHTML = html + ' <span class="caret">▾</span>'; }

  setData(values, { valid = null, label = "", unit = "", fixedRange = null } = {}) {
    this.values = values; this.valid = valid; this.label = label; this.unit = unit;
    this.fixedRange = fixedRange;
    this._shownCache = null;
    this._range();
    this._dirty = true;
    this.render();
  }

  // the values drawn: the map itself, or filtered over valid neighbours
  shown() {
    const v = this.values;
    if (!v || this.filter === "none") return v;
    if (this._shownCache && this._shownCache.f === this.filter) return this._shownCache.v;
    const { nx, ny, valid } = this, k = +this.filter.slice(-1), r = (k - 1) / 2, med = this.filter.startsWith("median");
    const g = Array.from({ length: k }, (_, i) => Math.exp(-0.5 * ((i - r) / (k / 4)) ** 2));
    const ok = p => (!valid || valid[p]) && Number.isFinite(v[p]);
    const out = new Float32Array(v.length).fill(NaN), buf = [];
    for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
      const p = i * nx + j;
      if (!ok(p)) continue;
      let s = 0, w = 0; buf.length = 0;
      for (let di = -r; di <= r; di++) for (let dj = -r; dj <= r; dj++) {
        const a = i + di, b = j + dj, q = a * nx + b;
        if (a < 0 || a >= ny || b < 0 || b >= nx || !ok(q)) continue;
        if (med) buf.push(v[q]); else { const wt = g[di + r] * g[dj + r]; s += wt * v[q]; w += wt; }
      }
      if (med) { buf.sort((x, y) => x - y); const m = buf.length; out[p] = m % 2 ? buf[(m - 1) / 2] : 0.5 * (buf[m / 2 - 1] + buf[m / 2]); }
      else out[p] = s / w;
    }
    this._shownCache = { f: this.filter, v: out };
    return out;
  }

  _range() {
    const v = this.shown(); if (!v) return;
    if (this.scale === "manual" && this.manual[0] != null) { [this.vmin, this.vmax] = this.manual; return; }
    if (this.fixedRange && this.scale !== "full") { [this.vmin, this.vmax] = this.fixedRange; return; }
    const a = [];
    for (let i = 0; i < v.length; i++) if ((!this.valid || this.valid[i]) && Number.isFinite(v[i])) a.push(v[i]);
    if (!a.length) { this.vmin = 0; this.vmax = 1; return; }
    a.sort((p, q) => p - q);
    if (this.scale === "full") { this.vmin = a[0]; this.vmax = a[a.length - 1]; }
    else { this.vmin = quantile(a, 0.01); this.vmax = quantile(a, 0.99); }
    if (!(this.vmax > this.vmin)) this.vmax = this.vmin + (Math.abs(this.vmin) || 1) * 1e-3;
  }

  _paintImage() {
    const { nx, ny, valid } = this, v = this.shown();
    const g = this.off.getContext("2d");
    const img = g.createImageData(nx, ny);
    const L = lut(this.cmap), span = this.vmax - this.vmin;
    for (let r = 0; r < ny; r++) {
      const i = this.yAsc ? r : ny - 1 - r;
      for (let c = 0; c < nx; c++) {
        const j = this.xAsc ? c : nx - 1 - c, p = i * nx + j, o = (r * nx + c) * 4;
        if ((valid && !valid[p]) || !Number.isFinite(v[p])) { img.data[o + 3] = 0; continue; }
        const t = Math.max(0, Math.min(255, Math.round((v[p] - this.vmin) / span * 255))) * 3;
        img.data[o] = L[t]; img.data[o + 1] = L[t + 1]; img.data[o + 2] = L[t + 2]; img.data[o + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
  }

  // stage um -> canvas px, and back
  sx(xu) { return this.plot.x + (xu - this.x0) * this.k; }
  sy(yu) { return this.plot.y + (yu - this.y0) * this.k; }

  render() {
    if (!this.nx) return;
    const W = this.body.clientWidth, H = this.body.clientHeight;
    if (W < 20 || H < 20) return;
    const ctx = hidpi(this.canvas, W, H);
    this.ctx = ctx;
    const padL = 34, padB = 26, padT = 6, cbW = 12, cbPad = 50;
    const aw = W - padL - cbW - cbPad - 8, ah = H - padB - padT;
    const wu = this.x1 - this.x0, hu = this.y1 - this.y0;
    this.k = Math.max(0.1, Math.min(aw / wu, ah / hu));
    const pw = wu * this.k, ph = hu * this.k;
    this.plot = { x: padL + (aw - pw) / 2, y: padT + (ah - ph) / 2, w: pw, h: ph };
    const P = this.plot;

    ctx.fillStyle = ctx.createPattern(HATCH, "repeat");
    ctx.fillRect(P.x, P.y, P.w, P.h);
    if (this.values) {
      if (this._dirty) { this._paintImage(); this._dirty = false; }
      ctx.imageSmoothingEnabled = this.smooth;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(this.off, P.x, P.y, P.w, P.h);
    }
    for (const o of this.overlays) o(ctx, this);

    // axes
    ctx.strokeStyle = "rgba(160,200,255,.35)"; ctx.lineWidth = 1;
    ctx.strokeRect(P.x - .5, P.y - .5, P.w + 1, P.h + 1);
    ctx.fillStyle = "#9db6db"; ctx.font = "10px var(--mono, monospace)";
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of niceTicks(this.x0, this.x1, Math.max(2, Math.floor(P.w / 60)))) {
      const X = this.sx(t); ctx.fillRect(X, P.y + P.h, 1, 3); ctx.fillText(fmt(t, 3), X, P.y + P.h + 5);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const t of niceTicks(this.y0, this.y1, Math.max(2, Math.floor(P.h / 45)))) {
      const Y = this.sy(t); ctx.fillRect(P.x - 3, Y, 3, 1); ctx.fillText(fmt(t, 3), P.x - 5, Y);
    }
    ctx.save(); ctx.fillStyle = "#6f8ab3"; ctx.font = "10px var(--sans, sans-serif)";
    ctx.textAlign = "center"; ctx.fillText("X (µm)", P.x + P.w / 2, H - 6);
    ctx.translate(9, P.y + P.h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("Y (µm)", 0, 0); ctx.restore();

    // scale bar (as on the instrument: cyan, bottom right)
    const target = wu * 0.22, mag = 10 ** Math.floor(Math.log10(target));
    const len = [1, 2, 5, 10].map(m => m * mag).filter(v => v <= target).pop() || mag;
    const bx = P.x + P.w - 10 - len * this.k, by = P.y + P.h - 12;
    if (P.w > 120) {
      ctx.fillStyle = "rgba(0,0,0,.35)"; ctx.fillRect(bx - 5, by - 15, len * this.k + 10, 22);
      ctx.fillStyle = "#3fe0ff"; ctx.fillRect(bx, by, len * this.k, 4);
      ctx.font = "10px var(--sans, sans-serif)"; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.fillText(`${len} µm`, bx + len * this.k / 2, by - 2);
    }

    // colour bar
    const cx = P.x + P.w + 12, cy = P.y, ch = P.h;
    const grad = ctx.createLinearGradient(0, cy + ch, 0, cy);
    const L = lut(this.cmap);
    for (let s = 0; s <= 16; s++) { const q = Math.round(s / 16 * 255) * 3; grad.addColorStop(s / 16, `rgb(${L[q]},${L[q + 1]},${L[q + 2]})`); }
    ctx.fillStyle = grad; ctx.fillRect(cx, cy, cbW, ch);
    ctx.strokeStyle = "rgba(160,200,255,.35)"; ctx.strokeRect(cx - .5, cy - .5, cbW + 1, ch + 1);
    ctx.fillStyle = "#9db6db"; ctx.font = "10px var(--mono, monospace)"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    for (const t of niceTicks(this.vmin, this.vmax, Math.max(2, Math.floor(ch / 40)))) {
      const Y = cy + ch - (t - this.vmin) / (this.vmax - this.vmin) * ch;
      ctx.fillRect(cx + cbW, Y, 3, 1); ctx.fillText(fmt(t, 3), cx + cbW + 5, Y);
    }
    if (this.label) {
      ctx.save(); ctx.translate(W - 6, cy + ch / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = "#6f8ab3"; ctx.font = "10px var(--sans, sans-serif)"; ctx.textAlign = "center";
      ctx.fillText(this.label + (this.unit ? ` (${this.unit})` : ""), 0, 0); ctx.restore();
    }
    this._drawCursor();
  }

  idxXY(p) { const i = Math.floor(p / this.nx), j = p % this.nx; return [this.x[j], this.y[i]]; }

  _drawCursor() {
    const ctx = this.ctx, P = this.plot; if (!ctx) return;
    const mark = (p, colour, cross) => {
      if (p == null) return;
      const [xu, yu] = this.idxXY(p), X = this.sx(xu), Y = this.sy(yu);
      ctx.save();
      ctx.strokeStyle = colour; ctx.lineWidth = 1;
      if (cross) {
        ctx.beginPath(); ctx.moveTo(P.x, Y + .5); ctx.lineTo(P.x + P.w, Y + .5);
        ctx.moveTo(X + .5, P.y); ctx.lineTo(X + .5, P.y + P.h); ctx.stroke();
      }
      ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X, Y, Math.max(3, this.k * this.dx * 0.6), 0, 7); ctx.stroke();
      ctx.restore();
    };
    mark(this.pinned, "#7dff6a", true);
    if (this.cursor !== this.pinned) mark(this.cursor, "rgba(255,255,255,.75)", false);
  }

  setCursor(hover, pinned) {
    this.cursor = hover; this.pinned = pinned;
    this.render();
  }

  _pick(e) {
    if (!this.plot) return null;
    const r = this.canvas.getBoundingClientRect();
    const xu = this.x0 + (e.clientX - r.left - this.plot.x) / this.k;
    const yu = this.y0 + (e.clientY - r.top - this.plot.y) / this.k;
    if (xu < this.x0 || xu > this.x1 || yu < this.y0 || yu > this.y1) return null;
    let j = Math.floor((xu - this.x0) / this.dx), i = Math.floor((yu - this.y0) / this.dy);
    j = Math.max(0, Math.min(this.nx - 1, this.xAsc ? j : this.nx - 1 - j));
    i = Math.max(0, Math.min(this.ny - 1, this.yAsc ? i : this.ny - 1 - i));
    return i * this.nx + j;
  }

  _hover(e) {
    const p = this._pick(e);
    if (p == null) { this.readout.textContent = ""; return; }
    const [xu, yu] = this.idxXY(p);
    const v = this.values ? this.values[p] : NaN;
    const bad = this.valid && !this.valid[p];
    const filt = this.filter !== "none" && this.values ? ` (filtered ${fmt(this.shown()[p])})` : "";
    this.readout.textContent = `${fmt(xu, 3)}, ${fmt(yu, 3)} µm  →  ${bad ? "invalid" : fmt(v) + filt}`;
    this.opts.onHover && this.opts.onHover(p);
  }

  menu(e) {
    e.stopPropagation();
    document.querySelectorAll(".popover").forEach(p => p.remove());
    const sel = el("div", { class: "cmap-grid" }, ...COLORMAPS.map(n => el("button", {
      class: "cmap-swatch" + (n === this.cmap ? " on" : ""), title: n,
      style: `background:${cssGradient(n, "to right")}`,
      onclick: () => { this.cmap = n; this._dirty = true; this.render(); pop.remove(); this.opts.onChange && this.opts.onChange(this); },
    })));
    const mode = el("select", {}, ...[["robust", "1–99 % of valid points"], ["full", "Full range"], ["manual", "Manual"]]
      .map(([v, t]) => el("option", { value: v, selected: v === this.scale }, t)));
    const lo = el("input", { type: "number", step: "any", value: +this.vmin.toPrecision(4) });
    const hi = el("input", { type: "number", step: "any", value: +this.vmax.toPrecision(4) });
    const apply = () => {
      this.scale = mode.value;
      if (this.scale === "manual") this.manual = [parseFloat(lo.value), parseFloat(hi.value)];
      this._range(); this._dirty = true; this.render();
      lo.value = +this.vmin.toPrecision(4); hi.value = +this.vmax.toPrecision(4);
    };
    mode.onchange = apply; lo.onchange = () => { mode.value = "manual"; apply(); }; hi.onchange = lo.onchange;
    const smooth = el("input", { type: "checkbox", checked: this.smooth, onchange: () => { this.smooth = smooth.checked; this.render(); } });
    const filter = el("select", { title: "Filter the map over its valid neighbours. Display and PNG only: statistics, histograms and exports use the raw pixels." },
      ...[["none", "None"], ["gauss3", "Gaussian 3 × 3"], ["gauss5", "Gaussian 5 × 5"], ["median3", "Median 3 × 3"], ["median5", "Median 5 × 5"]]
        .map(([v, t]) => el("option", { value: v, selected: v === this.filter }, t)));
    filter.onchange = () => { this.filter = filter.value; this._range(); this._dirty = true; this.render(); this.opts.onChange && this.opts.onChange(this); };
    const pop = el("div", { class: "popover", onclick: ev => ev.stopPropagation() },
      el("div", { class: "pop-h" }, "Colour map"), sel,
      el("div", { class: "pop-h" }, "Scale"), mode, el("div", { class: "row2" }, lo, hi),
      el("div", { class: "pop-h" }, "Filter ", el("span", { class: "muted" }, "(display only, stats use raw pixels)")), filter,
      el("label", { class: "chk" }, smooth, " Interpolate pixels ", el("span", { class: "muted" }, "(display only)")));
    document.body.append(pop);
    const r = e.currentTarget.getBoundingClientRect();
    pop.style.left = Math.min(window.innerWidth - 250, r.left - 200) + "px";
    pop.style.top = r.bottom + 6 + "px";
    setTimeout(() => document.addEventListener("click", () => pop.remove(), { once: true }));
  }

  toggleMax() {
    this.host.classList.toggle("maxed");
    document.body.classList.toggle("has-max", !!document.querySelector(".maxed"));
    setTimeout(() => this.render(), 30);
  }

  save() {
    const a = el("a", { href: this.canvas.toDataURL("image/png"), download: (this.titleEl.textContent || "map").replace(/[^\w\- ]+/g, "").trim() + ".png" });
    a.click();
  }
}
