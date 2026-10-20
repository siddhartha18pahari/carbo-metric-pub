// A Raman map panel: isotropic um axes, colour bar, scale bar, hover read-out,
// linked crosshair, optional map filter and interpolation (display only) and overlays.
// Stage convention as on the instrument: X to the right, Y increasing DOWN.
// Two geometries: a pixel grid, or ParticleFinder points (one circle per
// particle at its stage position, sized by its diameter).

import { cssGradient, lut, COLORMAPS } from "./colormaps.js";
import { el, fmt, hidpi, niceTicks, quantile } from "./util.js";
import { T, font } from "./theme.js";

// margins around the map: axis, colour bar and its labels
const PAD = { l: 34, b: 26, t: 18, cb: 12, cbPad: 50 };        // t: room for the scale bar

// diagonal hatch behind invalid points, in the current theme's colours
let hatch = null;
function HATCH() {
  if (hatch && hatch.key === T.hatchBg + T.hatchLine) return hatch.c;
  const c = document.createElement("canvas"); c.width = c.height = 8;
  const g = c.getContext("2d");
  g.fillStyle = T.hatchBg; g.fillRect(0, 0, 8, 8);
  g.strokeStyle = T.hatchLine; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, 8); g.lineTo(8, 0); g.stroke();
  hatch = { key: T.hatchBg + T.hatchLine, c };
  return c;
}

export class MapView {
  constructor(host, opts = {}) {
    this.host = host;
    this.opts = opts;
    this.cmap = opts.cmap || "Viridis";      // even brightness steps; the LabSpec rainbow is in the menu
    this.scale = "robust";
    this.vmin = 0; this.vmax = 1;
    this.manual = [null, null];
    this.smooth = false;
    this.filter = "none";            // display filter: none, gauss3, gauss5, median3, median5
    this._shownCache = null;
    this.cursor = null; this.pinned = null;
    this.values = null; this.valid = null;
    this.overlays = [];
    this.emptyText = null;           // shown over the map when it cannot have values
    this.categories = null;          // particles coloured by group: { index, colours, names }
    this.backdrop = null;            // the camera image under the particles: { img, x: [first, last], y: [first, last], w, h }
    this.backdropOn = true;
    this.profile = false;            // a particle list drawn as Q-Carbon does: value against particle number
    this.label = ""; this.unit = "";

    host.classList.add("mapview");
    this.titleEl = el("button", { class: "map-title", title: "Choose what this map shows" });
    this.titleEl.addEventListener("click", e => opts.onTitle && opts.onTitle(this, e));
    this.readout = el("span", { class: "map-readout" });
    // one click between smoothed and raw pixels (display only)
    this.smoothBtn = el("button", { class: "map-chip", onclick: () => {
      this.setSmoothing(this.smoothed() ? { filter: "none", smooth: false } : { filter: "gauss3", smooth: true });
      this.render();
      opts.onSmoothChange && opts.onSmoothChange(this);
      opts.onChange && opts.onChange(this);
    } });
    const tools = el("div", { class: "map-tools" },
      this.resetBtn = el("button", { class: "icon hidden", title: "Show the whole map (or double-click the map)", onclick: () => this.setView(null, true) }, "⟲"),
      el("button", { class: "icon", title: "Display options", onclick: e => this.menu(e) }, "⚙"),
      el("button", { class: "icon", title: "Save as PNG", onclick: () => this.save() }, "⤓"),
      el("button", { class: "icon", title: "Enlarge", onclick: () => this.toggleMax() }, "⤢"));
    this.viewBtn = el("button", { class: "map-chip", onclick: () => (opts.onProfile ? opts.onProfile(!this.profile) : this.setProfile(!this.profile)) });
    this.head = el("div", { class: "map-head" }, this.titleEl, this.smoothBtn, this.viewBtn, this.readout);
    this._chip();
    this.canvas = el("canvas", { class: "map-canvas" });
    this.body = el("div", { class: "map-body" }, this.canvas, tools);     // tools float over the map
    host.append(this.head, this.body);
    this.off = document.createElement("canvas");

    this.canvas.addEventListener("mousemove", e => { if (!this._pan(e)) this._hover(e); });
    this.canvas.addEventListener("mouseleave", () => { this.readout.textContent = ""; opts.onHover && opts.onHover(null); });
    this.canvas.addEventListener("click", e => {
      if (this._dragged) { this._dragged = false; return; }
      const p = this._pick(e); if (p != null && opts.onPick) opts.onPick(p);
    });
    // zoom at the pointer, drag to pan, double-click for the whole map
    this.canvas.addEventListener("wheel", e => { e.preventDefault(); this._wheel(e); }, { passive: false });
    this.canvas.addEventListener("mousedown", e => { if (e.button === 0) this._drag = { x: e.clientX, y: e.clientY, v: this.view() }; });
    window.addEventListener("mouseup", () => { this._drag = null; });
    this.canvas.addEventListener("dblclick", () => this.setView(null, true));
    new ResizeObserver(() => this.render()).observe(this.body);
  }

  // for laying out the cards: the body height at which the whole map fills
  // a body W px wide, and the size the whole map is drawn at in W x H
  fitBody(W) {
    const E = this.ext; if (!E) return 0;
    return Math.ceil((W - PAD.l - PAD.cb - PAD.cbPad - 8) * (E.y1 - E.y0) / (E.x1 - E.x0)) + PAD.b + PAD.t;
  }
  plotSize(W, H) {
    const E = this.ext; if (!E) return { w: 0, h: 0 };
    const wu = E.x1 - E.x0, hu = E.y1 - E.y0;
    const k = Math.max(0, Math.min((W - PAD.l - PAD.cb - PAD.cbPad - 8) / wu, (H - PAD.b - PAD.t) / hu));
    return { w: k * wu, h: k * hu };
  }

  // the region shown, in stage um; the full extent is this.ext
  view() { return { x0: this.x0, x1: this.x1, y0: this.y0, y1: this.y1 }; }
  zoomed() { return !!this.ext && (this.x1 - this.x0 < (this.ext.x1 - this.ext.x0) * 0.999 || this.y1 - this.y0 < (this.ext.y1 - this.ext.y0) * 0.999); }

  setView(v, emit = false) {
    if (!this.ext) return;
    const E = this.ext;
    if (!v) v = { ...E };
    let w = v.x1 - v.x0, h = v.y1 - v.y0;
    const minW = Math.max(3 * this.dx, 1e-6);
    if (w < minW) { const c = (v.x0 + v.x1) / 2, f = minW / w; v = { x0: c - minW / 2, x1: c + minW / 2, y0: (v.y0 + v.y1) / 2 - h * f / 2, y1: (v.y0 + v.y1) / 2 + h * f / 2 }; w = minW; h *= f; }
    if (w >= E.x1 - E.x0 && h >= E.y1 - E.y0) v = { ...E };
    else {                                         // keep the window over the map
      const sx = Math.min(0, E.x1 - v.x1) + Math.max(0, E.x0 - v.x0), sy = Math.min(0, E.y1 - v.y1) + Math.max(0, E.y0 - v.y0);
      if (w < E.x1 - E.x0) { v.x0 += sx; v.x1 += sx; }
      if (h < E.y1 - E.y0) { v.y0 += sy; v.y1 += sy; }
    }
    Object.assign(this, v);
    this._ptsCache = null;
    this.render();
    if (emit && this.opts.onView) this.opts.onView(this.zoomed() ? this.view() : null, this);
  }

  // centre the view on a stage point, half-width hw um (keeps the panel's aspect)
  zoomTo(xu, yu, hw) {
    const W = this.body.clientWidth || 1, H = this.body.clientHeight || 1, a = H / W;
    this.setView({ x0: xu - hw, x1: xu + hw, y0: yu - hw * a, y1: yu + hw * a }, true);
  }

  _wheel(e) {
    if (!this.plot || !this.ext || this.profile) return;
    const r = this.canvas.getBoundingClientRect();
    const xu = this.x0 + (e.clientX - r.left - this.plot.x) / this.k, yu = this.y0 + (e.clientY - r.top - this.plot.y) / this.k;
    const f = Math.exp(Math.max(-1, Math.min(1, e.deltaY * 0.0015)));
    this.setView({ x0: xu - (xu - this.x0) * f, x1: xu + (this.x1 - xu) * f, y0: yu - (yu - this.y0) * f, y1: yu + (this.y1 - yu) * f }, true);
  }

  _pan(e) {
    const d = this._drag;
    if (!d || !(e.buttons & 1) || !this.zoomed() || this.profile) return false;
    const dx = e.clientX - d.x, dy = e.clientY - d.y;
    if (!this._dragged && Math.hypot(dx, dy) < 4) return false;
    this._dragged = true;
    this.setView({ x0: d.v.x0 - dx / this.k, x1: d.v.x1 - dx / this.k, y0: d.v.y0 - dy / this.k, y1: d.v.y1 - dy / this.k }, true);
    return true;
  }

  // particles coloured by group instead of by value (index -1: in no group)
  setCategories(c) { this.categories = c; this._ptsCache = null; this._dirty = true; this.render(); }

  // the camera image under the particles (null for none)
  setBackdrop(b) { this.backdrop = b; this._ptsCache = null; this.render(); }

  // display smoothing: a filter over valid neighbours and/or interpolated pixels
  smoothed() { return this.filter !== "none" || this.smooth; }
  setSmoothing({ filter, smooth }) {
    if (filter !== this.filter) { this.filter = filter; this._shownCache = null; if (this.values) this._range(); this._dirty = true; }
    this.smooth = smooth;
    this._chip();
  }
  _chip() {
    const on = this.smoothed();
    this.smoothBtn.textContent = on ? "Smooth" : "Raw";
    this.smoothBtn.classList.toggle("on", on);
    this.smoothBtn.classList.toggle("hidden", !!this.points);     // particles are drawn one by one
    this.smoothBtn.title = on ? "Smoothed for display (statistics, histograms and exports use the raw pixels). Click for raw pixels."
                              : "Raw pixels. Click to smooth the display (3 × 3 filter, interpolated pixels).";
    this.viewBtn.textContent = this.profile ? "By particle" : "Stage view";
    this.viewBtn.classList.toggle("on", this.profile);
    this.viewBtn.classList.toggle("hidden", !this.points || !!this.opts.noProfile);
    this.viewBtn.title = this.profile ? "Each particle's value against its number, as Q-Carbon shows a particle list. Click for the particles on the stage."
                                      : "The particles at their stage positions. Click for each value against its particle number, as in Q-Carbon.";
  }
  setProfile(on) { this.profile = on; this._chip(); this.render(); }

  // Q-Carbon's view of a particle list: each particle's value against its number
  _renderProfile(ctx, W, H) {
    const n = this.points.x.length, v = this.values, valid = this.valid;
    const P = { x: 52, y: 8, w: Math.max(10, W - 52 - 12), h: Math.max(10, H - 8 - 32) };
    this.plot = P;
    ctx.fillStyle = T.bg; ctx.fillRect(P.x, P.y, P.w, P.h);
    if (!v) return;
    const lo = this.vmin, hi = this.vmax, X = i => P.x + (n > 1 ? i / (n - 1) : 0.5) * P.w, Y = y => P.y + (1 - (y - lo) / ((hi - lo) || 1)) * P.h;
    ctx.font = font(11); ctx.textBaseline = "middle"; ctx.textAlign = "right";
    for (const t of niceTicks(lo, hi, 5)) { ctx.fillStyle = T.grid; ctx.fillRect(P.x, Y(t), P.w, 1); ctx.fillStyle = T.ink; ctx.fillText(fmt(t, 3), P.x - 5, Y(t)); }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of niceTicks(1, n, 6)) { if (t < 1 || t > n) continue; ctx.fillStyle = T.grid; ctx.fillRect(X(t - 1), P.y, 1, P.h); ctx.fillStyle = T.ink; ctx.fillText(fmt(t), X(t - 1), P.y + P.h + 4); }
    ctx.fillStyle = T.label; ctx.fillText("Particle", P.x + P.w / 2, P.y + P.h + 17);
    if (this.label) { ctx.save(); ctx.translate(13, P.y + P.h / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle"; ctx.fillText(this.label + (this.unit ? ` (${this.unit})` : ""), 0, 0); ctx.restore(); }
    ctx.save(); ctx.beginPath(); ctx.rect(P.x, P.y, P.w, P.h); ctx.clip();
    ctx.strokeStyle = T.fit; ctx.lineWidth = 1; ctx.beginPath();
    let pen = false;
    for (let i = 0; i < n; i++) {
      const y = v[i];
      if (!Number.isFinite(y) || (valid && !valid[i])) { pen = false; continue; }
      if (pen) ctx.lineTo(X(i), Y(y)); else { ctx.moveTo(X(i), Y(y)); pen = true; }
    }
    ctx.stroke();
    for (const [p, col] of [[this.cursor, T.hover], [this.pinned, T.pin]]) if (p != null) { ctx.strokeStyle = col; ctx.beginPath(); ctx.moveTo(X(p) + 0.5, P.y); ctx.lineTo(X(p) + 0.5, P.y + P.h); ctx.stroke(); }
    ctx.restore();
    ctx.strokeStyle = T.frame; ctx.strokeRect(P.x - 0.5, P.y - 0.5, P.w + 1, P.h + 1);
  }

  setGeometry({ nx, ny, x, y, points = null }) {
    this.nx = nx; this.ny = ny; this.x = x; this.y = y;
    this.points = points; this._ptsCache = null; this._shp = null; this._chip();
    this._dirty = true;
    if (points) {
      // extent of all particles plus a margin; "pixel" size = median diameter
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      const sh = this._shapes();
      for (let p = 0; p < points.x.length; p++) {
        const r = sh.a[p];
        x0 = Math.min(x0, points.x[p] - r); x1 = Math.max(x1, points.x[p] + r);
        y0 = Math.min(y0, points.y[p] - r); y1 = Math.max(y1, points.y[p] + r);
      }
      const m = 0.03 * Math.max(x1 - x0, y1 - y0, 1);
      Object.assign(this, { x0: x0 - m, x1: x1 + m, y0: y0 - m, y1: y1 + m, xAsc: true, yAsc: true });
      this.ext = this.view();
      const d = Float64Array.from(points.ecd).sort();
      this.dx = this.dy = d[Math.floor(d.length / 2)] || 1;
      const k = 2048 / Math.max(this.x1 - this.x0, this.y1 - this.y0);   // overlay buffer, ~2k px across
      this.off.width = Math.max(1, Math.round((this.x1 - this.x0) * k));
      this.off.height = Math.max(1, Math.round((this.y1 - this.y0) * k));
      return;
    }
    this.off.width = nx; this.off.height = ny;
    this.dx = nx > 1 ? Math.abs(x[1] - x[0]) : 1;
    this.dy = ny > 1 ? Math.abs(y[1] - y[0]) : 1;
    this.xAsc = nx < 2 || x[nx - 1] > x[0];
    this.yAsc = ny < 2 || y[ny - 1] > y[0];
    this.x0 = Math.min(x[0], x[nx - 1]) - this.dx / 2; this.x1 = Math.max(x[0], x[nx - 1]) + this.dx / 2;
    this.y0 = Math.min(y[0], y[ny - 1]) - this.dy / 2; this.y1 = Math.max(y[0], y[ny - 1]) + this.dy / 2;
    this.ext = this.view();
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
    if (!v || this.filter === "none" || this.points) return v;
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

  // Each particle's outline: an ellipse with the particle's own area, the
  // aspect of its Feret width over its Feret length, tilted as the reader found
  // (pfAngles); a circle of its diameter where a file carries no shape. Semi
  // axes a >= b in um, and the tilt t with its cosine and sine.
  _shapes() {
    const pt = this.points;
    if (this._shp && this._shp.pt === pt) return this._shp;
    const n = pt.x.length, a = new Float32Array(n), b = new Float32Array(n), t = pt.angle || new Float32Array(n), c = new Float32Array(n), s = new Float32Array(n);
    for (let p = 0; p < n; p++) {
      const r = pt.ecd[p] / 2, L = pt.feret_max ? pt.feret_max[p] : 0, D = pt.min_diam ? pt.min_diam[p] : 0;
      const q = L > 0 && D > 0 ? Math.sqrt(Math.min(1, Math.max(0.05, D / L))) : 1;
      a[p] = r / q; b[p] = r * q;
      c[p] = Math.cos(t[p]); s[p] = Math.sin(t[p]);
    }
    return (this._shp = { pt, a, b, t, c, s });
  }

  // How big to draw the particles when many are in view and each is only a few screen
  // pixels across, so their colours can still be compared. Sizes are compressed about a
  // readable size R0 (a sixth of the typical spacing between particles, 2.5 to 4 px): a
  // particle of the median size is drawn at R0, others at R0 x sqrt(size / median), so
  // the bigger still look bigger, up to 2 x R0; nothing is drawn smaller than it is. Null
  // when the median particle is already that big. The scale bar says when this is on.
  _boostFor(P) {
    const pt = this.points, sh = this._shapes(), r = [];
    for (let p = 0; p < pt.x.length; p++)
      if (pt.x[p] >= this.x0 && pt.x[p] <= this.x1 && pt.y[p] >= this.y0 && pt.y[p] <= this.y1) r.push(Math.sqrt(sh.a[p] * sh.b[p]));
    if (!r.length) return null;
    r.sort((u, v) => u - v);
    const R0 = Math.min(4, Math.max(2.5, Math.sqrt(P.w * P.h / r.length) / 6)), rmed = r[r.length >> 1] * this.k;
    return rmed > 0 && rmed < R0 ? { R0, rmed } : null;
  }
  // the factor on particle p's own size when drawn on screen
  _grow(p) {
    const b = this._boost; if (!b || this.profile) return 1;
    const sh = this._shapes(), rt = Math.sqrt(sh.a[p] * sh.b[p]) * this.k;
    if (!(rt > 0)) return 1;
    return Math.max(rt, Math.min(b.R0 * Math.sqrt(rt / b.rmed), 2 * b.R0)) / rt;
  }

  // particles as filled shapes: invalid ones as faint outlines underneath, valid
  // ones in increasing value so the high values sit on top. clip: [x0, y0, x1, y1]
  // in um, to leave out particles that cannot show; grow: particle -> factor on its size
  _paintPoints(g, k, ox, oy, minR, origin = this, labels = false, clip = null, grow = null) {
    const { points: pt, valid } = this, v = this.shown(), L = lut(this.cmap), span = this.vmax - this.vmin;
    const X = p => ox + (pt.x[p] - origin.x0) * k, Y = p => oy + (pt.y[p] - origin.y0) * k;
    const sh = this._shapes(), f = p => (grow ? grow(p) : 1), A = p => Math.max(minR, sh.a[p] * k * f(p)), B = p => Math.max(minR, sh.b[p] * k * f(p));
    const outline = p => { g.beginPath(); g.ellipse(X(p), Y(p), A(p), B(p), sh.t[p], 0, 2 * Math.PI); };
    const shows = p => !clip || (pt.x[p] + sh.a[p] >= clip[0] && pt.x[p] - sh.a[p] <= clip[2] && pt.y[p] + sh.a[p] >= clip[1] && pt.y[p] - sh.a[p] <= clip[3]);
    const order = [];
    for (let p = 0; p < pt.x.length; p++) if ((!valid || valid[p]) && shows(p)) order.push(p);
    order.sort((a, b) => v[a] - v[b]);
    g.lineWidth = 1; g.strokeStyle = T.hatchLine;
    if (valid) for (let p = 0; p < pt.x.length; p++) {
      if (valid[p] || !shows(p)) continue;
      outline(p); g.stroke();
    }
    const C = this.categories, hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
    for (const p of order) {
      let rgb;
      if (C) {
        const c = C.index[p];
        if (c < 0) { g.lineWidth = 1; g.strokeStyle = T.hatchLine; outline(p); g.stroke(); continue; }
        g.fillStyle = C.colours[c % C.colours.length]; rgb = hex(g.fillStyle);
      } else {
        if (!Number.isFinite(v[p])) continue;
        const t = Math.max(0, Math.min(255, Math.round((v[p] - this.vmin) / span * 255))) * 3;
        rgb = [L[t], L[t + 1], L[t + 2]]; g.fillStyle = `rgb(${rgb})`;
      }
      outline(p); g.fill();
      // a rim once the shape is big enough to read, so touching particles stay apart: darker
      // than the fill, or lighter on the darkest colours (which would vanish on the camera image)
      if (B(p) >= 2.5) {
        const dark = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 70;
        g.lineWidth = Math.min(1.5, 0.25 * B(p)); g.strokeStyle = dark ? "rgba(255,255,255,.6)" : "rgba(0,0,0,.32)"; g.stroke();
      }
    }
    if (!labels) return;
    // particle numbers once few enough are in view to read them
    const P = this.plot, inView = [];
    for (let p = 0; p < pt.x.length; p++) { const xx = X(p), yy = Y(p); if (xx >= P.x && xx <= P.x + P.w && yy >= P.y && yy <= P.y + P.h) inView.push(p); }
    if (inView.length > Math.min(160, P.w * P.h / 900)) return;      // about one number per 30 x 30 px
    g.font = font(11, "bold"); g.textAlign = "left"; g.textBaseline = "middle";
    g.lineWidth = 3; g.strokeStyle = T.paper; g.fillStyle = T.strong;
    for (const p of inView) {
      // just past the shape's right-hand edge
      const ex = Math.hypot(A(p) * sh.c[p], B(p) * sh.s[p]), tx = X(p) + ex + 2, ty = Y(p);
      g.strokeText(String(p + 1), tx, ty); g.fillText(String(p + 1), tx, ty);
    }
  }

  _paintImage() {
    if (this.points) {
      const g = this.off.getContext("2d"), k = this.off.width / (this.ext.x1 - this.ext.x0);
      g.clearRect(0, 0, this.off.width, this.off.height);
      this._paintPoints(g, k, 0, 0, 2, this.ext);
      return;
    }
    const { nx, ny, valid } = this, v = this.shown();
    const g = this.off.getContext("2d");
    const img = g.createImageData(nx, ny);
    const L = lut(this.cmap), span = this.vmax - this.vmin;
    // pixels coloured by group (hex colours); index -1 is left clear
    const C = this.categories, rgb = C && C.colours.map(h => [1, 3, 5].map(k => parseInt(h.slice(k, k + 2), 16)));
    for (let r = 0; r < ny; r++) {
      const i = this.yAsc ? r : ny - 1 - r;
      for (let c = 0; c < nx; c++) {
        const j = this.xAsc ? c : nx - 1 - c, p = i * nx + j, o = (r * nx + c) * 4;
        if (C) {
          const k = C.index[p];
          if (k < 0) { img.data[o + 3] = 0; continue; }
          const q = rgb[k % rgb.length];
          img.data[o] = q[0]; img.data[o + 1] = q[1]; img.data[o + 2] = q[2]; img.data[o + 3] = 255;
          continue;
        }
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
    if (this.points && this.profile) { this._renderProfile(ctx, W, H); return; }
    const { l: padL, b: padB, t: padT, cb: cbW, cbPad } = PAD;
    const aw = W - padL - cbW - cbPad - 8, ah = H - padB - padT;
    const wu = this.x1 - this.x0, hu = this.y1 - this.y0;
    this.k = Math.max(1e-6, Math.min(aw / wu, ah / hu));        // a large field in a small card needs well under 0.1 px/um
    const pw = wu * this.k, ph = hu * this.k;
    this.plot = { x: padL + (aw - pw) / 2, y: padT + (ah - ph) / 2, w: pw, h: ph };
    const P = this.plot;

    ctx.save(); ctx.beginPath(); ctx.rect(P.x, P.y, P.w, P.h); ctx.clip();
    if (this.points) {
      // particles drawn at screen resolution over the camera image (greyscale, so it
      // cannot be read as part of the colour scale), cached until the data, size or view changes
      ctx.fillStyle = T.hatchBg; ctx.fillRect(P.x, P.y, P.w, P.h);
      if (this.values) {
        if (this._dirty) { this._paintImage(); this._dirty = false; this._ptsCache = null; }
        const B = this.backdropOn && this.backdrop;
        const dpr = this.canvas.width / W, key = `${W}x${H}:${this.x0}:${this.x1}:${this.y0}:${this.y1}:${!!B}:${T.paper}`;
        if (!this._ptsCache || this._ptsCache.key !== key) {
          const c = document.createElement("canvas"); c.width = this.canvas.width; c.height = this.canvas.height;
          const g = c.getContext("2d"); g.scale(dpr, dpr);
          if (B) {
            const dxv = (B.x[1] - B.x[0]) / (B.w - 1), dyv = (B.y[1] - B.y[0]) / (B.h - 1);
            g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
            g.globalAlpha = 0.3; g.filter = "grayscale(1)"; g.imageSmoothingEnabled = true;
            g.translate(this.sx(B.x[0] - dxv / 2), this.sy(B.y[0] - dyv / 2)); g.scale(dxv * this.k, dyv * this.k);
            g.drawImage(B.img, 0, 0);
            g.restore();
          }
          this._boost = this._boostFor(P);
          this._paintPoints(g, this.k, P.x, P.y, 2.2, this, true, null, q => this._grow(q));
          this._ptsCache = { key, c };
        }
        ctx.drawImage(this._ptsCache.c, 0, 0, W, H);
      }
    } else {
      ctx.fillStyle = ctx.createPattern(HATCH(), "repeat");
      ctx.fillRect(P.x, P.y, P.w, P.h);
      if (this.values) {
        if (this._dirty) { this._paintImage(); this._dirty = false; }
        ctx.imageSmoothingEnabled = this.smooth && !this.categories;
        ctx.imageSmoothingQuality = "high";
        // the part of the map image inside the view (the image spans this.ext)
        const E = this.ext, fx = this.off.width / (E.x1 - E.x0), fy = this.off.height / (E.y1 - E.y0);
        ctx.drawImage(this.off, (this.x0 - E.x0) * fx, (this.y0 - E.y0) * fy, (this.x1 - this.x0) * fx, (this.y1 - this.y0) * fy, P.x, P.y, P.w, P.h);
      }
    }
    for (const o of this.overlays) o(ctx, this);
    if (this.emptyText) {
      // why this map has nothing to show, word-wrapped on a plain box
      ctx.font = font(12); ctx.textAlign = "center"; ctx.textBaseline = "middle";
      const lines = [];
      for (const w of this.emptyText.split(" ")) {
        const l = lines.length ? lines[lines.length - 1] + " " + w : w;
        if (lines.length && ctx.measureText(l).width <= P.w - 32) lines[lines.length - 1] = l; else lines.push(w);
      }
      const lh = 17, bw = Math.min(P.w - 12, Math.max(...lines.map(l => ctx.measureText(l).width)) + 24), bh = lines.length * lh + 14;
      const cx = P.x + P.w / 2, cy = P.y + P.h / 2;
      ctx.fillStyle = T.paper; ctx.fillRect(cx - bw / 2, cy - bh / 2, bw, bh);
      ctx.strokeStyle = T.frame; ctx.strokeRect(cx - bw / 2 + 0.5, cy - bh / 2 + 0.5, bw - 1, bh - 1);
      ctx.fillStyle = T.label;
      lines.forEach((l, i) => ctx.fillText(l, cx, cy - (lines.length - 1) * lh / 2 + i * lh));
    }
    ctx.restore();

    // axes
    ctx.strokeStyle = T.frame; ctx.lineWidth = 1;
    ctx.strokeRect(P.x - .5, P.y - .5, P.w + 1, P.h + 1);
    ctx.fillStyle = T.ink; ctx.font = font(11);
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of niceTicks(this.x0, this.x1, Math.max(2, Math.floor(P.w / 60)))) {
      const X = this.sx(t); ctx.fillRect(X, P.y + P.h, 1, 3); ctx.fillText(fmt(t, 3), X, P.y + P.h + 5);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const t of niceTicks(this.y0, this.y1, Math.max(2, Math.floor(P.h / 45)))) {
      const Y = this.sy(t); ctx.fillRect(P.x - 3, Y, 3, 1); ctx.fillText(fmt(t, 3), P.x - 5, Y);
    }
    ctx.save(); ctx.fillStyle = T.label; ctx.font = font(12);
    ctx.textAlign = "center"; ctx.fillText("X (µm)", P.x + P.w / 2, H - 6);
    ctx.translate(9, P.y + P.h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("Y (µm)", 0, 0); ctx.restore();

    // scale bar above the plot, at the right, so it never covers the data; with how much the
    // particles are enlarged when they are drawn bigger than they are
    const target = wu * 0.22, decade = 10 ** Math.floor(Math.log10(target));
    const len = [1, 2, 5, 10].map(m => m * decade).filter(v => v <= target).pop() || decade;
    if (P.w > 90) {
      const bw = len * this.k, bx = P.x + P.w - bw, by = P.y - 10;
      ctx.fillStyle = T.accent; ctx.fillRect(bx, by, bw, 3);
      ctx.fillRect(bx, by - 2, 1, 7); ctx.fillRect(bx + bw - 1, by - 2, 1, 7);
      ctx.fillStyle = T.ink; ctx.font = font(11); ctx.textAlign = "right"; ctx.textBaseline = "middle";
      // the longest note that fits in the margin to the left of the bar
      const notes = this.points && this._boost ? [" · small particles drawn larger", " · enlarged", ""] : [""];
      const text = notes.map(n => `${fmt(len, 3)} µm${n}`).find(t => ctx.measureText(t).width <= bx - 10) || `${fmt(len, 3)} µm`;
      ctx.fillText(text, bx - 6, by + 1.5);
    }

    // colour bar, or the legend of the groups
    const cx = P.x + P.w + 12, cy = P.y, ch = P.h;
    if (this.categories) this._legend(ctx, cx, cy);
    else {
      const grad = ctx.createLinearGradient(0, cy + ch, 0, cy);
      const L = lut(this.cmap);
      for (let s = 0; s <= 16; s++) { const q = Math.round(s / 16 * 255) * 3; grad.addColorStop(s / 16, `rgb(${L[q]},${L[q + 1]},${L[q + 2]})`); }
      ctx.fillStyle = grad; ctx.fillRect(cx, cy, cbW, ch);
      ctx.strokeStyle = T.frame; ctx.strokeRect(cx - .5, cy - .5, cbW + 1, ch + 1);
      ctx.fillStyle = T.ink; ctx.font = font(11); ctx.textAlign = "left"; ctx.textBaseline = "middle";
      // at least three labels however short the bar (one alone cannot show the scale)
      const n = Math.max(4, Math.floor(ch / 28));
      let ticks = niceTicks(this.vmin, this.vmax, n);
      if (ticks.length < 3) ticks = niceTicks(this.vmin, this.vmax, n + 3);
      for (const t of ticks) {
        const Y = cy + ch - (t - this.vmin) / (this.vmax - this.vmin) * ch;
        ctx.fillRect(cx + cbW, Y, 3, 1); ctx.fillText(fmt(t, 3), cx + cbW + 5, Y);
      }
      if (this.label) {
        ctx.save(); ctx.translate(W - 6, cy + ch / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = T.label; ctx.font = font(12); ctx.textAlign = "center";
        ctx.fillText(this.label + (this.unit ? ` (${this.unit})` : ""), 0, 0); ctx.restore();
      }
    }
    this._drawCursor();
    if (this.resetBtn) this.resetBtn.classList.toggle("hidden", !this.zoomed());
  }

  _legend(ctx, cx, cy) {
    const C = this.categories, row = c => cy + 8 + c * 18;
    ctx.font = font(11, "bold"); ctx.textAlign = "left"; ctx.textBaseline = "middle";
    C.names.forEach((nm, c) => {
      ctx.fillStyle = C.colours[c % C.colours.length]; ctx.beginPath(); ctx.arc(cx + 5, row(c), 5, 0, 7); ctx.fill();
      ctx.fillStyle = T.ink; ctx.fillText(nm, cx + 14, row(c));
    });
    const n = C.names.length;
    ctx.strokeStyle = T.hatchLine; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx + 5, row(n), 4.5, 0, 7); ctx.stroke();
    ctx.fillStyle = T.label; ctx.font = font(10); ctx.fillText("none", cx + 14, row(n));
  }

  idxXY(p) {
    if (this.points) return [this.points.x[p], this.points.y[p]];
    const i = Math.floor(p / this.nx), j = p % this.nx; return [this.x[j], this.y[i]];
  }

  _drawCursor() {
    const ctx = this.ctx, P = this.plot; if (!ctx) return;
    const mark = (p, colour, cross) => {
      if (p == null) return;
      const [xu, yu] = this.idxXY(p), X = this.sx(xu), Y = this.sy(yu);
      ctx.save();
      ctx.beginPath(); ctx.rect(P.x, P.y, P.w, P.h); ctx.clip();
      ctx.strokeStyle = colour; ctx.lineWidth = 1;
      if (cross) {                                   // light and dashed, so the data shows through
        ctx.save(); ctx.globalAlpha = 0.5; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(P.x, Y + .5); ctx.lineTo(P.x + P.w, Y + .5);
        ctx.moveTo(X + .5, P.y); ctx.lineTo(X + .5, P.y + P.h); ctx.stroke();
        ctx.restore();
      }
      ctx.lineWidth = 1.5; ctx.beginPath();
      if (this.points) {                             // a ring just outside the particle's own shape
        const sh = this._shapes();
        const f = this.k * this._grow(p);
        ctx.ellipse(X, Y, Math.max(5, f * sh.a[p] + 3), Math.max(5, f * sh.b[p] + 3), sh.t[p], 0, 2 * Math.PI);
      } else ctx.arc(X, Y, Math.max(3, this.k * this.dx * 0.6), 0, 7);
      ctx.stroke();
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
    if (this.points && this.profile) {
      const P = this.plot, n = this.points.x.length, f = (e.clientX - r.left - P.x) / P.w;
      return f < -0.01 || f > 1.01 ? null : Math.max(0, Math.min(n - 1, Math.round(f * (n - 1))));
    }
    const xu = this.x0 + (e.clientX - r.left - this.plot.x) / this.k;
    const yu = this.y0 + (e.clientY - r.top - this.plot.y) / this.k;
    if (xu < this.x0 || xu > this.x1 || yu < this.y0 || yu > this.y1) return null;
    if (this.points) {
      // the smallest particle whose shape the pointer is on, else the nearest
      // centre within a few screen pixels
      const pt = this.points, sh = this._shapes();
      let best = -1, bd = Infinity, hit = -1, ha = Infinity;
      for (let p = 0; p < pt.x.length; p++) {
        const ex = xu - pt.x[p], ey = yu - pt.y[p], d = ex * ex + ey * ey;
        if (d < bd) { bd = d; best = p; }
        const g = this._grow(p), u = (ex * sh.c[p] + ey * sh.s[p]) / (sh.a[p] * g), w = (ey * sh.c[p] - ex * sh.s[p]) / (sh.b[p] * g);
        if (u * u + w * w <= 1 && sh.a[p] * sh.b[p] < ha) { ha = sh.a[p] * sh.b[p]; hit = p; }
      }
      if (hit >= 0) return hit;
      return best >= 0 && Math.sqrt(bd) <= 8 / this.k ? best : null;
    }
    let j = Math.floor((xu - this.ext.x0) / this.dx), i = Math.floor((yu - this.ext.y0) / this.dy);
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
    const who = this.points ? `#${p + 1} · ` : "";
    const C = this.categories;
    const grp = !C ? "" : this.points ? ` · ${C.index[p] < 0 ? "in no population" : `population ${C.names[C.index[p]]}`}`
      : ` · ${C.index[p] < 0 ? "not analysed" : C.names[C.index[p]]}`;
    this.readout.textContent = `${who}${fmt(xu, 3)}, ${fmt(yu, 3)} µm  →  ${bad ? "invalid" : fmt(v) + filt}${grp}`;
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
    const smooth = el("input", { type: "checkbox", checked: this.smooth, onchange: () => {
      this.smooth = smooth.checked; this._chip(); this.render(); this.opts.onSmoothChange && this.opts.onSmoothChange(this);
    } });
    const filter = el("select", { title: "Filter the map over its valid neighbours. Display and PNG only: statistics, histograms and exports use the raw pixels." },
      ...[["none", "None"], ["gauss3", "Gaussian 3 × 3"], ["gauss5", "Gaussian 5 × 5"], ["median3", "Median 3 × 3"], ["median5", "Median 5 × 5"]]
        .map(([v, t]) => el("option", { value: v, selected: v === this.filter }, t)));
    filter.onchange = () => {
      this.filter = filter.value; this._range(); this._dirty = true; this._chip(); this.render();
      this.opts.onSmoothChange && this.opts.onSmoothChange(this); this.opts.onChange && this.opts.onChange(this);
    };
    const pop = el("div", { class: "popover", onclick: ev => ev.stopPropagation() },
      el("div", { class: "pop-h" }, "Colour map"), sel,
      el("div", { class: "pop-h" }, "Scale"), mode, el("div", { class: "row2" }, lo, hi),
      el("div", { class: "pop-h" }, "Filter ", el("span", { class: "muted" }, "(display only, stats use raw pixels)")), filter,
      el("label", { class: "chk" }, smooth, " Interpolate pixels ", el("span", { class: "muted" }, "(display only)")));
    if (this.points && this.backdrop) {
      const under = el("input", { type: "checkbox", checked: this.backdropOn, onchange: () => {
        this.backdropOn = under.checked; this._ptsCache = null; this.render(); this.opts.onBackdrop && this.opts.onBackdrop(under.checked);
      } });
      pop.append(el("label", { class: "chk" }, under, " Camera image underneath ", el("span", { class: "muted" }, "(greyscale)")));
    }
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
