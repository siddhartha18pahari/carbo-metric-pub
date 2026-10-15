// Particle segmentation: threshold a map, label 4-connected regions, drop
// specks. Neighbouring pixels of one particle are not independent samples
// (an earlier map had ~30 effective samples among 379 pixels), so the
// particle, not the pixel, is the unit to compare between samples.

function components(inside, nx, ny) {
  const N = nx * ny, labels = new Int32Array(N), stack = new Int32Array(N), groups = [];
  let next = 1;
  for (let s = 0; s < N; s++) {
    if (labels[s] || !inside[s]) continue;
    let top = 0; stack[top++] = s; labels[s] = next;
    const px = [];
    while (top) {
      const p = stack[--top]; px.push(p);
      const i = (p / nx) | 0, j = p - i * nx;
      if (j > 0 && !labels[p - 1] && inside[p - 1]) { labels[p - 1] = next; stack[top++] = p - 1; }
      if (j < nx - 1 && !labels[p + 1] && inside[p + 1]) { labels[p + 1] = next; stack[top++] = p + 1; }
      if (i > 0 && !labels[p - nx] && inside[p - nx]) { labels[p - nx] = next; stack[top++] = p - nx; }
      if (i < ny - 1 && !labels[p + nx] && inside[p + nx]) { labels[p + nx] = next; stack[top++] = p + nx; }
    }
    groups.push(px); next++;
  }
  return { labels, groups };
}

// Chamfer (3-4) distance to the nearest outside pixel, in pixels.
function distance(inside, nx, ny) {
  const N = nx * ny, d = new Float32Array(N);
  for (let p = 0; p < N; p++) d[p] = inside[p] ? 1e9 : 0;
  const at = (i, j) => (i < 0 || j < 0 || i >= ny || j >= nx) ? 0 : d[i * nx + j];
  for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
    const p = i * nx + j; if (!d[p]) continue;
    d[p] = Math.min(d[p], at(i, j - 1) + 3, at(i - 1, j) + 3, at(i - 1, j - 1) + 4, at(i - 1, j + 1) + 4);
  }
  for (let i = ny - 1; i >= 0; i--) for (let j = nx - 1; j >= 0; j--) {
    const p = i * nx + j; if (!d[p]) continue;
    d[p] = Math.min(d[p], at(i, j + 1) + 3, at(i + 1, j) + 3, at(i + 1, j + 1) + 4, at(i + 1, j - 1) + 4);
  }
  for (let p = 0; p < N; p++) d[p] /= 3;
  return d;
}

// Touching particles: each one has its own peak in the distance-to-edge map
// (the centre of its largest inscribed disc). A peak is kept as a new particle
// only if it lies outside every accepted particle's disc by at least half its
// own radius: two round grains in contact pass (centres ~ r1 + r2 apart), the
// ridge maxima along one irregular grain do not. Pixels are then grown back
// from the peaks in order of decreasing depth (a marker watershed). A depth
// threshold cannot do this: on one test map the neck is 14.6 px deep against
// 17 and 21.5 px at the two centres.
function splitTouching(inside, nx, ny) {
  const d = distance(inside, nx, ny), N = nx * ny;
  const cand = [];
  for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
    const p = i * nx + j; if (d[p] < 3) continue;
    let mx = true;
    for (let a = -2; a <= 2 && mx; a++) for (let c = -2; c <= 2; c++) {
      const ii = i + a, jj = j + c;
      if ((a || c) && ii >= 0 && jj >= 0 && ii < ny && jj < nx && d[ii * nx + jj] > d[p]) { mx = false; break; }
    }
    if (mx) cand.push([i, j, d[p]]);
  }
  cand.sort((a, b) => b[2] - a[2]);
  const seeds = [];
  for (const c of cand) {
    if (seeds.every(s => Math.hypot(s[0] - c[0], s[1] - c[1]) > s[2] + 0.5 * c[2])) seeds.push(c);
  }
  const labels = new Int32Array(N);
  seeds.forEach((s, k) => { labels[s[0] * nx + s[1]] = k + 1; });
  const order = [];
  for (let p = 0; p < N; p++) if (inside[p] && !labels[p]) order.push(p);
  order.sort((a, b) => d[b] - d[a]);
  for (let changed = true; changed;) {
    changed = false;
    for (const p of order) {
      if (labels[p]) continue;
      const i = (p / nx) | 0, j = p - i * nx;
      const l = (j > 0 && labels[p - 1]) || (j < nx - 1 && labels[p + 1]) || (i > 0 && labels[p - nx]) || (i < ny - 1 && labels[p + nx]);
      if (l) { labels[p] = l; changed = true; }
    }
  }
  // thin regions with no peak of their own keep their connected components
  const rest = new Uint8Array(N); let any = false;
  for (let p = 0; p < N; p++) if (inside[p] && !labels[p]) { rest[p] = 1; any = true; }
  if (any) { const c = components(rest, nx, ny); for (let p = 0; p < N; p++) if (c.labels[p]) labels[p] = seeds.length + c.labels[p]; }
  return labels;
}

export function segment(values, nx, ny, thr, minPx = 6, split = true) {
  const N = nx * ny, inside = new Uint8Array(N);
  for (let p = 0; p < N; p++) inside[p] = values[p] > thr ? 1 : 0;
  let labels;
  if (split) labels = splitTouching(inside, nx, ny);
  else labels = components(inside, nx, ny).labels;
  const byLabel = new Map();
  for (let p = 0; p < N; p++) if (labels[p]) { if (!byLabel.has(labels[p])) byLabel.set(labels[p], []); byLabel.get(labels[p]).push(p); }
  const parts = [];
  for (const [id, px] of byLabel) {
    if (px.length < minPx) { for (const p of px) labels[p] = 0; continue; }
    parts.push({ id, pixels: px });
  }
  let next = 0; for (const pt of parts) next = Math.max(next, pt.id);
  // renumber by size, largest = 1
  parts.sort((a, b) => b.pixels.length - a.pixels.length);
  const remap = new Int32Array(next + 1);
  parts.forEach((pt, k) => { remap[pt.id] = k + 1; pt.id = k + 1; });
  for (let p = 0; p < N; p++) labels[p] = remap[labels[p]];
  return { labels, parts };
}

export const PARTICLE_COLOURS = ["#3fe0ff", "#ff9f40", "#b18cff", "#7dff6a", "#ff6b9a", "#ffd84d", "#6bb8ff", "#ff7b54"];

// MapView overlay: particle outlines plus their numbers
export function outlineOverlay(seg) {
  return (ctx, mv) => {
    if (!seg) return;
    const { labels } = seg, nx = mv.nx, ny = mv.ny, hx = mv.dx / 2, hy = mv.dy / 2;
    ctx.save(); ctx.lineWidth = 1.4;
    const byLabel = new Map();
    const seg2 = (lab, x0, y0, x1, y1) => { if (!byLabel.has(lab)) byLabel.set(lab, []); byLabel.get(lab).push(x0, y0, x1, y1); };
    for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
      const p = i * nx + j, l = labels[p]; if (!l) continue;
      const xu = mv.x[j], yu = mv.y[i];
      if (j === nx - 1 || labels[p + 1] !== l) seg2(l, xu + Math.sign(mv.x[nx - 1] - mv.x[0] || 1) * hx, yu - hy, xu + Math.sign(mv.x[nx - 1] - mv.x[0] || 1) * hx, yu + hy);
      if (j === 0 || labels[p - 1] !== l) seg2(l, xu - Math.sign(mv.x[nx - 1] - mv.x[0] || 1) * hx, yu - hy, xu - Math.sign(mv.x[nx - 1] - mv.x[0] || 1) * hx, yu + hy);
      if (i === ny - 1 || labels[p + nx] !== l) seg2(l, xu - hx, yu + Math.sign(mv.y[ny - 1] - mv.y[0] || 1) * hy, xu + hx, yu + Math.sign(mv.y[ny - 1] - mv.y[0] || 1) * hy);
      if (i === 0 || labels[p - nx] !== l) seg2(l, xu - hx, yu - Math.sign(mv.y[ny - 1] - mv.y[0] || 1) * hy, xu + hx, yu - Math.sign(mv.y[ny - 1] - mv.y[0] || 1) * hy);
    }
    for (const [l, s] of byLabel) {
      ctx.strokeStyle = PARTICLE_COLOURS[(l - 1) % PARTICLE_COLOURS.length];
      ctx.beginPath();
      for (let k = 0; k < s.length; k += 4) { ctx.moveTo(mv.sx(s[k]), mv.sy(s[k + 1])); ctx.lineTo(mv.sx(s[k + 2]), mv.sy(s[k + 3])); }
      ctx.stroke();
    }
    ctx.font = "bold 11px var(--sans, sans-serif)"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (const pt of seg.parts) {
      let sx = 0, sy = 0;
      for (const p of pt.pixels) { const [xu, yu] = mv.idxXY(p); sx += xu; sy += yu; }
      const X = mv.sx(sx / pt.pixels.length), Y = mv.sy(sy / pt.pixels.length);
      ctx.fillStyle = "rgba(0,0,0,.55)"; ctx.beginPath(); ctx.arc(X, Y, 8, 0, 7); ctx.fill();
      ctx.fillStyle = PARTICLE_COLOURS[(pt.id - 1) % PARTICLE_COLOURS.length]; ctx.fillText(pt.id, X, Y + .5);
    }
    ctx.restore();
  };
}
