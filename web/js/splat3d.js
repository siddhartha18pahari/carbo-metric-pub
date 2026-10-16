// WebGL2 viewer for a spectral Gaussian field V(x, y, nu).
//
// Each primitive is a 3-D Gaussian in (stage x, stage y, Raman shift). It is
// drawn as in 3-D Gaussian splatting (Kerbl et al. 2023): the 3-D covariance
// is projected to a 2-D screen-space ellipse through the local Jacobian of
// the perspective projection. Because the field is a SUM of Gaussians (an
// emission density, not opaque blobs), splats are blended additively with a
// weight equal to the Gaussian's line integral along the viewing ray,
// a*sqrt(2*pi / (d' S^-1 d)), into a float buffer, then tone-mapped. What you
// see is therefore the projection of the fitted field itself, no sorting needed.
//
// Display frame: X = stage x, Z = stage y (towards the viewer, so Y-down on
// the instrument reads correctly from the default view), height = Raman shift.

import { rgbAt } from "./colormaps.js";
import { DARK_PEAK_COLOURS as PEAK_COLOURS, font } from "./theme.js";

const VS_SPLAT = `#version 300 es
precision highp float;
layout(location=0) in vec2 corner;
layout(location=1) in vec3 center;
layout(location=2) in vec3 covA;
layout(location=3) in vec3 covB;
layout(location=4) in vec4 color;
uniform mat4 view, proj;
uniform vec2 viewport;
uniform float sliceY, sliceOn;
out vec3 vColor; out vec2 vOff; out vec3 vConic; out float vW;
void main(){
  vec4 cam = view * vec4(center, 1.0);
  float z = -cam.z;
  if (z < 0.02) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  mat3 S = mat3(covA.x, covA.y, covA.z,  covA.y, covB.x, covB.y,  covA.z, covB.y, covB.z);
  mat3 R = mat3(view);
  mat3 Sc = R * S * transpose(R);
  float fx = proj[0][0] * viewport.x * 0.5, fy = proj[1][1] * viewport.y * 0.5;
  mat3 J = mat3(fx / z, 0.0, 0.0,  0.0, fy / z, 0.0,  fx * cam.x / (z * z), fy * cam.y / (z * z), 0.0);
  mat3 C = J * Sc * transpose(J);
  float a = C[0][0] + 0.3, b = C[0][1], c = C[1][1] + 0.3;
  float det = a * c - b * b;
  if (det <= 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  float mid = 0.5 * (a + c);
  float lam = mid + sqrt(max(0.1, mid * mid - det));
  float r = ceil(3.0 * sqrt(lam));
  vConic = vec3(c, -b, a) / det;
  vOff = corner * r;
  vec3 d = normalize(cam.xyz);
  float q = dot(d, inverse(Sc) * d);
  float w = color.a * sqrt(6.2831853 / max(q, 1e-12));
  if (sliceOn > 0.5) {
    float sy = S[1][1];
    w *= 0.4 + 0.6 * exp(-0.5 * (center.y - sliceY) * (center.y - sliceY) / max(sy, 1e-6));
  }
  vW = w; vColor = color.rgb;
  vec4 clip = proj * cam;
  gl_Position = vec4(clip.xy / clip.w + vOff * 2.0 / viewport, 0.0, 1.0);
}`;

const FS_SPLAT = `#version 300 es
precision highp float;
in vec3 vColor; in vec2 vOff; in vec3 vConic; in float vW;
uniform float exposure;
out vec4 o;
void main(){
  float p = -0.5 * (vConic.x * vOff.x * vOff.x + 2.0 * vConic.y * vOff.x * vOff.y + vConic.z * vOff.y * vOff.y);
  if (p < -4.5) discard;
  float g = exp(p) * vW * exposure;
  o = vec4(vColor * g, g);
}`;

const VS_QUAD = `#version 300 es
layout(location=0) in vec2 p; out vec2 uv;
void main(){ uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }`;

const FS_TONE = `#version 300 es
precision highp float;
in vec2 uv; uniform sampler2D hdr; out vec4 o;
void main(){
  vec4 h = texture(hdr, uv);
  vec3 c = vec3(1.0) - exp(-h.rgb);
  c = pow(c, vec3(0.9));
  o = vec4(c, 1.0);
}`;

const VS_GEO = `#version 300 es
layout(location=0) in vec3 pos; layout(location=1) in vec2 tex;
uniform mat4 view, proj; out vec2 uv;
void main(){ uv = tex; gl_Position = proj * view * vec4(pos, 1.0); }`;

const FS_GEO = `#version 300 es
precision highp float;
in vec2 uv; uniform sampler2D img; uniform vec4 col; uniform float useTex; out vec4 o;
void main(){ o = useTex > 0.5 ? vec4(texture(img, uv).rgb * col.rgb, col.a) : col; }`;

function compile(gl, vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {}; const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const name = gl.getActiveUniform(p, i).name; u[name] = gl.getUniformLocation(p, name); }
  return { p, u };
}

// column-major 4x4 helpers
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}
function lookAt(e, t, up) {
  const z = norm(sub(e, t)), x = norm(cross(up, z)), y = cross(z, x);
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
    -dot(x, e), -dot(y, e), -dot(z, e), 1]);
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
function mulv(m, v) {
  return [0, 1, 2, 3].map(r => m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3]);
}

export class SplatView {
  constructor(host) {
    this.host = host;
    this.canvas = document.createElement("canvas");
    this.labels = document.createElement("canvas");
    this.labels.className = "splat-labels";
    host.append(this.canvas, this.labels);
    const gl = this.canvas.getContext("webgl2", { antialias: true, premultipliedAlpha: false });
    if (!gl) { host.append(Object.assign(document.createElement("div"), { className: "empty", textContent: "WebGL2 is not available in this browser." })); return; }
    this.gl = gl;
    this.hasFloat = !!gl.getExtension("EXT_color_buffer_float");
    this.progSplat = compile(gl, VS_SPLAT, FS_SPLAT);
    this.progTone = compile(gl, VS_QUAD, FS_TONE);
    this.progGeo = compile(gl, VS_GEO, FS_GEO);
    this.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    this.tri = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.tri);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    this.inst = gl.createBuffer();
    this.geo = gl.createBuffer();
    this.n = 0;
    this.yaw = -0.75; this.pitch = 0.36; this.dist = 3.1; this.target = [0, 0.45, 0];
    this.exposure = 1; this.depth = 1.0; this.colorBy = "band"; this.autoRotate = false; this.equalise = true;
    this.slice = null; this.showFloor = true;
    this._bind();
    new ResizeObserver(() => this.draw()).observe(host);
  }

  _bind() {
    let drag = null;
    this.canvas.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; this.canvas.setPointerCapture(e.pointerId); });
    this.canvas.addEventListener("pointermove", e => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan) {
        const s = this.dist * 0.0016, cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
        this.target[0] -= (dx * cy) * s; this.target[2] += (dx * sy) * s; this.target[1] += dy * s;
      } else {
        this.yaw -= dx * 0.008; this.pitch = Math.max(-0.2, Math.min(1.5, this.pitch + dy * 0.008));
      }
      this.draw();
    });
    this.canvas.addEventListener("pointerup", () => { drag = null; });
    this.canvas.addEventListener("contextmenu", e => e.preventDefault());
    this.canvas.addEventListener("wheel", e => { e.preventDefault(); this.dist = Math.max(0.5, Math.min(12, this.dist * Math.exp(e.deltaY * 0.0012))); this.draw(); }, { passive: false });
    this.canvas.addEventListener("dblclick", () => this.resetView());
  }

  resetView() { this.yaw = -0.75; this.pitch = 0.36; this.dist = 3.1; this.target = [0, this.depth * 0.45, 0]; this.draw(); }

  // scene: { field:{mu,cov,amp}, peaks:[{name,pos,width}], wnRange, mapRect:[x0,x1,y0,y1], video:{img,x,y} }
  setScene(scene) {
    this.scene = scene;
    const [x0, x1, y0, y1] = scene.mapRect;
    this.cx = (x0 + x1) / 2; this.cy = (y0 + y1) / 2; this.L = Math.max(x1 - x0, y1 - y0);
    if (scene.video && scene.video.img && scene.video.img !== this._vidImg) this._uploadVideo(scene.video.img);
    this.setField(scene.field);
    this._buildGeo();
    this.draw();
  }

  toDisp(x, y, nu) {
    const [lo, hi] = this.scene.wnRange;
    return [(x - this.cx) / this.L, (nu - lo) / (hi - lo) * this.depth, (y - this.cy) / this.L];
  }

  setDepth(d) { this.depth = d; if (this.scene) { this.setField(this.scene.field); this._buildGeo(); } this.draw(); }

  setField(field) {
    if (!field || !this.gl) { this.n = 0; return; }
    this.scene.field = field;
    const { gl } = this, n = field.mu.length;
    const [lo, hi] = this.scene.wnRange;
    const sX = 1 / this.L, sY = this.depth / (hi - lo);
    const data = new Float32Array(n * 13);
    const peaks = this.scene.peaks || [];
    // nearest band for each primitive (colour, and per-band brightness when
    // equalised: G is several times brighter than the D bands and would hide them)
    const band = new Int32Array(n), near = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let best = -1, bd = Infinity;
      peaks.forEach((pk, j) => { const d = Math.abs(field.mu[i][2] - pk.pos) / Math.max(pk.width, 15); if (d < bd) { bd = d; best = j; } });
      band[i] = best; near[i] = bd;
    }
    const amaxBand = new Map(); let amax = 0;
    for (let i = 0; i < n; i++) { amax = Math.max(amax, field.amp[i]); amaxBand.set(band[i], Math.max(amaxBand.get(band[i]) || 0, field.amp[i])); }
    for (let i = 0; i < n; i++) {
      const [x, y, nu] = field.mu[i], c = field.cov[i];
      // stage (x, y, nu) -> display (X, H, Z) = (x*sX, nu*sY, y*sX): permute + scale
      const s = [sX, sX, sY], perm = [0, 2, 1];   // display axis k takes stage axis perm[k]
      const M = (a, b) => c[perm[a] * 3 + perm[b]] * s[perm[a]] * s[perm[b]];
      const p = this.toDisp(x, y, nu);
      let rgb;
      if (this.colorBy === "band" && peaks.length) {
        const h = PEAK_COLOURS[Math.max(0, band[i]) % PEAK_COLOURS.length];
        rgb = [1, 3, 5].map(k => parseInt(h.slice(k, k + 2), 16) / 255);
        if (near[i] > 1.5) rgb = rgb.map(v => v * 0.35 + 0.25);
      } else {
        rgb = rgbAt("Turbo", (nu - lo) / (hi - lo)).map(v => v / 255);
      }
      data.set([p[0], p[1], p[2], M(0, 0), M(0, 1), M(0, 2), M(1, 1), M(1, 2), M(2, 2),
        rgb[0], rgb[1], rgb[2], field.amp[i] / ((this.equalise ? amaxBand.get(band[i]) : amax) || 1)], i * 13);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.inst);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    this.n = n;
    // auto exposure: typical line-integral weight is ~ amp * sigma; normalise
    let sig = 0; for (let i = 0; i < n; i++) sig += Math.sqrt(Math.max(data[i * 13 + 6], 1e-9));
    this.baseExposure = 0.9 / Math.max(1e-6, sig / n * 2.5);
  }

  _uploadVideo(img) {
    const gl = this.gl; this._vidImg = img;
    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  _buildGeo() {
    const S = this.scene, verts = [], segs = [];
    const [x0, x1, y0, y1] = S.mapRect, [lo, hi] = S.wnRange;
    const push = (arr, x, y, nu, u = 0, v = 0) => { const p = this.toDisp(x, y, nu); arr.push(p[0], p[1], p[2], u, v); };
    // floor: the white-light image around the map (map size +/- 60 %)
    this.floorCount = 0;
    if (S.video) {
      const V = S.video, mx = (x1 - x0) * 0.6, my = (y1 - y0) * 0.6;
      const vx0 = Math.min(V.x[0], V.x[1]), vx1 = Math.max(V.x[0], V.x[1]);
      const vy0 = Math.min(V.y[0], V.y[1]), vy1 = Math.max(V.y[0], V.y[1]);
      const fx0 = Math.max(vx0, x0 - mx), fx1 = Math.min(vx1, x1 + mx), fy0 = Math.max(vy0, y0 - my), fy1 = Math.min(vy1, y1 + my);
      const u = x => (x - V.x[0]) / (V.x[1] - V.x[0]), v = y => (y - V.y[0]) / (V.y[1] - V.y[0]);
      for (const [x, y] of [[fx0, fy0], [fx1, fy0], [fx0, fy1], [fx1, fy0], [fx1, fy1], [fx0, fy1]]) push(verts, x, y, lo, u(x), v(y));
      this.floorCount = 6;
    }
    // map footprint, the box, and tick marks up the spectral axis
    const L = [];
    const line = (a, b) => { push(L, ...a); push(L, ...b); };
    for (const nu of [lo, hi]) { line([x0, y0, nu], [x1, y0, nu]); line([x1, y0, nu], [x1, y1, nu]); line([x1, y1, nu], [x0, y1, nu]); line([x0, y1, nu], [x0, y0, nu]); }
    for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) line([x, y, lo], [x, y, hi]);
    this.ticks = [];
    for (let nu = Math.ceil(lo / 100) * 100; nu <= hi; nu += 100) {
      line([x0, y1, nu], [x0 - (x1 - x0) * 0.03, y1, nu]);
      this.ticks.push(nu);
    }
    this.lineStart = verts.length / 5; this.lineCount = L.length / 5;
    segs.push(...L);
    // slice plane (two triangles)
    this.sliceStart = (verts.length + segs.length) / 5;
    const nuS = this.slice ?? lo;
    for (const [x, y] of [[x0, y0], [x1, y0], [x0, y1], [x1, y0], [x1, y1], [x0, y1]]) push(segs, x, y, nuS);
    const all = new Float32Array([...verts, ...segs]);
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.geo);
    gl.bufferData(gl.ARRAY_BUFFER, all, gl.DYNAMIC_DRAW);
  }

  setSlice(nu) { this.slice = nu; if (this.scene) this._buildGeo(); this.draw(); }

  _target(W, H) {
    const gl = this.gl;
    if (this.fbW === W && this.fbH === H) return;
    this.fbW = W; this.fbH = H;
    if (this.fbo) { gl.deleteFramebuffer(this.fbo); gl.deleteTexture(this.fboTex); }
    this.fboTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.fboTex);
    if (this.hasFloat) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, W, H, 0, gl.RGBA, gl.HALF_FLOAT, null);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.fboTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  draw() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = null; this._draw(); if (this.autoRotate) { this.yaw += 0.004; this.draw(); } });
  }

  _draw() {
    const gl = this.gl; if (!gl || !this.scene) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = this.host.clientWidth, h = this.host.clientHeight; if (w < 10 || h < 10) return;
    const W = Math.round(w * dpr), H = Math.round(h * dpr);
    if (this.canvas.width !== W || this.canvas.height !== H) { this.canvas.width = W; this.canvas.height = H; this.labels.width = W; this.labels.height = H; }
    this.canvas.style.width = this.labels.style.width = w + "px"; this.canvas.style.height = this.labels.style.height = h + "px";
    const eye = [this.target[0] + this.dist * Math.cos(this.pitch) * Math.sin(this.yaw),
      this.target[1] + this.dist * Math.sin(this.pitch),
      this.target[2] + this.dist * Math.cos(this.pitch) * Math.cos(this.yaw)];
    const view = lookAt(eye, this.target, [0, 1, 0]);
    const proj = perspective(0.75, W / H, 0.02, 50);
    this.view = view; this.proj = proj;

    // 1) splats -> HDR buffer
    this._target(W, H);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, W, H); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    if (this.n) {
      const P = this.progSplat; gl.useProgram(P.p);
      gl.uniformMatrix4fv(P.u.view, false, view); gl.uniformMatrix4fv(P.u.proj, false, proj);
      gl.uniform2f(P.u.viewport, W, H);
      gl.uniform1f(P.u.exposure, this.exposure * (this.baseExposure || 1) * (this.hasFloat ? 1 : 0.4));
      const sl = this.slice != null ? this.toDisp(0, 0, this.slice)[1] : 0;
      gl.uniform1f(P.u.sliceY, sl); gl.uniform1f(P.u.sliceOn, this.slice != null ? 1 : 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0); gl.vertexAttribDivisor(0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.inst);
      const st = 13 * 4;
      [[1, 3, 0], [2, 3, 3], [3, 3, 6], [4, 4, 9]].forEach(([loc, size, off]) => {
        gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, st, off * 4); gl.vertexAttribDivisor(loc, 1);
      });
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.n);
      [1, 2, 3, 4].forEach(l => { gl.vertexAttribDivisor(l, 0); gl.disableVertexAttribArray(l); });
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // 2) floor + box on screen
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.016, 0.035, 0.085, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    const G = this.progGeo; gl.useProgram(G.p);
    gl.uniformMatrix4fv(G.u.view, false, view); gl.uniformMatrix4fv(G.u.proj, false, proj);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.geo);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 20, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 20, 12);
    if (this.floorCount && this.showFloor && this.tex) {
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.tex); gl.uniform1i(G.u.img, 0);
      gl.uniform1f(G.u.useTex, 1); gl.uniform4f(G.u.col, 0.55, 0.6, 0.7, 1);
      gl.drawArrays(gl.TRIANGLES, 0, this.floorCount);
    }
    gl.uniform1f(G.u.useTex, 0);
    gl.uniform4f(G.u.col, 0.45, 0.75, 1.0, 0.35);
    gl.drawArrays(gl.LINES, this.lineStart, this.lineCount);
    if (this.slice != null) {
      gl.uniform4f(G.u.col, 0.25, 0.88, 1.0, 0.09);
      gl.drawArrays(gl.TRIANGLES, this.sliceStart, 6);
    }
    gl.disableVertexAttribArray(1);

    // 3) tone-mapped splats added on top
    gl.blendFunc(gl.ONE, gl.ONE);
    const T = this.progTone; gl.useProgram(T.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.fboTex); gl.uniform1i(T.u.hdr, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.tri);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this._drawLabels(W, H, dpr);
  }

  _project(p) {
    const c = mulv(this.view, [p[0], p[1], p[2], 1]), q = mulv(this.proj, c);
    if (q[3] <= 0) return null;
    return [(q[0] / q[3] * 0.5 + 0.5) * this.canvas.width, (1 - (q[1] / q[3] * 0.5 + 0.5)) * this.canvas.height];
  }

  _drawLabels(W, H, dpr) {
    const g = this.labels.getContext("2d");
    g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, W, H); g.scale(dpr, dpr);
    const S = this.scene, [x0, x1, y0, y1] = S.mapRect;
    g.font = font(12); g.fillStyle = "rgba(170,205,255,.85)"; g.textAlign = "right"; g.textBaseline = "middle";
    for (const nu of this.ticks || []) {
      const p = this._project(this.toDisp(x0 - (x1 - x0) * 0.05, y1, nu)); if (!p) continue;
      g.fillText(String(nu), p[0] / dpr, p[1] / dpr);
    }
    const top = this._project(this.toDisp(x0 - (x1 - x0) * 0.05, y1, S.wnRange[1] + 60));
    if (top) { g.textAlign = "center"; g.fillStyle = "#6f8ab3"; g.fillText("Raman shift (cm⁻¹)", top[0] / dpr, top[1] / dpr); }
    for (const pk of S.peaks || []) {
      if (pk.pos < S.wnRange[0] || pk.pos > S.wnRange[1]) continue;
      const p = this._project(this.toDisp(x1, y0, pk.pos)); if (!p) continue;
      g.textAlign = "left"; g.fillStyle = "rgba(230,240,255,.7)"; g.fillText(`— ${pk.name}`, p[0] / dpr + 4, p[1] / dpr);
    }
    const xl = this._project(this.toDisp((x0 + x1) / 2, y1 + (y1 - y0) * 0.08, S.wnRange[0]));
    if (xl) { g.textAlign = "center"; g.fillStyle = "#6f8ab3"; g.fillText(`X  ${x0.toFixed(1)} … ${x1.toFixed(1)} µm`, xl[0] / dpr, xl[1] / dpr); }
    if (this.slice != null) {
      const p = this._project(this.toDisp(x1, y1, this.slice));
      if (p) { g.textAlign = "left"; g.fillStyle = "#3fe0ff"; g.fillText(`slice ${this.slice.toFixed(0)} cm⁻¹`, p[0] / dpr + 6, p[1] / dpr); }
    }
  }

  snapshot() { this._draw(); return this.canvas.toDataURL("image/png"); }
}

// ---------------------------------------------------------------------------
// The field sliced at one Raman shift, evaluated on a fine (x, y) grid on the
// CPU. Conditioning a 3-D Gaussian on nu gives a 2-D Gaussian whose centre
// shifts with (nu - mu_nu) and whose height falls as exp(-0.5 k dnu^2).
export function sliceField(field, nu, xs, ys) {
  const nx = xs.length, ny = ys.length, out = new Float32Array(nx * ny);
  const dx = Math.abs(xs[1] - xs[0]) || 1, dy = Math.abs(ys[1] - ys[0]) || 1;
  const x0 = Math.min(xs[0], xs[nx - 1]), y0 = Math.min(ys[0], ys[ny - 1]);
  for (let i = 0; i < field.mu.length; i++) {
    const [mx, my, mn] = field.mu[i], c = field.cov[i], a = field.amp[i];
    const inv = inv3(c); if (!inv) continue;
    const Pxx = inv[0], Pxy = inv[1], Pyy = inv[4], Pxn = inv[2], Pyn = inv[5], Pnn = inv[8];
    const det = Pxx * Pyy - Pxy * Pxy; if (det <= 0) continue;
    const dn = nu - mn;
    // shift = -Pxy2^-1 p dn
    const sx = -(Pyy * Pxn - Pxy * Pyn) / det * dn, sy = -(-Pxy * Pxn + Pxx * Pyn) / det * dn;
    const kq = Pnn - (Pxn * (Pyy * Pxn - Pxy * Pyn) + Pyn * (-Pxy * Pxn + Pxx * Pyn)) / det;
    const h = a * Math.exp(-0.5 * kq * dn * dn);
    if (h < 1e-4 * a || !(h > 0)) continue;
    const cx = mx + sx, cy = my + sy;
    const rx = 3.2 * Math.sqrt(Pyy / det), ry = 3.2 * Math.sqrt(Pxx / det);
    const j0 = Math.max(0, Math.floor((cx - rx - x0) / dx)), j1 = Math.min(nx - 1, Math.ceil((cx + rx - x0) / dx));
    const i0 = Math.max(0, Math.floor((cy - ry - y0) / dy)), i1 = Math.min(ny - 1, Math.ceil((cy + ry - y0) / dy));
    for (let r = i0; r <= i1; r++) {
      const yy = y0 + r * dy - cy;
      for (let q = j0; q <= j1; q++) {
        const xx = x0 + q * dx - cx;
        out[r * nx + q] += h * Math.exp(-0.5 * (Pxx * xx * xx + 2 * Pxy * xx * yy + Pyy * yy * yy));
      }
    }
  }
  return out;
}

function inv3(m) {
  const [a, b, c, d, e, f, g, h, k] = m;
  const A = e * k - f * h, B = -(d * k - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!(Math.abs(det) > 1e-30)) return null;
  const D = -(b * k - c * h), E = a * k - c * g, F = -(a * h - b * g);
  const G = b * f - c * e, H = -(a * f - c * d), K = a * e - b * d;
  return [A / det, D / det, G / det, B / det, E / det, H / det, C / det, F / det, K / det];
}
