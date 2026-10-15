// Colour maps as 256-entry RGB lookup tables built from anchor colours.
// "LabSpec" mimics the instrument's rainbow so maps read the same as on the
// Xplora; the others are perceptually ordered and safer for comparisons.

const ANCHORS = {
  LabSpec: ["#00007f", "#0000ff", "#007fff", "#00ffff", "#7fff7f", "#ffff00", "#ff7f00", "#ff0000", "#7f0000"],
  Viridis: ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"],
  Inferno: ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4"],
  Magma: ["#000004", "#180f3d", "#440f76", "#721f81", "#9e2f7f", "#cd4071", "#f1605d", "#fd9668", "#feca8d", "#fcfdbf"],
  Turbo: ["#30123b", "#4662d7", "#36aaf9", "#1ae4b6", "#72fe5e", "#c8ef34", "#faba39", "#f66b19", "#ca2a04", "#7a0403"],
  Cividis: ["#00224e", "#123570", "#3b496c", "#575d6d", "#707173", "#8a8779", "#a69d75", "#c4b56c", "#e4cf5b", "#fee838"],
  Ice: ["#04060f", "#0b1d3f", "#123a72", "#1b5fa6", "#2a8ad0", "#4db6e8", "#8ad9f2", "#cdf1fa", "#ffffff"],
  Red: ["#0a0000", "#3d0000", "#7a0a0a", "#b81414", "#e8331f", "#ff6a4d", "#ffb3a3"],
  Grey: ["#000000", "#ffffff"],
};

function hex(c) { return [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16)); }

const LUTS = {};
for (const [name, cols] of Object.entries(ANCHORS)) {
  const rgb = cols.map(hex), n = rgb.length - 1, lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * n, k = Math.min(n - 1, Math.floor(t)), f = t - k;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = rgb[k][c] * (1 - f) + rgb[k + 1][c] * f;
  }
  LUTS[name] = lut;
}

export const COLORMAPS = Object.keys(ANCHORS);

export function lut(name) { return LUTS[name] || LUTS.LabSpec; }

export function cssGradient(name, dir = "to top") {
  const l = lut(name), stops = [];
  for (let i = 0; i <= 8; i++) {
    const k = Math.round(i / 8 * 255) * 3;
    stops.push(`rgb(${l[k]},${l[k + 1]},${l[k + 2]}) ${i * 12.5}%`);
  }
  return `linear-gradient(${dir}, ${stops.join(",")})`;
}

export function rgbAt(name, t) {
  const l = lut(name), k = Math.max(0, Math.min(255, Math.round(t * 255))) * 3;
  return [l[k], l[k + 1], l[k + 2]];
}
