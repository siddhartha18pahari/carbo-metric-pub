// Light and dark themes. Colours live as CSS custom properties (app.css); the
// canvases read the --cv-* ones here, so plots and maps follow the theme too.
// The choice is kept per browser; the first visit follows the system setting.

export const FONT = '"Inter", "Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif';
export const font = (px, weight = "") => `${weight ? weight + " " : ""}${px}px ${FONT}`;

// canvas palette, refreshed on every theme change
export const T = {};
const VARS = {
  bg: "--cv-bg", ink: "--cv-ink", label: "--cv-label", grid: "--cv-grid", frame: "--cv-frame",
  hatchBg: "--cv-hatch-bg", hatchLine: "--cv-hatch-line", shade: "--cv-shade", range: "--cv-range",
  raw: "--cv-raw", raw0: "--cv-raw0", baseline: "--cv-baseline", fit: "--cv-fit", hair: "--cv-hair",
  tipBg: "--cv-tip-bg", tipInk: "--cv-tip-ink", pin: "--cv-pin", hover: "--cv-hover", strong: "--cv-strong",
  mean: "--cv-mean", residBg: "--cv-resid-bg", resid: "--cv-resid", tagBg: "--cv-tag-bg", tagInk: "--cv-tag-ink",
  accent: "--accent", accent2: "--accent-2", paper: "--cv-paper",
};

// band and particle colours, readable on either background (arrays mutated
// in place so every importer sees the current set)
export const PEAK_COLOURS = [];
export const PARTICLE_COLOURS = [];
const PEAKS = {
  dark: ["#38d6ff", "#ff9f40", "#b18cff", "#ffd84d", "#ff6b9a", "#6be3a4", "#f4f4f4", "#8fb4ff"],
  light: ["#0a86b8", "#d9700b", "#7346d1", "#a8820a", "#d02f68", "#16935a", "#3a465c", "#2f62d0"],
};
const PARTICLES = {
  dark: ["#3fe0ff", "#ff9f40", "#b18cff", "#7dff6a", "#ff6b9a", "#ffd84d", "#6bb8ff", "#ff7b54"],
  light: ["#0a86b8", "#d9700b", "#7346d1", "#169e3a", "#d02f68", "#a8820a", "#2f62d0", "#cc4a26"],
};

export const DARK_PEAK_COLOURS = PEAKS.dark;

export const currentTheme = () => (document.documentElement.dataset.theme === "light" ? "light" : "dark");

export function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  for (const [k, v] of Object.entries(VARS)) T[k] = cs.getPropertyValue(v).trim();
  const t = currentTheme();
  PEAK_COLOURS.splice(0, PEAK_COLOURS.length, ...PEAKS[t]);
  PARTICLE_COLOURS.splice(0, PARTICLE_COLOURS.length, ...PARTICLES[t]);
}

export function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("carbo-theme", t); } catch { /* private window: not remembered */ }
  readTheme();
  window.dispatchEvent(new CustomEvent("themechange", { detail: t }));
}

readTheme();
