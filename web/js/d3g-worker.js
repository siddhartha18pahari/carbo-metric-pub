// One worker of a D3/G calibration build: the free S Carbon fits of a chunk of
// spectra, handed back as heights, positions, widths and R².

import { freeFits } from "./d3g-calibration.js";

self.onmessage = e => {
  const { id, nu, N, Y, settings, init, opts } = e.data;
  try {
    const n = Y.length / N, f = freeFits({ nu, N, Y, settings }, Array.from({ length: n }, (_, i) => i), init, opts);
    self.postMessage({ id, h: f.h, c: f.c, w: f.w, r2: f.r2 }, [f.h.buffer, f.c.buffer, f.w.buffer, f.r2.buffer]);
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message || err) });
  }
};
