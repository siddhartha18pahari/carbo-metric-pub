// The Identify tab's search off the page: the spectra prepared as on the Layers and MVA tabs
// (layers.js), kept between searches on the same file, range and background, then every one
// compared with every reference (identify.js).
import { prepare } from "./layers.js";
import { refVectors, searchAll } from "./identify.js";

let kept = null;
self.onmessage = e => {
  const { key, cube, K, wn, geom, settings, refs, havePrep } = e.data;
  try {
    let fresh = false;
    if (!kept || kept.key !== key) {
      kept = { key, prep: prepare(cube, K, wn, geom, settings, (d, n) => self.postMessage({ stage: `Preparing ${d} of ${n} spectra` })) };
      fresh = true;
    }
    const R = refVectors(refs, kept.prep, settings.deriv);
    const res = searchAll(kept.prep, R, settings.deriv, (d, n) => self.postMessage({ stage: `Searching ${d} of ${n} spectra` }));
    self.postMessage({ res, prep: fresh || !havePrep ? kept.prep : null });
  } catch (err) { self.postMessage({ error: err.message }); }
};
