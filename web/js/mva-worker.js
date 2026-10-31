// The MVA tab's spectra prepared (layers.js) and analysed (mva.js) off the page. The prepared
// spectra are kept between runs on the same file, range and background, so trying another method
// is quick; they go back to the page only when new (or when the page asks for them).
import { prepare } from "./layers.js";
import { runMva } from "./mva.js";

let kept = null;
self.onmessage = e => {
  const { key, cube, K, wn, geom, settings, refs, havePrep } = e.data;
  try {
    let fresh = false;
    if (!kept || kept.key !== key) {
      kept = { key, prep: prepare(cube, K, wn, geom, settings, (d, n) => self.postMessage({ stage: `Preparing ${d} of ${n} spectra` })) };
      fresh = true;
    }
    const R = runMva(kept.prep, settings, refs, text => self.postMessage({ stage: text }));
    self.postMessage({ R, prep: fresh || !havePrep ? kept.prep : null });
  } catch (err) { self.postMessage({ error: err.message }); }
};
