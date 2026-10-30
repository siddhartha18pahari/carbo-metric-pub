// The Layers tab's spectra prepared and decomposed off the page (layers.js): the slow part, so
// the page stays responsive. The layers themselves are found on the page, from what this returns.
import { prepare, decompose } from "./layers.js";

self.onmessage = e => {
  const { cube, K, wn, geom, settings, refs } = e.data;
  try {
    const prep = prepare(cube, K, wn, geom, settings, (d, n) => self.postMessage({ stage: `Preparing ${d} of ${n} spectra` }));
    const comps = decompose(prep, settings, refs, text => self.postMessage({ stage: text }));
    self.postMessage({ prep, comps }, [prep.V.buffer, prep.base.buffer, prep.noise.buffer, prep.rowOf.buffer, prep.list.buffer, prep.measured.buffer,
      comps.H.buffer, comps.W.buffer, comps.resid.buffer]);
  } catch (err) { self.postMessage({ error: err.message }); }
};
