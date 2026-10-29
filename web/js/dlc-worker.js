// One worker of a DLC fit: the two-band fits of a chunk of spectra, handed back as the
// parameters, their errors, the fitted vectors and a status per spectrum.

import { fitSpectra } from "./dlc.js";

self.onmessage = e => {
  const { id, nu, N, Y, settings, shapes } = e.data;
  try {
    const n = Y.length / N, r = fitSpectra({ nu, N, Y, settings, shapes }, Array.from({ length: n }, (_, i) => i));
    self.postMessage({ id, ...r }, [r.cols.buffer, r.ses.buffer, r.th.buffer, r.status.buffer, r.flags.buffer]);
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message || err) });
  }
};
