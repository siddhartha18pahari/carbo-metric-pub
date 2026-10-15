// One fitting worker: runs the per-pixel Levenberg-Marquardt on a chunk of
// spectra and hands the parameters back.

import { lmBatch } from "./core.js";

self.onmessage = e => {
  const { id, job, Y, TH } = e.data;
  const iters = new Int32Array(TH.length / job.lo.length);
  try {
    lmBatch(job, Y, TH, iters);
    self.postMessage({ id, TH, iters }, [TH.buffer, iters.buffer]);
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message || err) });
  }
};
