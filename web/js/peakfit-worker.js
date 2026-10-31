// Fits a chunk of spectra with one model (peakfit.js fitMany), for the Peak fit, Graphene and
// Silicon tabs: { id, model, wn, x0, Y } -> { id, vals, ses, xs, stats, status }.
import { fitMany } from "./peakfit.js";

self.onmessage = e => {
  const { id, model, wn, x0, Y } = e.data;
  try {
    const r = fitMany({ model, wn, x0 }, Y);
    self.postMessage({ id, ...r }, [r.vals.buffer, r.ses.buffer, r.xs.buffer, r.stats.buffer, r.status.buffer]);
  } catch (err) { self.postMessage({ id, error: String(err.message || err) }); }
};
