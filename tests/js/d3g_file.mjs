// Run the D3/G calibration on lab files and print the results as JSON.
// Used by tests/test_d3g_package.py to hold it to the calibration scripts' own numbers.
//   node tests/js/d3g_file.mjs <request.json>
// request: { graphite, glassy, files: [{ path, project }], chain: [uncoated index, coated index],
//            compare: [{ name, a: { values, keys }, b: { values, keys } }] }
import fs from "node:fs";
import path from "node:path";
import { readMap, readL6pf, readL6s, placeSpectraList } from "../../web/js/engine/readers.js";
import { BANDS, PRESET_21OCT, prepare, unmix, freeFits, calibrate, standards, blocksOf, compareMedians } from "../../web/js/d3g-calibration.js";

const req = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const buf = f => { const d = fs.readFileSync(f); return d.buffer.slice(d.byteOffset, d.byteOffset + d.byteLength); };
const open = ({ path: p, project }) => {
  try { return readMap(buf(p), path.basename(p)); }
  catch (e) { if (!e.pointList || !project) throw e; return placeSpectraList(e, readL6pf(buf(project), path.basename(project)), path.basename(p)); }
};
const arr = a => Array.from(a, v => (Number.isFinite(v) ? v : null));
const shapes = s => Object.fromEntries(BANDS.map(b => [b, { pos: s[b].pos, fwhm: s[b].fwhm, support: s[b].support ?? null }]));
const B2 = PRESET_21OCT.steps[2].shapes, out = { preset: PRESET_21OCT.steps.map(s => shapes(s.shapes)) };

let B0 = null;
if (req.graphite && req.glassy) {
  B0 = standards(readL6s(buf(req.graphite), path.basename(req.graphite)), readL6s(buf(req.glassy), path.basename(req.glassy))).shapes;
  out.standards = shapes(B0);
}
const preps = (req.files || []).map(f => { const m = open(f); return prepare(m.cube, m.wn.length, m.wn, m.nx * m.ny); });
out.files = preps.map(p => {
  const U = unmix(p, B2), F = freeFits(p, p.carbon, B2), own = calibrate(F, p.sig, B2), U2 = unmix(p, own);
  return { carbon: p.carbon, snr: arr(p.snr), sig: arr(p.sig), h: arr(U.h), se: arr(U.se), rmse: arr(U.rmse), r2: arr(U.r2),
           ratio: Object.fromEntries(Object.entries(U.ratio).map(([k, v]) => [k, arr(v)])), own: shapes(own), own_ratio: arr(U2.ratio["ID3/IG"]),
           own_rn: arr(Array.from(U2.list, (q, i) => U2.rmse[i] / p.sig[q])), rn: arr(Array.from(U.list, (q, i) => U.rmse[i] / p.sig[q])) };
});
if (req.chain && B0) {
  const [a, b] = req.chain.map(i => preps[i]), b1 = calibrate(freeFits(a, a.carbon, B0), a.sig, B0), b2 = calibrate(freeFits(b, b.carbon, b1), b.sig, b1);
  out.chain = { b1: shapes(b1), b2: shapes(b2) };
}
out.compare = (req.compare || []).map(c => ({ name: c.name, ...compareMedians(blocksOf(c.a.values, c.a.keys), blocksOf(c.b.values, c.b.keys)) }));
process.stdout.write(JSON.stringify(out));
