// Fit one file with the browser engine and print the result as JSON.
// Used by tests/test_web_engine.py to hold the two engines to the same numbers.
//   node tests/js/fit_file.mjs <file>
import fs from "node:fs";
import path from "node:path";
import { readMap } from "../../web/js/engine/readers.js";
import { fitSpectra } from "../../web/js/engine/core.js";
import { DEFAULT_CONFIG } from "../../web/js/engine/config.js";

const file = process.argv[2];
const b = fs.readFileSync(file);
const m = readMap(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), path.basename(file));
const r = fitSpectra(m.cube, m.nx * m.ny, m.wn, structuredClone(DEFAULT_CONFIG));
const arr = a => Array.from(a);
console.log(JSON.stringify({
  nx: m.nx, ny: m.ny, x: arr(m.x), y: arr(m.y), names: r.names, held: r.held,
  intensity: arr(r.intensity), position: arr(r.position), width: arr(r.width), area: arr(r.area),
  std_error: arr(r.std_error),
}));
