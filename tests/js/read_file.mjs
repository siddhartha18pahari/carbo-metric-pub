// Read one file with the browser engine and print a summary as JSON.
// Used by tests/test_web_engine.py to hold both readers to the same result.
//   node tests/js/read_file.mjs <file> [<sibling .l6pf>]
import fs from "node:fs";
import path from "node:path";
import { readMap, readL6pf, placeSpectraList } from "../../web/js/engine/readers.js";

const [file, project] = process.argv.slice(2);
const buf = b => { const d = fs.readFileSync(b); return d.buffer.slice(d.byteOffset, d.byteOffset + d.byteLength); };
const name = path.basename(file);
let map, video = null;
if (name.toLowerCase().endsWith(".l6pf")) ({ map, video } = readL6pf(buf(file), name));
else {
  try { map = readMap(buf(file), name); }
  catch (e) {
    if (!e.pointList || !project) throw e;
    const pf = readL6pf(buf(project), path.basename(project));
    map = placeSpectraList(e, pf, name); video = pf.video;
  }
}
const pick = a => (a ? [a[0], a[a.length - 1]] : null);
let sum = 0; for (let i = 0; i < map.cube.length; i += 997) sum += map.cube[i];
let rgb = 0; if (video) for (let i = 0; i < video.rgb.length; i += 997) rgb += video.rgb[i] * (i % 7 + 1);
const out = { nx: map.nx, ny: map.ny, k: map.wn.length, wn: pick(map.wn), x: pick(map.x), y: pick(map.y), kind: map.kind || null,
              cube_probe: sum, video: video ? { w: video.w, h: video.h, x: pick(video.x), y: pick(video.y), rgb_probe: rgb } : null };
if (map.points) out.points = Object.fromEntries(Object.entries(map.points).map(([k, v]) => [k, Array.from(v.slice(0, 5))]));
process.stdout.write(JSON.stringify(out));
