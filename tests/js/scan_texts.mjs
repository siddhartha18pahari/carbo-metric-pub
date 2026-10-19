// Try every text file under the given folders with the browser readers and
// report which open (as a map or as a fitted-results table) and which fail.
//   node tests/js/scan_texts.mjs <folder> [<folder> ...]
import fs from "node:fs";
import path from "node:path";
import { readText, isResultsTable, readResultsTable, TEXT_EXT } from "../../web/js/engine/readers.js";

const files = [];
const walk = d => {
  for (const e of fs.readdirSync(d, { withFileTypes: true })) {
    const p = path.join(d, e.name);
    if (e.isDirectory()) walk(p);
    else if (TEXT_EXT.includes(path.extname(e.name).toLowerCase()) && fs.statSync(p).size > 2000) files.push(p);
  }
};
for (const d of process.argv.slice(2)) walk(d);
const out = [];
for (const p of files) {
  const name = path.basename(p);
  const buf = fs.readFileSync(p);
  const text = new TextDecoder("utf-8").decode(buf);
  let res;
  try {
    if (isResultsTable(text.slice(0, 65536))) { const t = readResultsTable(text, name); res = `results ${t.n ?? ""}`.trim(); }
    else { const m = readText(text, name); res = `map ${m.nx}x${m.ny} k=${m.wn.length}`; }
  } catch (e) { res = `FAIL ${String(e.message).replace(/\s+/g, " ").slice(0, 150)}`; }
  const head = buf.subarray(0, 4).toString("hex");
  out.push({ p: path.relative(process.argv[2], p), size: buf.length, head, res });
}
process.stdout.write(JSON.stringify(out));
