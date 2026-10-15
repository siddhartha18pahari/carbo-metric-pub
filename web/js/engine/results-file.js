// Fitted-result tables (Q-Carbon's "Export data", or this app's full CSV).
// They carry fitted values and no spectra, so they open straight into maps,
// histograms and statistics, in either mode, without the server. Q-Carbon's
// export has no coordinates: the grid comes from a .l6m opened with it, or is
// inferred from the point count and offered for correction.

import { isResultsTable, readResultsTable, inferGrid, placeResults, readL6m, readL6v } from "./readers.js";
import { videoUrl } from "./local-api.js";

const TEXT = /\.(txt|csv|tsv|dat)$/i;
let seq = 0;

export async function findResultsFile(files) {
  for (const f of files) if (TEXT.test(f.name) && isResultsTable(await f.slice(0, 65536).text())) return f;
  return null;
}

export async function openResults(file, files) {
  const t = readResultsTable(await file.text(), file.name);
  let grid = null, choices = null, video = null, note = "";
  const l6m = files.find(f => /\.l6m$/i.test(f.name));
  if (t.x) note = "Grid and coordinates from the file's X and Y columns.";
  else if (l6m) {
    const m = readL6m(await l6m.arrayBuffer(), l6m.name);
    if (m.nx * m.ny === t.n) {
      grid = { nx: m.nx, ny: m.ny, x: m.x, y: m.y };
      note = `Grid and stage coordinates from ${l6m.name}.`;
      const vf = files.find(f => /\.l6v$/i.test(f.name));
      if (vf) {
        try {
          const v = readL6v(await vf.arrayBuffer(), vf.name);
          video = { url: await videoUrl(v), meta: { w: v.w, h: v.h, x: [v.x[0], v.x[v.x.length - 1]], y: [v.y[0], v.y[v.y.length - 1]] } };
        } catch { video = null; }
      }
    } else note = `${l6m.name} has ${m.nx * m.ny} points and the table ${t.n}, so the grid was inferred instead. `;
  }
  if (!t.x && !grid) {
    const den = t.bands.includes(t.ratio[1]) ? t.ratio[1] : t.bands[0];
    choices = inferGrid(t.n, t.cols[den].intensity);
    grid = choices[0];
    note += `This file has no coordinates: the grid was inferred from its ${t.n} points (the most isotropic map). ` +
      "Positions are point numbers, not µm. If the maps look sheared, pick another grid, or open the table together with its .l6m.";
  }
  return { id: `results-${++seq}`, table: t, grid, choices, video, note };
}

// the dataset and fit payload the interface expects, for one grid
export function resultsDataset(r, grid, cfg) {
  const t = r.table, { nx, ny, x, y, fit } = placeResults(t, grid), P = nx * ny;
  const nan = () => new Float64Array(P).fill(NaN);
  return {
    info: { id: r.id, name: t.name, path: t.name, nx, ny, k: 0, wn: new Float64Array(0), x, y,
            format: "results", resultsOnly: true, video: r.video ? r.video.meta : undefined },
    fit: { names: t.bands, shapes: t.bands.map(() => "Lorentz"), m: t.bands.length, nb: 0,
           range: [cfg.range_lo, cfg.range_hi], seconds: 0, n_channels: 0, held: [], results_only: true,
           config: { ...structuredClone(cfg), ratio_num: t.ratio[0], ratio_den: t.ratio[1] },
           ...fit, baseline: new Float64Array(0), std_error: nan(), red_chi2: nan(), n_iter: nan() },
  };
}
