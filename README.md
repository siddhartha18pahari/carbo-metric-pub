# Carbo Metrics

Per-pixel D and G band fitting of Raman maps of carbons, with maps of every band, their ratio and the fit error. It follows the conventions of the Q-Carbon module of HORIBA LabSpec 6 (same panel layout, configuration and export), and adds views Q-Carbon does not have: a registered white-light overlay, linked cursors, particle-level statistics, spectral Gaussian splatting and a printable report.

**Web app:** https://carbo-metrics.vercel.app. It runs entirely in your browser: files are read on your computer and never uploaded.

## What it opens

- LabSpec 6 binary maps (`.l6m`), read directly with no export step, plus the white-light image saved with them (`.l6v`).
- Text exports from any instrument (`.txt`, `.csv`, `.tsv`, `.dat`; tab, comma, semicolon or space separated, decimal commas allowed):
  - a LabSpec matrix export (first row = Raman shift, then one row per point);
  - an x / y / shift / intensity table (e.g. Renishaw WiRE);
  - a two-column spectrum (shift, intensity), shown as a single point.
- Fitted-result tables: Q-Carbon's own *Export data* `.txt` (per point: D/G intensity, width, area and ratios) and this app's *Export all (CSV)*. They hold fitted values and no spectra, so they open as maps, histograms, statistics, particles and a report, with nothing to refit. Q-Carbon's export has no coordinates: open it together with its `.l6m` (and `.l6v`) to use the real grid and white-light image. Opened alone, its grid is inferred from the point count, choosing the most isotropic map with a mild preference for square shapes. On 34 exports with a same-size `.l6m` on disk the first choice was right 33 times. The Spectral data panel lets you pick another grid.

Point lists that do not form a full grid (e.g. ParticleFinder) and older 32-bit LabSpec files are refused with a message rather than guessed at.

## Two ways to run it

**In the browser.** Open the web app and click **Open file…** (or drop a file onto the window). Pick the `.l6m` and its `.l6v` together to get the optical overlay. The fit runs in Web Workers on your machine.

**On your computer** (faster numba engine, a library of your data folders, and Splat 3D):

```
pip install -r requirements.txt
python run.py
```

It opens at http://127.0.0.1:8765. On Windows you can double-click `Carbo Metrics.bat`. To list your own data folders in the library, copy `data_roots.example.txt` to `data_roots.txt` and add one folder per line. Splat 3D also needs `ramansp` (`pip install git+https://github.com/siddhartha18pahari/RAMANSP`).

Both run the same engine: `web/js/engine` is a port of `qcarbon/fitter.py`, and a test holds the two to the same numbers (to 1e-7).

## Same as Q-Carbon

| Q-Carbon | Here |
|---|---|
| Processing: baseline (poly order), Fit separately, spectral range | same. *Fit separately* = modified-polynomial baseline first, then the bands |
| Constraints: Intensity / Ratio(I) / Ratio(Ar) ranges with tick boxes | same; they decide which points are *valid* |
| Peaks table: Pos/Min/Max, Width/Min/Max, Lorentz/Gauss, Fit, peak names, Skip fitting step | same |
| Configuration open/save, Ratio type (D/G or G/D) | same; saved as this app's own `.ini` |
| D, G, Ratio and Error maps, each switchable to Intensity / Width / Area | same, plus Position, red. χ², iterations and mean counts |
| Cursor spectrum, histogram, white-light image, all spectra | same |
| Quality (total/valid points, std. error, red. χ²), Statistics (mean, SD) | same, plus median and IQR |
| Export data (.txt) | same column order as the vendor export; also a full CSV with X, Y, every band and the valid flag |
| Report / Save as PDF | Report tab, then Print, then Save as PDF |

The default configuration is a typical five-band carbon model: poly order 2 baseline fitted with the bands over 1000–1800 cm⁻¹; G 1580 Lorentz (width 10–35), D1 1350 Lorentz, D2 1620 Lorentz, D3 1500 Gauss (40–200), D4 1200 Lorentz; ratio D3/G. Every value is editable and can be saved.

## Added

- **Shapes from the mean spectrum** (Processing, on by default). See *D3* below.
- **Registered optical image.** The map is drawn on the `.l6v` white-light frame at its true stage position, with an opacity slider. The crosshair follows the cursor.
- **Linked cursor.** Hovering any map updates all four maps, the spectrum and the optical image. Click to pin a point, and use the arrow keys to step through pixels. The spectrum shows each fitted band, the baseline and the residual. Drag to zoom.
- **Exclude substrate (Otsu).** One click sets the intensity limit so bare substrate stops counting. On one test map the D3/G intensity ratio over all points is 0.37 ± 2.09; on the 3 325 particle points it is 0.060 ± 0.035. Per-pixel ratios on bare substrate are noise divided by noise.
- **Particles tab.** Finds particles (Otsu on log(1 + I_G)) and splits touching ones at the neck with a marker watershed. It reports pixel-level vs particle-level statistics. Neighbouring pixels are not independent samples, so compare samples on particle means.
- **Splat 3D tab** (desktop version). Spectral Gaussian splatting: the baseline-corrected cube on the particles becomes a field of 3-D Gaussians in (x, y, Raman shift), fitted with `ramansp.splatting`. The WebGL viewer projects each Gaussian as in 3D Gaussian splatting. Every 5th channel is held out; on a 9 000-point map the held-out error is 0.97× the map's own noise, with 1 568 Gaussians and 58× fewer numbers than the cube.
- **Display.** Colour maps (LabSpec rainbow, Viridis, Inferno, Magma, Turbo, Cividis, ...), 1–99 % / full / manual scaling, optional smoothing (display only; statistics always use raw pixels), enlarged map view, PNG export.

## Checked against Q-Carbon

| Check | Data | Result |
|---|---|---|
| G band, per pixel | map A (12 382 points) vs its Q-Carbon export | intensity r = 0.9997, area r = 0.9998 |
| G band area map | map B vs Q-Carbon `G(ar).l6i` | r = 0.9999, medians 1955 vs 1961 |
| Area convention | vendor area / (height × FWHM) | line shape integrated over the fit range: 1.541 for a 19 cm⁻¹ Lorentzian (vendor 1.5409); Gauss 1.0645 |
| Std. error convention | map B vs Q-Carbon `ERROR.l6i` | √(SSR / (N − 3 × bands)): ratio 1.00 on pixels where both fits agree |
| Map orientation | map C vs LabSpec's own screenshot of it | r = 0.949; rotated 0.48, mirrored ≈ 0 |
| Text export | a LabSpec text matrix vs the same map's `.l6m` | identical cube and axes |

**D3: shapes from the mean spectrum.**

- **Problem.** Fitted freely per pixel, the broad D3 band wandered to its bounds: on map B, 62 % of pixels had D3's position on a limit. It then soaked up background curvature the quadratic baseline cannot follow, so the D3 map showed the substrate, not the carbon. It correlated with Q-Carbon's own `D3(ar).l6i` at only r = 0.38, and the particle was not visible.
- **What the app does.** It first fits the mean spectrum of the carbon pixels. A band whose position or width a single pixel cannot determine keeps that mean shape, and only its intensity is fitted per pixel. "Cannot determine" means: standard error at the map's own noise level above ¼ of the width (or ½ of the width for the position).
- **Effect.** The particle is visible again, and map B's D3 correlates with Q-Carbon at r = 0.72. G is unchanged: it stays free, and still matches Q-Carbon at r = 0.9999.
- **Which bands are held** is decided per map, and the app says which under the Peaks table. Typically D3 and D4, plus D2 on noisier maps.
- **Limits.** A single pixel carries D3 at only about 1–2 σ at typical count levels, so some speckle is physics. D3 values left on bare substrate are background, not carbon: use *Exclude substrate (Otsu)*. A higher baseline order does not help (order 4 makes D3 worse).
- **Turning it off** gives the plain per-pixel fit. Q-Carbon itself sets D3 to zero in 38 % of map A's pixels. Compare D3 numbers like-for-like only, and not between programs.

**Map orientation.** LabSpec rasters X fast and draws Y downwards, and its text export writes the Y column first. Reading the slow axis as X transposes the map: statistics are unaffected, but images and overlays are not. Both readers here follow LabSpec's convention.

## Conventions that could not be read off Q-Carbon

- The intensity constraint applies to the larger of the two ratio bands' intensities.
- A ratio with a zero denominator is 0 (the vendor exports do this).
- An unticked Fit box holds that band's position and width; its intensity is still fitted.
- *Skip fitting step* holds every position and width; intensities and baseline are still fitted.
- Red. χ² uses Poisson variance, max(model, 1). Q-Carbon's definition is not documented.
- LabSpec's own `.ini` files are not read. Configurations are saved in this app's format.

## Tests

```
pytest -q tests                           # Python engine, readers, and Python-vs-browser parity
node --test tests/js/engine.test.mjs      # browser engine
```

The vendor-agreement tests compare against Q-Carbon's own outputs. They need your files: copy `tests/vendor_data.example.json` to `tests/vendor_data.json` and fill in the paths; without it they skip. `python tools/serve_web.py` serves the web build the way Vercel does (static files, no Python engine).

## Layout

```
run.py                  start the desktop server and open the browser
data_roots.example.txt  folders the desktop library scans (copy to data_roots.txt)
qcarbon/labspec.py      .l6m / .l6v / .l6i reader; text reader (LabSpec matrix, x/y/shift/intensity, spectrum)
qcarbon/config.py       configuration, INI load/save
qcarbon/fitter.py       per-pixel bounded Levenberg-Marquardt (numba, parallel), shapes from the mean spectrum
qcarbon/results.py      ratios, valid points, quality, statistics, vendor-format export
qcarbon/splat.py        spectral Gaussian splatting (wraps ramansp)
qcarbon/library.py      folder scan with cache (.cache/)
qcarbon/server.py       FastAPI app
web/                    the interface (plain JS modules, canvas + WebGL2, no build step)
web/js/engine/          the same readers and fitter in JavaScript, for the browser-only build;
                        results-file.js opens fitted-result tables (both builds)
tools/serve_web.py      serve web/ as Vercel does, to test the browser build locally
tests/                  pytest suite; tests/js: Node tests of the browser engine
vercel.json             static deployment of web/
```

## License

MIT. See `LICENSE`.
