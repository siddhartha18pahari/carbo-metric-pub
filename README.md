# Carbo Metrics

Per-pixel D and G band fitting of Raman maps of carbons, with maps of every band, their ratio and the fit error. It follows the conventions of the Q-Carbon module of HORIBA LabSpec 6 (same panel layout, configuration and export), and adds views Q-Carbon does not have: a registered white-light overlay, linked cursors, particle-level statistics, spectral Gaussian splatting and a printable report.

**Web app:** https://carbo-metrics.vercel.app. It runs entirely in your browser: files are read on your computer and never uploaded.

## What it opens

- LabSpec 6 binary maps (`.l6m`), read directly with no export step, plus the white-light image saved with them (`.l6v`). Maps recorded through the laser line (spectra starting in the anti-Stokes region, e.g. −1130 to 1940 cm⁻¹) are read too.
- LabSpec ParticleFinder projects (`.l6pf`): one spectrum per particle, with each particle's stage position, size and shape and the mosaic camera image. A ParticleFinder spectra list (`.l6m` with no grid) opens with the `.l6pf` saved beside it (in the web app, drop both together).
- Text exports from any instrument (`.txt`, `.csv`, `.tsv`, `.dat`; tab, comma, semicolon or space separated, decimal commas allowed):
  - a LabSpec matrix export (first row = Raman shift, then one row per point). A map whose acquisition stopped early (e.g. 118 of an 11 × 11 grid) opens on its grid, with the points never measured left out of the fit and the statistics;
  - an x / y / shift / intensity table (e.g. Renishaw WiRE);
  - a two-column spectrum (shift, intensity), shown as a single point.
- Fitted-result tables: Q-Carbon's own *Export data* `.txt` (per point: D/G intensity, width, area and ratios) and this app's *Export all (CSV)*. They hold fitted values and no spectra, so they open as maps, histograms, statistics, particles and a report, with nothing to refit. Q-Carbon's export carries only the two bands of its ratio (G and D3, or D1 and G: all 61 exports checked) and no standard error, so the other bands and the Error map need the map itself; the app says so and offers to fit the `.l6m` (one click when it was opened with the export). The spectral Processing settings are greyed out for the same reason; *Smoothing (spatial)* still works, on the fitted maps. The cursor spectrum and the Spectral data panel draw the bands the values describe: each band at its exported height and width, with the line shape read off the exported areas (area / (height × width) is 1.06 for a Gaussian, 1.3 or more for a Lorentzian cut to the fit range) and no baseline. Q-Carbon exports no band positions, so those bands sit at their configured positions and the plot says so. Q-Carbon's export has no coordinates: open it together with its `.l6m` (and `.l6v`) to use the real grid and white-light image. Opened alone, its grid is inferred from the point count, choosing the most isotropic map with a mild preference for square shapes. On 34 exports with a same-size `.l6m` on disk the first choice was right 33 times. The Spectral data panel lets you pick another grid.

Older 32-bit LabSpec files, scattered points that share no grid, and spectra lists without their ParticleFinder project are refused with a message rather than guessed at.

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
- **Smoothing, as options** (all off by default, so the fit sees the raw spectra as in Q-Carbon):
  - *Smoothing (spectral)* in Processing: Savitzky–Golay along the Raman shift (5–21 points, order 2–4) before the fit. Spectra look cleaner; fitted values change little, and a wide window broadens narrow bands such as G.
  - *Smoothing (spatial)* in Processing: each spectrum smoothed with its 3 × 3, 5 × 5 or 7 × 7 neighbours on the map (Gaussian weights over measured neighbours) before the fit, so every map follows, Ratio and Error included. Weak-band maps such as D3 get much cleaner (3 × 3 cuts per-pixel noise about 2×), at the cost of resolution. Neighbouring points then share data, so per-point statistics overstate the number of independent points. On a file of fitted values (no spectra) the fitted maps themselves are smoothed and the ratios recomputed. Off for ParticleFinder data, where neighbouring particle numbers are not neighbouring places.
  - *Filter* in each map's ⚙ menu: Gaussian or median, 3 × 3 or 5 × 5, over valid neighbours. Display and PNG only; statistics, histograms and data exports keep the raw per-pixel values.
  - *Smooth / Raw* chip on each map: D1 to D4 are weak bands, noisy pixel to pixel, so their maps are drawn smoothed by default (Gaussian 3 × 3 filter and interpolated pixels); G, the ratios and the error are drawn as measured. One click switches a map between the two, remembered per map and band. Display only, like the filter.
  - With either pre-fit option on, the cursor spectrum shows the smoothed spectrum the fit used, over the raw one, and Std. error and χ²ᵣ describe the smoothed spectra.
- **Registered optical image.** The map is drawn on the `.l6v` white-light frame at its true stage position, with an opacity slider. The crosshair follows the cursor.
- **Linked cursor.** Hovering any map updates all four maps, the spectrum and the optical image. Click to pin a point, and use the arrow keys to step through pixels. The spectrum shows each fitted band, the baseline and the residual. Drag to zoom.
- **Exclude substrate (Otsu).** One click sets the intensity limit so bare substrate stops counting. On one test map the D3/G intensity ratio over all points is 0.37 ± 2.09; on the 3 325 particle points it is 0.060 ± 0.035. Per-pixel ratios on bare substrate are noise divided by noise.
- **Particles tab.** Finds particles (Otsu on log(1 + I_G)) and splits touching ones at the neck with a marker watershed. It reports pixel-level vs particle-level statistics. Neighbouring pixels are not independent samples, so compare samples on particle means.
- **ParticleFinder particles.** Every particle's spectrum is fitted with the same configuration as a map. The maps draw each particle as a circle at its stage position, sized by its diameter and coloured by the fitted value, over the mosaic image. The Particles tab plots any fitted value against the instrument's morphology (diameter, area, min diameter, convexity, elongation, volume) with binned medians and a rank correlation, and lists the particles; click one to see its spectrum, or its number to open the D, G, Ratio and Error maps zoomed onto it. *Export all (CSV)* adds position, morphology and class per particle. Splat 3D needs a grid and is off for these files.
- **Semi-quantitative particle results** (Particles tab, ParticleFinder data). Every share comes with a 95 % interval, and every figure uses only the particles whose spectra can carry it:
  - *Detection and quality*: the noise is each particle's own fit residual. Carbon is *detected* when G ≥ 3 × noise and *quantified* when G ≥ 10 × noise (both editable), with G inside its fit bounds (a band on a bound is fitting noise) and the Constraints passed. Flat (all-zero) spectra count as not measured. Detected share: Wilson interval.
  - *Populations* on a chosen band ratio over G (D1/G by default), the G width and the G position: a Gaussian mixture (lowest ICL of 1 to 4 components), whose components are merged when their density shows no dip between them (ridge ratio above 0.2, Hennig 2010), so one continuous spread stays one population; or cut-offs on the ratio. Particles whose band is below detection form their own *n.d.* group, since their ratio is noise. Shares by number (Wilson) and by ParticleFinder volume (bootstrap over particles; a mass share only at equal density), and a mean spectrum per population, baseline removed and scaled to G. On a 4 575-particle project D3 was detected in fewer than 1 % of quantified particles: a single particle's D3/G is noise, which is why D1/G is the default.
  - *Crystallite size* L<sub>a</sub> = 2.4 × 10⁻¹⁰ λ⁴ (A<sub>D1</sub>/A<sub>G</sub>)⁻¹ nm (Cançado et al. 2006), with the laser wavelength read from the file (LabSpec stores it in the acquisition record; a project saved with the laser "Off" leaves it to be typed in). Values outside the calibrated range (about 2 to 500 nm) are counted, not hidden.
  - *Size*: D10, D50, D90 by number and by volume, the ratio by diameter quartile, the volume-weighted mean ratio and its rank correlation with diameter. *Export particle analysis (CSV)* writes every particle with its signal-to-noise, detection, population, ratio, G width and position, A<sub>D1</sub>/A<sub>G</sub> and L<sub>a</sub>.
- **Splat 3D tab** (desktop version; a work in progress in the web app). Spectral Gaussian splatting: the baseline-corrected cube on the particles becomes a field of 3-D Gaussians in (x, y, Raman shift), fitted with `ramansp.splatting`. The WebGL viewer projects each Gaussian as in 3D Gaussian splatting. Every 5th channel is held out; on a 9 000-point map the held-out error is 0.97× the map's own noise, with 1 568 Gaussians and 58× fewer numbers than the cube.
- **Zoom.** Scroll on any map to zoom at the pointer, drag to pan, double-click (or ⟲) for the whole map. The four maps zoom together and the optical image follows; on ParticleFinder data each particle shows its number once you are close enough to read it.
- **Small windows.** Below 680 px of height the page scrolls (the top bar stays) rather than squeeze the maps and spectra.
- **Map grid.** The map cards are as tall as the map needs, so a wide map leaves no empty band. The four maps sit in a row as in Q-Carbon, or 2 × 2 when that draws each map at least 1.5 times bigger (wide maps on a laptop screen: about four times). Map tools (display options, PNG, enlarge) appear over the map on hover.
- **Several files.** Drop or pick several maps at once (each `.l6v` goes to the map whose name it shares most of; a ParticleFinder `.l6pf` goes with the spectra list named after it) and they are fitted one after another with the Carbon fit settings. The *Files* tab lists every map opened in the session with its status and a summary of the valid points (ratio median, mean ± SD), exports that summary as one CSV row per file, and shows the same map (ratio, band intensity or error) of up to four ticked files side by side on one colour scale. A file fitted with other settings is marked *settings changed*; *Fit all* refits only those. Click a file, or use the file menu in the top bar, to open it on the Carbon fit tab: its fit is reused when the settings still match. Fitted-result tables still open one at a time. In the web version fits run one at a time (they share the worker pool), and *Remove* frees the file's memory.
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
| ParticleFinder fields and order | a 4 575-particle project vs rows of LabSpec's own Results table | each row's red-cursor height finds one spectrum; that particle's min diameter, convexity, elongation and volume all match (7 of 7 rows) |
| Maps through the laser line | two such maps vs LabSpec's red map of each | r = 0.999 and 0.989; mirrored ≈ 0 |

**D3: shapes from the mean spectrum.**

- **Problem.** Fitted freely per pixel, the broad D3 band wandered to its bounds: on map B, 62 % of pixels had D3's position on a limit. It then soaked up background curvature the quadratic baseline cannot follow, so the D3 map showed the substrate, not the carbon. It correlated with Q-Carbon's own `D3(ar).l6i` at only r = 0.38, and the particle was not visible.
- **What the app does.** It first fits the mean spectrum of the carbon pixels. A band whose position or width a single pixel cannot determine keeps that mean shape, and only its intensity is fitted per pixel. "Cannot determine" means: standard error at the map's own noise level above ¼ of the width (or ½ of the width for the position).
- **Effect.** The particle is visible again, and map B's D3 correlates with Q-Carbon at r = 0.72. G is unchanged: it stays free, and still matches Q-Carbon at r = 0.9999.
- **Which bands are held** is decided per map, and the app says which under the Peaks table. Typically D3 and D4, plus D2 on noisier maps.
- **Limits.** A single pixel carries D3 at only about 1–2 σ at typical count levels, so some speckle is physics. D3 values left on bare substrate are background, not carbon: use *Exclude substrate (Otsu)*. A higher baseline order does not help (order 4 makes D3 worse).
- **Turning it off** gives the plain per-pixel fit. Q-Carbon itself sets D3 to zero in 38 % of map A's pixels. Compare D3 numbers like-for-like only, and not between programs.

**ParticleFinder format.** A `.l6pf` holds the mosaic (uint8 RGB with its X and Y stage axes), the spectra (one per particle, with a 1…P index axis) and a particle table of 24 float32 per particle: X, Y, area, circle-equivalent diameter, …, min diameter (field 11), convexity (15), elongation (16), volume (18), bounding box (19–22), class (23). The table is found by an identity that cannot hold by chance, diameter = √(4 · area / π), and every position must lie inside its own box. Fields not pinned against the instrument are not shown.

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

Copyright (c) 2026 Siddhartha Pahari. All rights reserved. No part of this software may be copied, modified, distributed or used in other work without prior written permission. See `LICENSE`.
