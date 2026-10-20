"""Engine tests. The vendor-agreement tests compare against LabSpec Q-Carbon's
own outputs; they need ``tests/vendor_data.json`` (paths to those files, kept
out of the repository: see ``vendor_data.example.json``) and skip without it."""

import json
import os

import numpy as np
import pytest
from scipy.integrate import quad

from qcarbon.config import Config, Peak
from qcarbon.fitter import fit_spectra, shape_eval, windowed_area, baseline_basis
from qcarbon.labspec import read_l6_image, read_l6m, read_l6v, read_map, read_text, find_video
from qcarbon import labspec
from qcarbon.results import export_table, safe_ratio, statistics, valid_mask

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR_FILE = os.path.join(HERE, "vendor_data.json")


def need(*paths):
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f"data not on this machine: {p}")


def vendor(*keys):
    """Paths of vendor-comparison files by role, from tests/vendor_data.json."""
    try:
        with open(VENDOR_FILE, encoding="utf-8") as fh:
            paths = json.load(fh)
    except FileNotFoundError:
        pytest.skip("no tests/vendor_data.json (vendor comparison data)")
    out = [paths.get(k, "") for k in keys]
    need(*out)
    return out[0] if len(out) == 1 else out


# ---------------------------------------------------------------- line shapes
@pytest.mark.parametrize("kind", ["Lorentz", "Gauss"])
@pytest.mark.parametrize("c,w", [(1580, 20), (1350, 90), (1500, 200), (1010, 40)])
def test_windowed_area_matches_numerical_integral(kind, c, w):
    num, _ = quad(lambda x: float(shape_eval(x, c, w, kind)), 1000, 1800, points=[c], limit=400)
    assert windowed_area(1.0, c, w, kind, 1000, 1800) == pytest.approx(num, rel=1e-7)


def test_area_factors_match_vendor_exports():
    # vendor area / (height * FWHM): Gaussian 1.0645 (any window), and a 19 cm-1
    # Lorentzian at 1580 windowed to 1000-1800 gives 1.541 (a vendor export: 1.5409)
    assert windowed_area(1, 1500, 100, "Gauss", 1000, 1800) / 100 == pytest.approx(1.0645, abs=1e-4)
    assert windowed_area(1, 1580, 19, "Lorentz", 1000, 1800) / 19 == pytest.approx(1.541, abs=2e-3)


# ---------------------------------------------------------------- the fit
def synthetic(n=300, seed=1):
    rng = np.random.default_rng(seed)
    wn = np.linspace(1000, 1800, 294)
    cfg = Config()
    truth = []
    Y = np.empty((n, wn.size))
    for i in range(n):
        A = {"G": rng.uniform(30, 120), "D1": rng.uniform(10, 60), "D2": rng.uniform(0, 15),
             "D3": rng.uniform(0, 10), "D4": rng.uniform(0, 8)}
        C = {p.name: rng.uniform(p.pos_min + 2, p.pos_max - 2) for p in cfg.peaks}
        W = {p.name: rng.uniform(p.width_min * 1.3, p.width_max * 0.8) for p in cfg.peaks}
        y = 25 + 4 * baseline_basis(wn, 1000, 1800, 2) @ np.array([1.0, 0.6, -0.4])
        for p in cfg.peaks:
            y = y + A[p.name] * shape_eval(wn, C[p.name], W[p.name], p.shape)
        Y[i] = rng.poisson(y)
        truth.append((A, C, W))
    return wn, Y, truth, cfg


def test_recovers_the_well_determined_bands():
    wn, Y, truth, cfg = synthetic()
    r = fit_spectra(Y, wn, cfg)
    g = r.peak("G"); d1 = r.peak("D1")
    gA = np.array([t[0]["G"] for t in truth]); gC = np.array([t[1]["G"] for t in truth])
    d1A = np.array([t[0]["D1"] for t in truth])
    assert np.median(np.abs(g["intensity"] - gA) / gA) < 0.06
    assert np.median(np.abs(g["position"] - gC)) < 1.0
    assert np.median(np.abs(d1["intensity"] - d1A) / d1A) < 0.12
    # residual at the Poisson noise level
    assert 0.8 < np.median(r.red_chi2) < 1.3


def test_bounds_are_respected():
    wn, Y, _, cfg = synthetic(n=120, seed=2)
    r = fit_spectra(Y, wn, cfg)
    assert (r.intensity >= 0).all()
    for j, p in enumerate(cfg.peaks):
        assert (r.position[:, j] >= p.pos_min - 1e-9).all() and (r.position[:, j] <= p.pos_max + 1e-9).all()
        assert (r.width[:, j] >= p.width_min - 1e-9).all() and (r.width[:, j] <= p.width_max + 1e-9).all()


def test_flat_spectrum_gives_zero_bands_not_negative_ones():
    wn = np.linspace(1000, 1800, 294)
    Y = np.random.default_rng(3).poisson(np.full((50, wn.size), 30.0)).astype(float)
    r = fit_spectra(Y, wn, Config())
    assert (r.intensity >= 0).all()
    assert np.median(r.intensity[:, 0]) < 1.5


def test_fixed_shapes_are_held():
    wn, Y, _, cfg = synthetic(n=40, seed=4)
    cfg.peaks[0].fit = False
    r = fit_spectra(Y, wn, cfg)
    assert np.all(r.position[:, 0] == cfg.peaks[0].pos) and np.all(r.width[:, 0] == cfg.peaks[0].width)


def test_mean_shapes_make_the_weak_broad_band_map_follow_the_truth():
    """A particle whose D3 rises left to right, on a substrate whose
    fluorescence a quadratic cannot follow exactly. Fitted freely per pixel,
    D3 wanders and soaks up background; with shapes from the mean spectrum the
    undetermined bands are held and D3 tracks the truth better, while the
    well-determined G and D1 stay free."""
    rng = np.random.default_rng(4)
    wn = np.linspace(1000, 1800, 294)
    yy, xx = np.mgrid[0:24, 0:24]
    part = (((yy - 12) ** 2 + (xx - 12) ** 2) < 81).ravel()
    d3 = np.where(part, 10.0 * xx.ravel() / 23, 0.0)
    Y = np.empty((part.size, wn.size))
    for p in range(part.size):
        y = 30 + 25 * np.exp(-(wn - 1000) / 500.0)
        if part[p]:
            y = y + 60 * shape_eval(wn, 1580, 20, "Lorentz") + 25 * shape_eval(wn, 1350, 50, "Lorentz") \
                  + 8 * shape_eval(wn, 1615, 25, "Lorentz") + d3[p] * shape_eval(wn, 1500, 120, "Gauss")
        Y[p] = rng.poisson(y)
    off = Config(); off.auto_shapes = False
    r0, r1 = fit_spectra(Y, wn, off), fit_spectra(Y, wn, Config())
    assert r0.held == [] and "D3" in r1.held and not {"G", "D1"} & set(r1.held)
    j = r1.names.index("D3")
    assert np.all(r1.width[:, j] == r1.width[0, j])               # one shape for the map
    r_off = np.corrcoef(r0.intensity[:, j], d3)[0, 1]
    r_on = np.corrcoef(r1.intensity[:, j], d3)[0, 1]
    assert r_on > r_off + 0.03                                   # measured 0.856 vs 0.782
    assert np.std(r1.width[part, 0]) > 0.5                       # G width still per pixel


def test_fit_separately_runs_and_reports_the_baseline():
    wn, Y, _, cfg = synthetic(n=40, seed=5)
    cfg.fit_separately = True
    r = fit_spectra(Y, wn, cfg)
    assert r.baseline.shape == (40, 3)
    assert np.median(r.std_error) < 2 * np.sqrt(30)


# ---------------------------------------------------------------- results
def test_zero_denominator_ratio_is_zero_like_the_vendor():
    assert safe_ratio([1.0, 2.0, 0.0], [0.0, 4.0, 0.0]).tolist() == [0.0, 0.5, 0.0]


def test_export_column_order_follows_the_ratio_type():
    wn, Y, _, cfg = synthetic(n=5)
    r = fit_spectra(Y, wn, cfg)
    head = export_table(r, cfg).splitlines()[0].split("\t")
    assert head[1:10] == ["D3 Intensity", "D3 Width", "D3 Area", "G Intensity", "G Width", "G Area",
                          "Ratio Intensity", "Ratio Width", "Ratio Area"]
    cfg.ratio_num, cfg.ratio_den = "G", "D3"
    assert export_table(r, cfg).splitlines()[0].split("\t")[1] == "G Intensity"


def test_constraints_filter_valid_points():
    wn, Y, _, cfg = synthetic(n=60)
    r = fit_spectra(Y, wn, cfg)
    assert valid_mask(r, cfg).all()
    cfg.intensity.lo, cfg.intensity.use_lo = 1e9, True
    assert not valid_mask(r, cfg).any()
    rows = statistics(r, cfg)
    assert rows[0]["n"] == 0


def test_ini_round_trip():
    cfg = Config()
    cfg.peaks[3] = Peak("D3", 1505, 1490, 1520, 110, 40, 220, "Gauss", False)
    cfg.ratio_num, cfg.intensity.use_lo, cfg.intensity.lo = "D1", True, 12.5
    back = Config.from_ini(cfg.to_ini())
    assert back.to_dict() == cfg.to_dict()


# ---------------------------------------------------------------- vendor data
def test_matches_vendor_g_band_export():
    m_path, t_path = vendor("g_band_map", "g_band_export")
    m = read_l6m(m_path)
    vt = np.loadtxt(t_path, delimiter="\t", skiprows=1)[:, 1:]
    assert vt.shape[0] == m.n_points == 12382
    r = fit_spectra(m.cube.reshape(-1, m.wn.size), m.wn, Config())
    g = r.peak("G")
    assert np.corrcoef(g["intensity"], vt[:, 0])[0, 1] > 0.999
    assert np.corrcoef(g["area"], vt[:, 2])[0, 1] > 0.999
    strong = vt[:, 0] > 20
    assert np.median(np.abs(g["area"][strong] - vt[strong, 2]) / vt[strong, 2]) < 0.03


def test_std_error_convention_matches_vendor_error_map():
    mp, err, g_ar, d3_ar = vendor("error_pair_map", "error_pair_error", "error_pair_g_area", "error_pair_d3_area")
    m = read_l6m(mp)
    r = fit_spectra(m.cube.reshape(-1, m.wn.size), m.wn, Config())
    vendor_err = read_l6_image(err, m.cube.shape[:2]).ravel()
    gv = read_l6_image(g_ar, m.cube.shape[:2]).ravel()
    assert np.corrcoef(r.peak("G")["area"], gv)[0, 1] > 0.999      # also pins the X-fast layout
    same = np.abs(r.peak("D3")["area"] - read_l6_image(d3_ar, m.cube.shape[:2]).ravel()) < 20
    assert np.median(vendor_err[same] / r.std_error[same]) == pytest.approx(1.0, abs=0.01)


def test_map_orientation_matches_labspec_display():
    """LabSpec's own screenshot of this map ("Cursor image: Red map", the mean
    counts over 1545-1625 cm-1 in a black-to-red scale) is the ground truth.
    Axis calibration read off its tick labels: X = 0 at 361 px, 11.7 px/um,
    map top edge at 469 px, Y drawn downwards."""
    mp, png = vendor("screenshot_map", "screenshot_png")
    from PIL import Image
    m = read_l6m(mp)
    assert m.cube.shape[:2] == (97, 92)
    assert m.x[0] == pytest.approx(-17.59, abs=0.02) and m.x[-1] == pytest.approx(9.57, abs=0.02)
    shot = np.asarray(Image.open(png).convert("RGB")).astype(float)
    k = (m.wn > 1545) & (m.wn < 1625)
    g = m.cube[:, :, k].mean(axis=2)
    X, Y = np.meshgrid(m.x, m.y)
    red = shot[np.round(469 + (Y - (m.y[0] - 0.15)) * 11.7).astype(int), np.round(361 + X * 11.7).astype(int), 0]
    ok = red > 3   # skip the crosshair and scale bar
    r = {name: np.corrcoef(gg[ok], red[ok])[0, 1] for name, gg in
         [("right", g), ("rot180", g[::-1, ::-1]), ("flip_x", g[:, ::-1]), ("flip_y", g[::-1])]}
    # measured 2026-10-15: right 0.949, rot180 0.479, flips ~0
    assert r["right"] > 0.9
    assert max(v for k2, v in r.items() if k2 != "right") < 0.6

    # and the white-light image decodes with its axes (camera frame ~108 x 71 um)
    v = read_l6v(find_video(mp))
    assert v.rgb.shape == (664, 1056, 3)
    assert v.x[0] == pytest.approx(-54.26, abs=0.01) and v.y[0] == pytest.approx(35.63, abs=0.01)


def test_d3_map_follows_vendor_d3_map():
    """Fitted freely per pixel, D3 correlated with Q-Carbon's own D3 area map
    at only r = 0.38 and did not show the particle; with shapes from the mean
    spectrum it is 0.72."""
    mp, d3_ar = vendor("error_pair_map", "error_pair_d3_area")
    m = read_l6m(mp)
    r = fit_spectra(m.cube.reshape(-1, m.wn.size), m.wn, Config())
    vendor_d3 = read_l6_image(d3_ar, m.cube.shape[:2]).ravel()
    assert "D3" in r.held and "G" not in r.held
    assert np.corrcoef(r.peak("D3")["area"], vendor_d3)[0, 1] > 0.65


# ---------------------------------------------------------------- text files
def _labspec_matrix(path, ny=3, nx=4, K=40):
    """LabSpec's matrix export: Raman-shift row, then Y, X, intensities with X
    changing fastest. Each spectrum encodes its own (row, column)."""
    wn = np.linspace(1000, 1800, K)
    lines = ["\t\t" + "\t".join(f"{v:.3f}" for v in wn)]
    for iy in range(ny):
        for ix in range(nx):
            lines.append("\t".join([f"{-5 + 2 * iy:.4f}", f"{10 + 1.5 * ix:.4f}",
                                    *[f"{100 * iy + ix + 0.001 * k:.4f}" for k in range(K)]]))
    path.write_text("\n".join(lines) + "\n")
    return wn


def test_labspec_matrix_text_puts_x_on_columns(tmp_path):
    p = tmp_path / "map.txt"
    _labspec_matrix(p)
    m = read_map(str(p))
    assert m.cube.shape == (3, 4, 40)
    assert np.allclose(m.x, 10 + 1.5 * np.arange(4)) and np.allclose(m.y, -5 + 2 * np.arange(3))
    assert m.cube[2, 1, 0] == pytest.approx(201)          # row = Y index, column = X index


def test_labspec_text_export_matches_the_binary():
    """A map exported to text by LabSpec: the text and the binary must give
    the same map, the same way round (the export lists Y first)."""
    txt, binary = vendor("text_export", "text_export_binary")
    t = read_text(txt)
    b = read_l6m(binary)
    assert t.cube.shape == b.cube.shape == (30, 32, 798)
    assert np.array_equal(t.cube, b.cube)
    assert np.allclose(t.x, b.x, atol=1e-3) and np.allclose(t.y, b.y, atol=1e-3)


@pytest.mark.parametrize("sep,header", [("\t", "#X\t#Y\t#Wave\t#Intensity"), (",", "")])
def test_long_xy_table(tmp_path, sep, header):
    """One row per (x, y, shift, intensity), shift descending as WiRE writes it."""
    wn = np.linspace(1800, 1000, 30)
    rows = [header] if header else []
    for iy in range(2):
        for ix in range(3):
            rows += [sep.join(f"{v:g}" for v in (ix * 0.5, iy * 0.5, w, 10 * iy + ix)) for w in wn]
    p = tmp_path / ("map.txt" if sep == "\t" else "map.csv")
    p.write_text("\n".join(rows) + "\n")
    m = read_map(str(p))
    assert m.cube.shape == (2, 3, 30) and m.wn[0] < m.wn[-1]
    assert m.cube[1, 2, 0] == 12 and np.allclose(m.x, [0, 0.5, 1.0])


def test_two_column_spectrum_with_labels(tmp_path):
    wn = np.linspace(1000, 1800, 200)
    y = 20 + 80 * shape_eval(wn, 1580, 20, "Lorentz") + 30 * shape_eval(wn, 1350, 60, "Lorentz")
    p = tmp_path / "spectrum.csv"
    p.write_text("Raman shift (cm-1),Intensity\n" + "\n".join(f"{a},{b}" for a, b in zip(wn, y)))
    m = read_map(str(p))
    assert m.cube.shape == (1, 1, 200)
    r = fit_spectra(m.cube.reshape(1, -1), m.wn, Config())
    assert r.peak("G")["position"][0] == pytest.approx(1580, abs=1)


def test_decimal_commas_with_semicolons(tmp_path):
    wn = np.linspace(1000, 1800, 50)
    p = tmp_path / "spectrum.csv"
    p.write_text("\n".join(f"{a:.2f};{b:.3f}".replace(".", ",") for a, b in zip(wn, 1 + wn / 1e3)))
    m = read_map(str(p))
    assert m.wn[0] == pytest.approx(1000) and m.cube[0, 0, -1] == pytest.approx(2.8)


def test_three_points_of_a_2x2_grid_are_a_map_stopped_early(tmp_path):
    wn = np.linspace(1000, 1800, 20)
    rows = ["\t\t" + "\t".join(f"{v:g}" for v in wn)]
    for x, y in [(0, 0), (1, 0), (0, 1)]:                  # the last point was never measured
        rows.append("\t".join([str(y), str(x), *["1"] * wn.size]))
    p = tmp_path / "points.txt"
    p.write_text("\n".join(rows))
    m = read_map(str(p))
    assert m.cube.shape[:2] == (2, 2) and m.meta["missing"] == 1 and np.isnan(m.cube[1, 1]).all()


# ---------------------------------------------------------------- smoothing
def test_spectral_smoothing_is_savitzky_golay():
    from scipy.signal import savgol_filter
    from qcarbon.fitter import smooth_spectra
    cube = np.random.default_rng(3).normal(100, 5, (4, 5, 150))
    for w, o in [(5, 2), (11, 3), (21, 4)]:
        cfg = Config(smooth_window=w, smooth_order=o)
        np.testing.assert_allclose(smooth_spectra(cube, cfg), savgol_filter(cube, w, o, axis=-1, mode="interp"), atol=1e-8)


def test_spatial_averaging_keeps_a_flat_map_flat_and_cuts_noise():
    from qcarbon.fitter import smooth_spectra
    flat = np.full((6, 9, 4), 7.0)
    assert np.allclose(smooth_spectra(flat, Config(spatial_kernel=5)), 7.0)       # edges renormalised
    noise = np.random.default_rng(4).normal(0, 1, (40, 40, 50))
    assert smooth_spectra(noise, Config(spatial_kernel=3)).std() < 0.5 * noise.std()


def test_smoothing_settings_are_checked_and_saved():
    with pytest.raises(ValueError):
        Config(smooth_window=8).check()                  # even window
    with pytest.raises(ValueError):
        Config(spatial_kernel=4).check()
    cfg = Config(smooth_window=9, smooth_order=3, spatial_kernel=5)
    assert Config.from_ini(cfg.to_ini()).to_dict() == cfg.to_dict()


# ---------------------------------------------------------------- ParticleFinder
# Rows of LabSpec's own ParticleFinder Results table for the 4575-particle
# project (min diam., convexity, elongation, volume, "Red cursor" height),
# each matched to its spectrum through the red-cursor value.
PF_SCREENSHOT_ROWS = [(2.64, 0.95, 100.00, 6.76, 8078.00), (2.29, 0.95, 100.00, 5.68, 9365.00),
                      (2.64, 0.97, 100.00, 6.31, 7284.00), (3.03, 0.90, 51.98, 8.10, 7432.00),
                      (2.71, 0.92, 56.18, 6.86, 7866.00), (2.45, 0.95, 100.00, 6.81, 7310.00),
                      (2.64, 0.96, 100.00, 6.64, 6529.00)]


def _max_height(path, P):
    from qcarbon.labspec import _mat_blocks, _read
    raw = _read(path)
    for o, n in _mat_blocks(raw):
        if raw[o:o + 11] == b"Max height\x00":
            return np.frombuffer(raw, "<f4", count=P, offset=o + 11).astype(float)
    pytest.skip("no Max height block")


def test_particlefinder_table_rows_match_labspec_results_table():
    from qcarbon.labspec import read_l6pf
    path = vendor("pf_project")
    m, video = read_l6pf(path)
    P = m.cube.shape[1]
    pts = m.meta["points"]
    assert P == 4575 and video is not None
    np.testing.assert_allclose(pts["ecd"], np.sqrt(4 * pts["area"] / np.pi), rtol=1e-3)
    inside = (pts["x"] >= pts["xmin"]) & (pts["x"] <= pts["xmax"]) & (pts["y"] >= pts["ymin"]) & (pts["y"] <= pts["ymax"])
    assert inside.all()
    # the spectral "Max height" is in spectrum order...
    mh = _max_height(path, P)
    assert np.corrcoef(mh, m.cube[0].max(axis=1))[0, 1] > 0.95
    # ...and the row that carries a screenshot particle's height carries its morphology too
    for dmin, conv, elong, vol, red in PF_SCREENSHOT_ROWS:
        k = int(np.flatnonzero(np.abs(mh - red) < 0.6)[-1])
        assert pts["min_diam"][k] == pytest.approx(dmin, abs=0.006)
        assert pts["convexity"][k] == pytest.approx(conv, abs=0.006)
        assert pts["elongation"][k] == pytest.approx(elong, abs=0.006)
        assert pts["volume"][k] == pytest.approx(vol, abs=0.006)


def test_spectra_list_takes_positions_from_its_project():
    from qcarbon.labspec import PointListError, read_any, read_l6pf
    lst, prj = vendor("pf_spectra_list", "pf_project")
    with pytest.raises(PointListError):
        read_l6m(lst)
    m, video = read_any(lst)
    pm, _ = read_l6pf(prj)
    assert m.meta["kind"] == "particles" and video is not None
    np.testing.assert_array_equal(m.cube, pm.cube)
    np.testing.assert_array_equal(m.meta["points"]["x"], pm.meta["points"]["x"])


def test_small_project_mosaic_axes():
    from qcarbon.labspec import read_l6pf
    m, video = read_l6pf(vendor("pf_small_project"))
    assert m.cube.shape[:2] == (1, 364)
    assert video.rgb.shape == (video.y.size, video.x.size, 3)
    pts = m.meta["points"]
    assert (pts["x"] >= video.x.min()).all() and (pts["x"] <= video.x.max()).all()


@pytest.mark.parametrize("map_key,red_key,shape", [("antistokes_map", "antistokes_red_map", (57, 91)),
                                                   ("twin_axis_map", "twin_axis_red_map", (51, 51))])
def test_maps_recorded_through_the_laser_line_read_as_grids(map_key, red_key, shape):
    """Spectra from -1130 cm-1 used to be rejected (the 91-point X axis was taken
    for the spectrum); a file storing X twice used to pair X with itself."""
    mp, rp = vendor(map_key, red_key)
    m = read_l6m(mp)
    assert m.cube.shape[:2] == shape and m.wn[0] < -1000
    assert not np.array_equal(m.x, m.y)
    red = read_l6_image(rp, shape)       # values start after the "Max height" label, off the 4-byte grid
    best = max(np.corrcoef(m.cube[:, :, max(0, c - 9):c + 10].max(axis=2).ravel(), red.ravel())[0, 1]
               for c in range(0, m.wn.size, 4))
    assert best > 0.98



# ---------------------------------------------------------------- maps stopped early
def _map_text(tmp_path, drop=0, seed=11):
    rng = np.random.default_rng(seed)
    wn = np.linspace(1000, 1800, 200)
    lines = ["\t\t" + "\t".join(f"{v:.3f}" for v in wn)]
    for i in range(9):
        for j in range(10):
            y = 30 + 40 * shape_eval(wn, 1580, 20, "Lorentz") * (1 + 0.1 * j) + 15 * shape_eval(wn, 1350, 50, "Lorentz")
            lines.append("\t".join([f"{i * 0.5:.3f}", f"{j * 0.5:.3f}", *map(str, rng.poisson(y))]))
    path = tmp_path / f"map{drop}.txt"
    path.write_text("\n".join(lines[:len(lines) - drop]) + "\n")
    return path


def test_map_stopped_early_keeps_its_grid(tmp_path):
    full, short = read_text(str(_map_text(tmp_path))), read_text(str(_map_text(tmp_path, drop=3)))
    assert short.cube.shape == full.cube.shape == (9, 10, 200)
    assert short.meta["missing"] == 3
    gone = ~np.isfinite(short.cube).all(axis=-1)
    assert gone.sum() == 3 and gone[-1, -3:].all()            # the last three of the last row
    np.testing.assert_array_equal(short.cube[~gone], full.cube[~gone])
    from qcarbon.fitter import smooth_spectra
    from qcarbon.results import quality
    cfg = Config()
    r = fit_spectra(short.cube.reshape(-1, 200), short.wn, cfg)
    assert np.isnan(r.std_error[gone.ravel()]).all() and np.isfinite(r.std_error[~gone.ravel()]).all()
    q = quality(r, cfg)
    assert q["total_points"] == 87 and q["valid_points"] == 87
    # spatial smoothing averages over measured neighbours only: no NaN leaks in
    sm = smooth_spectra(short.cube, Config(spatial_kernel=5))
    assert np.isfinite(sm[~gone]).all() and np.isnan(sm[gone]).all()


def test_scattered_points_are_still_refused(tmp_path):
    rng = np.random.default_rng(3)
    wn = np.linspace(1000, 1800, 50)
    lines = ["\t\t" + "\t".join(f"{v:.3f}" for v in wn)]
    for _ in range(30):
        lines.append("\t".join([f"{rng.uniform(0, 50):.3f}", f"{rng.uniform(0, 50):.3f}", *["10"] * 50]))
    path = tmp_path / "scatter.txt"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="point lists cannot be mapped"):
        read_text(str(path))


def test_spatial_smoothing_unchanged_on_complete_maps():
    """Over measured neighbours only, but on a complete map it is the same
    normalised Gaussian as before: compare with a direct 2-D sum."""
    from qcarbon.fitter import smooth_spectra
    Y = np.random.default_rng(5).normal(size=(7, 9, 4))
    k, r = 5, 2
    g = np.exp(-0.5 * (np.arange(-r, r + 1) / (k / 4.0)) ** 2)
    ref = np.zeros_like(Y)
    for i in range(7):
        for j in range(9):
            s = w = 0.0
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if 0 <= i + di < 7 and 0 <= j + dj < 9:
                        s = s + g[di + r] * g[dj + r] * Y[i + di, j + dj]
                        w += g[di + r] * g[dj + r]
            ref[i, j] = s / w
    np.testing.assert_allclose(smooth_spectra(Y, Config(spatial_kernel=k)), ref, rtol=1e-12, atol=1e-12)


def test_lab_map_stopped_early_opens():
    m = read_text(vendor("incomplete_map"))
    assert m.meta["missing"] == 3 and np.isfinite(m.cube).all(axis=-1).sum() == 118


def _laser_record(nm):
    """A "Laser" record as LabSpec 6 saves it: name, padding, a "pos" tag 40
    bytes in, and the value as a double 8 bytes after the tag."""
    import struct
    rec = bytearray(b"Laser\x00" + bytes(34) + b"pos" + bytes(5) + struct.pack("<d", nm) + bytes(8))
    return bytes(rec)


def test_laser_wavelength_from_the_acquisition_record():
    # the last record with a plausible value wins; a laser saved as "Off" (0) does not count
    raw = b"LabSpec6" + bytes(100) + _laser_record(785.0) + bytes(50) + _laser_record(532.12) + bytes(20)
    assert labspec.laser_nm(raw) == 532.12
    assert labspec.laser_nm(b"LabSpec6" + _laser_record(0.0)) is None
    assert labspec.laser_nm(b"LabSpec6" + bytes(200)) is None


def test_laser_wavelength_on_lab_files():
    pf, small, grid = vendor("pf_project", "pf_small_project", "g_band_map")
    assert labspec.laser_nm(open(pf, "rb").read()) == 532.12
    assert labspec.laser_nm(open(grid, "rb").read()) == 532.12
    # this project saved only the laser's state
    assert labspec.laser_nm(open(small, "rb").read()) is None
