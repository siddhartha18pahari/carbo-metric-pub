"""The browser engine (web/js/engine, run here with Node) must give the same
numbers as the Python engine. Skips when Node is not installed."""

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from qcarbon.config import Config
from qcarbon.fitter import fit_spectra, shape_eval, smooth_spectra
from qcarbon.labspec import read_map

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
@pytest.mark.parametrize("overrides,missing", [({}, 0), ({"smooth_window": 9, "smooth_order": 3, "spatial_kernel": 3}, 0),
                                              ({"spatial_kernel": 3}, 3)],
                         ids=["raw", "smoothed", "stopped-early"])
def test_browser_engine_matches_python(tmp_path, overrides, missing):
    rng = np.random.default_rng(7)
    wn = np.linspace(1000, 1800, 240)
    ny, nx = 14, 16
    yy, xx = np.mgrid[0:ny, 0:nx]
    part = ((yy - 7) ** 2 + (xx - 8) ** 2) < 30
    lines = ["\t\t" + "\t".join(f"{v:.4f}" for v in wn)]
    for i in range(ny):
        for j in range(nx):
            y = 30 + 25 * np.exp(-(wn - 1000) / 500.0)
            if part[i, j]:
                y = y + 60 * shape_eval(wn, 1580, 20, "Lorentz") + 25 * shape_eval(wn, 1350, 50, "Lorentz") \
                      + 8 * shape_eval(wn, 1615, 25, "Lorentz") + 4 * shape_eval(wn, 1500, 120, "Gauss")
            lines.append("\t".join([f"{-3 + i:.4f}", f"{5 + 0.5 * j:.4f}", *map(str, rng.poisson(y))]))
    if missing:                       # the acquisition stopped before the last points
        lines = lines[:-missing]
    p = tmp_path / "synthetic map.txt"
    p.write_text("\n".join(lines) + "\n")

    out = subprocess.run([NODE, os.path.join(HERE, "js", "fit_file.mjs"), str(p), json.dumps(overrides)],
                         capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    m = read_map(str(p))
    cfg = Config(**overrides)
    r = fit_spectra(smooth_spectra(m.cube, cfg).reshape(-1, m.wn.size), m.wn, cfg)
    assert (js["ny"], js["nx"]) == m.cube.shape[:2]
    assert np.allclose(js["x"], m.x) and np.allclose(js["y"], m.y)
    assert js["held"] == r.held
    k = len(r.names)
    for q in ("intensity", "position", "width", "area"):
        np.testing.assert_allclose(np.reshape(js[q], (-1, k)).astype(float), getattr(r, q), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(np.asarray(js["std_error"], float), r.std_error, rtol=1e-9)
    assert int(np.isnan(r.std_error).sum()) == missing


def _vendor_path(key):
    try:
        with open(os.path.join(HERE, "vendor_data.json"), encoding="utf-8") as fh:
            p = json.load(fh).get(key, "")
    except FileNotFoundError:
        pytest.skip("no tests/vendor_data.json")
    if not os.path.exists(p):
        pytest.skip(f"data not on this machine: {key}")
    return p


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
@pytest.mark.parametrize("key", ["pf_small_project", "antistokes_map", "twin_axis_map"])
def test_browser_reader_matches_python_on_lab_files(key):
    from qcarbon.labspec import read_any
    p = _vendor_path(key)
    out = subprocess.run([NODE, os.path.join(HERE, "js", "read_file.mjs"), p], capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    m, video = read_any(p)
    assert (js["ny"], js["nx"], js["k"]) == m.cube.shape
    assert np.allclose(js["wn"], m.wn[[0, -1]]) and np.allclose(js["x"], m.x[[0, -1]]) and np.allclose(js["y"], m.y[[0, -1]])
    flat = m.cube.reshape(-1)
    assert js["cube_probe"] == pytest.approx(float(flat[::997].astype(np.float64).sum()), rel=1e-9)
    if "points" in m.meta:
        for k, v in js["points"].items():
            np.testing.assert_allclose(v, m.meta["points"][k][:5], rtol=1e-6)
        assert (js["video"]["w"], js["video"]["h"]) == (video.x.size, video.y.size)
        # the same pixels too (an untagged mosaic once started 16 pixels apart in the two readers)
        flat = video.rgb.reshape(-1).astype(np.int64)[::997]
        assert js["video"]["rgb_probe"] == int((flat * (np.arange(0, video.rgb.size, 997) % 7 + 1)).sum())
