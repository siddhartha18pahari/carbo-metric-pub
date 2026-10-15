"""The browser engine (web/js/engine, run here with Node) must give the same
numbers as the Python engine. Skips when Node is not installed."""

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from qcarbon.config import Config
from qcarbon.fitter import fit_spectra, shape_eval
from qcarbon.labspec import read_map

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
def test_browser_engine_matches_python(tmp_path):
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
    p = tmp_path / "synthetic map.txt"
    p.write_text("\n".join(lines) + "\n")

    out = subprocess.run([NODE, os.path.join(HERE, "js", "fit_file.mjs"), str(p)],
                         capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    m = read_map(str(p))
    r = fit_spectra(m.cube.reshape(-1, m.wn.size), m.wn, Config())
    assert (js["ny"], js["nx"]) == m.cube.shape[:2]
    assert np.allclose(js["x"], m.x) and np.allclose(js["y"], m.y)
    assert js["held"] == r.held
    k = len(r.names)
    for q in ("intensity", "position", "width", "area"):
        np.testing.assert_allclose(np.reshape(js[q], (-1, k)), getattr(r, q), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(js["std_error"], r.std_error, rtol=1e-9)
