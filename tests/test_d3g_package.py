"""The D3/G calibration (web/js/d3g-calibration.js, run here with Node) against the
calibration scripts' own results on the files they were made from: the standards, the
particle sample and, where they are on this machine, the uncoated and coated maps.
Paths come from tests/vendor_data.json; missing files skip their test."""

import glob
import json
import math
import os
import shutil
import subprocess

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = shutil.which("node")
BANDS = ["D4", "D1", "D3", "G", "D2"]
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")
pd = pytest.importorskip("pandas")


def _path(key):
    try:
        with open(os.path.join(HERE, "vendor_data.json"), encoding="utf-8") as fh:
            p = json.load(fh).get(key, "")
    except FileNotFoundError:
        pytest.skip("no tests/vendor_data.json")
    if not os.path.exists(p):
        pytest.skip(f"data not on this machine: {key}")
    return p


def _run(tmp_path, req):
    p = tmp_path / "request.json"
    p.write_text(json.dumps(req), encoding="utf-8")
    out = subprocess.run([NODE, os.path.join(HERE, "js", "d3g_file.mjs"), str(p)], capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


def _book(kind):
    """The package's result workbook for the two maps (two per-pixel sheets) or for the particles."""
    pytest.importorskip("openpyxl")
    for b in sorted(glob.glob(os.path.join(_path("d3g_package"), "*", "*results*.xlsx"))):
        names = pd.ExcelFile(b).sheet_names
        pix, parts = [n for n in names if n.startswith("Pixels")], [n for n in names if n.startswith("Particles")]
        if kind == "maps" and len(pix) == 2:
            return b, pix
        if kind == "particles" and parts:
            return b, parts
    pytest.skip(f"no {kind} workbook in the package")


def _baseline(book, name):
    t = pd.read_excel(book, sheet_name="Baselines")
    return {r.iloc[1]: (float(r.iloc[2]), float(r.iloc[3]), int(r.iloc[4])) for _, r in t[t.baseline == name].iterrows()}


def _particles():
    book, sheets = _book("particles")
    return book, pd.read_excel(book, sheet_name=sheets[0])


def _pf_request():
    return {"files": [{"path": _path("d3g_pf_spectra_list"), "project": _path("d3g_pf_project")}]}


def test_bundled_calibration_is_the_package_one(tmp_path):
    book, _ = _book("maps")
    preset = _run(tmp_path, {})["preset"]
    for step, name in zip(preset, ["B0 standards", "B1 median", "B2 median"]):
        want = _baseline(book, name)
        for b in BANDS:
            assert step[b]["pos"] == pytest.approx(want[b][0], rel=1e-14) and step[b]["fwhm"] == pytest.approx(want[b][1], rel=1e-14)


def test_standards_give_the_package_shapes(tmp_path):
    out = _run(tmp_path, {"graphite": _path("d3g_graphite"), "glassy": _path("d3g_glassy")})["standards"]
    want = _baseline(_book("maps")[0], "B0 standards")
    # a free fit of one spectrum: the same answer to fit tolerance (positions and widths on a window
    # limit are exact; the free ones agree to 0.1 cm-1)
    for b in BANDS:
        assert out[b]["pos"] == pytest.approx(want[b][0], abs=0.1), b
        assert out[b]["fwhm"] == pytest.approx(want[b][1], abs=0.1), b


def test_particles_match_the_package(tmp_path):
    _, X = _particles()
    out = _run(tmp_path, _pf_request())["files"][0]
    idx = X["particle"].to_numpy() - 1
    assert out["carbon"] == idx.tolist()
    np.testing.assert_allclose(np.asarray(out["snr"], float)[idx], X["SNR"], rtol=1e-9)
    np.testing.assert_allclose(np.asarray(out["sig"], float)[idx], X["noise"], rtol=1e-9)
    h, se = np.reshape(out["h"], (-1, 5)), np.reshape(out["se"], (-1, 5))
    for j, b in enumerate(BANDS):
        np.testing.assert_allclose(h[:, j], X[f"{b}_h"], atol=1e-7)
        np.testing.assert_allclose(se[:, j], X[f"{b}_se"], rtol=1e-8)
    for k in ("ID3/IG", "AD3/AG", "ID1/IG", "ID4/IG", "ID3/IG_se"):
        np.testing.assert_allclose(np.asarray(out["ratio"][k], float), X[k], atol=1e-9, err_msg=k)
    np.testing.assert_allclose(out["rmse"], X["rmse"], rtol=1e-9)
    assert np.median(out["ratio"]["ID3/IG"]) == pytest.approx(X["ID3/IG"].median(), abs=1e-12)


def test_shapes_from_the_particles_match_the_package_check(tmp_path):
    book, _ = _particles()
    out = _run(tmp_path, _pf_request())["files"][0]
    want = _baseline(book, "Sample 2 own (median)")
    for b in BANDS:
        assert out["own"][b]["support"] == want[b][2], b
        assert out["own"][b]["pos"] == pytest.approx(want[b][0], abs=0.05), b
        assert out["own"][b]["fwhm"] == pytest.approx(want[b][1], abs=0.15), b
    fq = pd.read_excel(book, sheet_name="Fit quality").set_index("baseline")
    row = fq.loc["Sample 2 own (median)"]
    assert np.median(out["own_ratio"]) == pytest.approx(row["median ID3/IG"], abs=3e-4)
    assert np.median(out["own_rn"]) == pytest.approx(row["residual / noise (median)"], abs=0.05)
    assert np.median(out["rn"]) == pytest.approx(fq.loc["B2 median", "residual / noise (median)"], abs=1e-9)


def test_comparison_of_medians_matches_the_package(tmp_path):
    maps, sheets = _book("maps")
    pf, X = _particles()
    U, C = (pd.read_excel(maps, sheet_name=s) for s in sheets)
    grid = lambda T: {"values": T["ID3/IG"].tolist(), "keys": [f"{c // 8},{r // 8}" for c, r in zip(T["col"], T["row"])]}
    stage = {"values": X["ID3/IG"].tolist(), "keys": [f"{math.floor(x / 150)},{math.floor(y / 150)}" for x, y in zip(X["X_um"], X["Y_um"])]}
    pairs = [(grid(C), grid(U)), (stage, grid(U)), (stage, grid(C))]
    out = _run(tmp_path, {"compare": [{"name": str(i), "a": a, "b": b} for i, (a, b) in enumerate(pairs)]})["compare"]
    m = pd.read_excel(maps, sheet_name="Compare medians")
    p = pd.read_excel(pf, sheet_name="Compare medians")
    want = [m[m.metric == "I(D3)/I(G)"].iloc[0]] + [r for _, r in p[p.metric == "I(D3)/I(G)"].iterrows()]
    for got, w in zip(out, want):
        diff = w["difference of medians"] if "difference of medians" in w else w["difference of medians (S2 - ref)"]
        assert got["diff"] == pytest.approx(diff, abs=1e-12)
        # 2 000 resamples of whole blocks each, with another generator: the ends agree to resampling error
        assert got["lo"] == pytest.approx(w["95% CI low (spatial block bootstrap)"], abs=6e-4)
        assert got["hi"] == pytest.approx(w["95% CI high"], abs=6e-4)


def test_maps_and_the_chain_match_the_package(tmp_path):
    book, sheets = _book("maps")
    req = {"graphite": _path("d3g_graphite"), "glassy": _path("d3g_glassy"),
           "files": [{"path": _path("d3g_uncoated_map")}, {"path": _path("d3g_coated_map")}], "chain": [0, 1]}
    out = _run(tmp_path, req)
    for f, sheet in zip(out["files"], sheets):
        T = pd.read_excel(book, sheet_name=sheet)
        assert f["carbon"] == T["pix"].tolist()
        np.testing.assert_allclose(np.asarray(f["ratio"]["ID3/IG"], float), T["ID3/IG"], atol=1e-9)
    for got, name in ((out["chain"]["b1"], "B1 median"), (out["chain"]["b2"], "B2 median")):
        want = _baseline(book, name)
        for b in BANDS:
            assert got[b]["pos"] == pytest.approx(want[b][0], abs=0.1) and got[b]["fwhm"] == pytest.approx(want[b][1], abs=0.3), (name, b)
