"""LabSpec line scans and depth profiles (.l6m), placed by both readers from the axes LabSpec
names (X, Y, Z; 'Spectr' for the Raman shift, 'Points' for a list's index). The files are built
with the record layout and axis names of LabSpec's own maps, and cut from a real map when one is
on this machine (tests/vendor_data.json)."""

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from scarbon import labspec
from scarbon.labspec import PointListError, read_l6m

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = shutil.which("node")
WORD = b"\x10\x27\x00\x00"            # the file's pointer word: every record carries it at offset 12
K = 120
WN = np.linspace(1000.0, 1800.0, K)


def _typm(name):
    """The record naming the matrix after it: 'typm', the pointer word, the name."""
    return bytes(8) + b"typm" + WORD + name.encode().ljust(8, b"\x00")


def _mat(values):
    return bytes(9) + b"tam" + WORD + bytes(8) + np.asarray(values, "<f4").tobytes()


def write_l6m(path, spectra, axes):
    """The spectra (n x K) as LabSpec stores them, then the Raman-shift axis and the named axes."""
    raw = b"LabSpec6" + bytes(40) + _typm("Intens") + _mat(np.ravel(spectra)) + _typm("Spectr") + _mat(WN)
    for name, values in axes:
        raw += _typm(name) + _mat(values)
    path.write_bytes(raw)
    return str(path)


def spectra(n, seed=0):
    return 100 + 50 * np.random.default_rng(seed).random((n, K))


def browser(path):
    """The same file through the browser reader (Node)."""
    out = subprocess.run([NODE, os.path.join(HERE, "js", "read_file.mjs"), path], capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


def same_in_browser(path, m):
    if NODE is None:
        return
    js = browser(path)
    assert (js["ny"], js["nx"], js["k"]) == m.cube.shape
    assert np.allclose(js["x"], m.x[[0, -1]]) and np.allclose(js["y"], m.y[[0, -1]])
    assert js["scan"] == m.meta["scan"] and js["axes"] == m.meta["axes"]
    assert js["cube_probe"] == pytest.approx(float(m.cube.reshape(-1)[::997].astype(np.float64).sum()), rel=1e-9)


N = 200                                   # longer than the Raman-shift axis, so a stage axis could pass for one
XS = np.linspace(-20, 30, N)


@pytest.mark.parametrize("axes,shape,x,y,scan,names", [
    ([("X", XS), ("X", XS), ("Y", [5.0])], (1, N), XS, [5.0], "line", ["X"]),
    ([("X", XS)], (1, N), XS, [0.0], "line", ["X"]),
    ([("X", [2.0]), ("Y", XS)], (N, 1), [2.0], XS, "line", ["Y"]),
    ([("X", [2.0]), ("Y", [3.0]), ("Z", XS[::-1])], (N, 1), [2.0], XS[::-1], "depth", ["Z"]),
], ids=["along X, Y one point", "along X, no Y", "along Y", "depth along Z"])
def test_line_and_depth_scans(tmp_path, axes, shape, x, y, scan, names):
    S = spectra(N)
    m = read_l6m(write_l6m(tmp_path / "scan.l6m", S, axes))
    assert m.cube.shape == (*shape, K)
    np.testing.assert_allclose(m.x, x, rtol=1e-6); np.testing.assert_allclose(m.y, y, rtol=1e-6)
    np.testing.assert_allclose(m.cube.reshape(N, K), S, rtol=1e-6)       # in acquisition order
    np.testing.assert_allclose(m.wn, WN, rtol=1e-6)
    assert m.meta["scan"] == scan and m.meta["axes"] == names
    same_in_browser(str(tmp_path / "scan.l6m"), m)


def test_a_line_at_an_angle_is_placed_by_its_distance(tmp_path):
    t = np.linspace(0, 1, N)
    m = read_l6m(write_l6m(tmp_path / "diag.l6m", spectra(N), [("X", 3 + 30 * t), ("Y", -4 + 40 * t)]))
    assert m.cube.shape == (1, N, K) and m.meta["axes"] == ["XY"]
    np.testing.assert_allclose(m.x, 50 * t, rtol=1e-5, atol=1e-5)          # 3-4-5: 50 um long
    same_in_browser(str(tmp_path / "diag.l6m"), m)


@pytest.mark.parametrize("second,scan", [("Y", "map"), ("Z", "depth")])
def test_maps_say_which_axes_they_span(tmp_path, second, scan):
    nx, ny = 10, 7
    xs, ys = np.linspace(0, 9, nx), np.linspace(-6, 0, ny)
    m = read_l6m(write_l6m(tmp_path / "map.l6m", spectra(nx * ny, 1), [("X", xs), (second, ys), ("X", xs)]))
    assert m.cube.shape == (ny, nx, K)
    assert m.meta["scan"] == scan and m.meta["axes"] == ["X", second]
    same_in_browser(str(tmp_path / "map.l6m"), m)


def test_named_axes_set_a_maps_orientation(tmp_path):
    """Stripes along X: read the wrong way round, the map looks more even, so the isotropy score
    alone would pick it. LabSpec's names (X fast, rows along Y) read it as stored."""
    nx, ny = 12, 5
    stripes = np.tile(1000 * (np.arange(nx) % 2), ny)[:, None] + spectra(nx * ny, 2)
    m = read_l6m(write_l6m(tmp_path / "stripes.l6m", stripes, [("X", np.arange(nx) * 1.0), ("Y", np.arange(ny) * 2.0), ("X", np.arange(nx) * 1.0)]))
    assert m.cube.shape == (ny, nx, K) and m.meta["axes"] == ["X", "Y"]
    np.testing.assert_allclose(m.cube.reshape(-1, K), stripes, rtol=1e-6)
    same_in_browser(str(tmp_path / "stripes.l6m"), m)


def test_lists_and_stopped_maps_are_not_taken_for_lines(tmp_path):
    # a ParticleFinder list: 'Points' 1..n, longer than the Raman shift; the shift is still found by name
    with pytest.raises(PointListError) as e:
        read_l6m(write_l6m(tmp_path / "list.l6m", spectra(N), [("Points", np.arange(1, N + 1)), ("Points", np.arange(1, N + 1))]))
    assert e.value.wn.size == K and e.value.spectra.shape == (N, K)
    # a map's autosave before its Y axis was written: 16 X positions for 320 spectra
    with pytest.raises(PointListError):
        read_l6m(write_l6m(tmp_path / "stopped.l6m", spectra(320), [("X", np.linspace(0, 15, 16)), ("X", np.linspace(0, 15, 16))]))
    # positions that do not run one way are not a line
    with pytest.raises(PointListError):
        read_l6m(write_l6m(tmp_path / "zigzag.l6m", spectra(N), [("X", np.tile([0.0, 1.0], N // 2)), ("Y", [0.0])]))


def _vendor(key):
    try:
        with open(os.path.join(HERE, "vendor_data.json"), encoding="utf-8") as fh:
            p = json.load(fh).get(key, "")
    except FileNotFoundError:
        pytest.skip("no tests/vendor_data.json")
    if not os.path.exists(p):
        pytest.skip(f"data not on this machine: {p}")
    return p


def _cut(raw, keep, single, rename=None):
    """A real map turned into a scan: the spectra cut to `keep` (indices in file order), the named
    axes in `single` cut to their first point, and the axis `rename` = (old, new) renamed."""
    blocks = labspec._mat_blocks(raw, 4)
    cube_off, cube_nb = max(blocks, key=lambda b: b[1])
    K0 = None
    edits = {}
    for off, nb in blocks:
        name = labspec._axis_type(raw, off)
        if name == "Spectr":
            K0 = nb // 4
        if name in single and off != cube_off:
            edits[off] = (nb, raw[off:off + 4])
    payload = np.frombuffer(raw, "<f4", cube_nb // 4, cube_off).reshape(-1, K0)
    edits[cube_off] = (cube_nb, payload[keep].astype("<f4").tobytes())
    out = raw
    for off in sorted(edits, reverse=True):
        nb, new = edits[off]
        out = out[:off] + new + out[off + nb:]
    if rename:
        old, new = rename
        for off, _ in labspec._mat_blocks(out, 4):
            if labspec._axis_type(out, off) == old:
                rec = off - labspec._RECORD
                t = out.rfind(b"typm", max(0, rec - 160), rec)
                out = out[:t + 8] + new.encode() + out[t + 8 + len(old):]
    return out


def test_a_map_of_noise_reads_as_labspec_draws_it():
    """A real 66 x 26 map of noise only: LabSpec draws it 66 wide (X -25 to 27 um) and 26 tall;
    the isotropy score alone read it the other way round."""
    m = read_l6m(_vendor("named_axes_map"))
    assert m.cube.shape[:2] == (26, 66)
    assert m.x.size == 66 and m.x.min() < -24 and m.x.max() > 27 and m.y.size == 26
    same_in_browser(_vendor("named_axes_map"), m)


def test_scans_cut_from_a_real_map(tmp_path):
    """A real LabSpec map (X stored twice, then Y) cut to its first row, its first column, and that
    column renamed Z: each read as a line or a depth profile with the map's own spectra."""
    path = _vendor("twin_axis_map")
    grid = read_l6m(path)
    assert grid.meta["axes"] == ["X", "Y"] and grid.meta["scan"] == "map"
    ny, nx = grid.cube.shape[:2]
    raw = open(path, "rb").read()
    cases = [("row", list(range(nx)), {"Y"}, None, (1, nx), grid.cube[:1], "line", ["X"]),
             ("column", [r * nx for r in range(ny)], {"X"}, None, (ny, 1), grid.cube[:, :1], "line", ["Y"]),
             ("depth", [r * nx for r in range(ny)], {"X"}, ("Y", "Z"), (ny, 1), grid.cube[:, :1], "depth", ["Z"])]
    for tag, keep, single, rename, shape, want, scan, names in cases:
        p = tmp_path / f"{tag}.l6m"
        p.write_bytes(_cut(raw, keep, single, rename))
        m = read_l6m(str(p))
        assert m.cube.shape[:2] == shape, tag
        np.testing.assert_array_equal(m.cube, want)
        assert m.meta["scan"] == scan and m.meta["axes"] == names
        np.testing.assert_allclose(m.x if shape[0] == 1 else m.y, grid.x if shape[0] == 1 else grid.y)
        same_in_browser(str(p), m)
