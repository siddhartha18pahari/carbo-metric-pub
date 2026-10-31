"""Text layouts added for CSV fitting: spectra in columns (names from the header), quoted
cells, decimal commas in tab or space separated files, and the server's set export. The
browser reader is checked on the same cases in tests/js/engine.test.mjs."""
import numpy as np

from scarbon.labspec import read_text


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_columns_names_quotes_and_order(tmp_path):
    wn = 1800 - 10 * np.arange(60)                      # descending, as some programs write
    lines = ['"Raman shift","Sample A, spot 1","B"'] + [f"{v},{100 + k},{200 + k}" for k, v in enumerate(wn)]
    m = read_text(_write(tmp_path, "set.csv", "\n".join(lines)))
    assert m.meta["layout"] == "columns" and m.cube.shape == (1, 2, 60)
    assert m.meta["names"] == ["Sample A, spot 1", "B"]
    assert m.wn[0] < m.wn[-1]
    assert m.cube[0, 0, 0] == 100 + 59 and m.cube[0, 1, -1] == 200      # channels reversed with the axis


def test_decimal_commas_with_tabs_and_spaces(tmp_path):
    wn = 1800 - 10 * np.arange(60)
    rows = ["shift\tx\ty\tz"] + ["\t".join(f"{x + 0.5 * k:g}".replace(".", ",") for x in (v, 1, 2, 3)) for k, v in enumerate(wn)]
    t = read_text(_write(tmp_path, "t.txt", "\n".join(rows)))
    assert t.meta["layout"] == "columns" and t.meta["names"] == ["x", "y", "z"]
    assert t.wn[-1] == 1800 and t.cube[0, 0, 58] == 1.5 and t.cube[0, 2, 58] == 3.5
    sp = read_text(_write(tmp_path, "w.txt", "\n".join(f"{v + 0.25:.2f} {3.5 * k:.1f}".replace(".", ",") for k, v in enumerate(wn))))
    assert sp.meta["layout"] == "spectrum" and abs(sp.wn[0] - 1210.25) < 1e-9


def test_matrix_of_rows_is_not_read_as_columns(tmp_path):
    wn = 400 + 7 * np.arange(30)
    rows = ["\t".join(f"{v:g}" for v in wn)] + ["\t".join(f"{(p + 1) ** 2 + k}" for k in range(30)) for p in range(12)]
    m = read_text(_write(tmp_path, "rows.txt", "\n".join(rows)))
    assert m.meta["layout"] == "matrix spectra" and m.cube.shape == (1, 12, 30)


def test_set_names_reach_the_info_and_the_export(tmp_path):
    from fastapi.testclient import TestClient
    from scarbon import server
    from scarbon.config import Config

    wn = np.linspace(1000, 1800, 200)
    lor = lambda c, w: 1 / (1 + (2 * (wn - c) / w) ** 2)
    cols = [20 + 80 * lor(1580, 25) + a * lor(1350, 60) for a in (10, 40)]
    lines = ["Raman shift (cm-1),first,second"] + [f"{v},{cols[0][k]},{cols[1][k]}" for k, v in enumerate(wn)]
    path = _write(tmp_path, "pair.csv", "\n".join(lines))
    c = TestClient(server.app)
    info = c.post("/api/open", json={"path": path}).json()
    assert info["names"] == ["first", "second"] and info["nx"] == 2
    cfg = Config().to_dict()
    cfg["auto_shapes"] = False
    assert c.post(f"/api/ds/{info['id']}/fit", json=cfg).status_code == 200
    for _ in range(600):
        if c.get(f"/api/ds/{info['id']}/fit").json()["state"] != "running":
            break
    text = c.post(f"/api/ds/{info['id']}/export", json={"format": "full"}).text.splitlines()
    assert text[0].startswith("Spectrum,X (um)") and text[1].startswith("first,") and text[2].startswith("second,")
