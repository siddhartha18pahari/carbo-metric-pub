"""Read LabSpec 6 binaries (.l6m maps, .l6v video images, .l6i/.l6s result
images) and text exports from any program (LabSpec matrix, x / y / shift /
intensity tables, single spectra).

The container format was reverse engineered in ``ramansp.io``
(github.com/siddhartha18pahari/RAMANSP) and verified bit-identical against the vendor's
own text export. In short: after the ``LabSpec6`` magic the file is a flat
table of 24-byte records ``[type][id][tag][ptr_hi][val][extra]``; a record
tagged ``mat`` is followed, at record start + 24, by a contiguous
little-endian array running up to the next record. ``ptr_hi`` is the high half
of a serialised 64-bit pointer, so it changes per file (ASLR) and is
calibrated per file.

The spatial convention was checked against a LabSpec screenshot of the same
file and against the white-light image the map was taken on:

* the *fast* (inner) stage index is LabSpec's X, the slow one is Y;
* LabSpec draws Y increasing downwards.

Reading the slow axis as x (as ``ramansp.io.read_l6`` did before October
2026) gives the transposed map. For statistics that is harmless; for an
overlay on the optical image it is not.
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass, field

import numpy as np

_MAGIC = b"LabSpec6"
_MAT = b"tam"
_TAGS = (b"tam", b"typm", b"uni", b"axi", b"temx")
_RECORD = 24
_SHIFT_MIN, _SHIFT_MAX = -300.0, 4500.0


@dataclass
class RamanMap:
    """A gridded hyperspectral map: ``cube[i, j, :]`` is the spectrum at
    ``(x[j], y[i])``. Rows follow ``y`` and columns follow ``x``."""

    cube: np.ndarray          # (ny, nx, K) float32 counts
    wn: np.ndarray            # (K,) ascending Raman shift, cm-1
    x: np.ndarray             # (nx,) stage X, um
    y: np.ndarray             # (ny,) stage Y, um
    name: str = ""
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.cube.shape

    @property
    def n_points(self) -> int:
        return int(self.cube.shape[0] * self.cube.shape[1])


@dataclass
class VideoImage:
    """White-light image with its stage axes (um). ``rgb[i, j]`` sits at
    ``(x[j], y[i])``."""

    rgb: np.ndarray           # (H, W, 3) uint8
    x: np.ndarray
    y: np.ndarray


# --------------------------------------------------------------------------
# container walking
# --------------------------------------------------------------------------
def _pointer_word(raw: bytes) -> bytes | None:
    seen: dict[bytes, int] = {}
    for tag in _TAGS:
        start = 0
        while True:
            t = raw.find(tag, start)
            if t < 0:
                break
            start = t + 1
            w = raw[t + 3:t + 7]
            if len(w) == 4 and w[2:] == b"\x00\x00" and w != b"\x00\x00\x00\x00":
                seen[w] = seen.get(w, 0) + 1
    return max(seen, key=seen.get) if seen else None


def _records(raw: bytes, word: bytes) -> list[int]:
    out, start = [], 0
    while True:
        i = raw.find(word, start)
        if i < 0:
            break
        if i >= 12:
            out.append(i - 12)
        start = i + 1
    return out


def _mat_blocks(raw: bytes) -> list[tuple[int, int]]:
    """``(data_offset, n_bytes)`` of every matrix payload, in file order."""
    word = _pointer_word(raw)
    if word is None:
        return []
    recs = sorted(set(_records(raw, word)))
    out = []
    for p in recs:
        if raw[p + 9:p + 12] != _MAT:
            continue
        data = p + _RECORD
        j = bisect.bisect_right(recs, data)
        end = recs[j] if j < len(recs) else len(raw)
        if end - data >= 8:
            out.append((data, end - data))
    return out


def _f32(raw: bytes, off: int, nbytes: int) -> np.ndarray:
    return np.frombuffer(raw, dtype="<f4", count=nbytes // 4, offset=off)


def _is_axis(a: np.ndarray) -> bool:
    return bool(a.size >= 2 and np.all(np.isfinite(a)) and np.all(np.abs(a) < 1e6)
                and (np.all(np.diff(a) > 0) or np.all(np.diff(a) < 0)))


def _is_stage_axis(a: np.ndarray, tol: float = 0.05) -> bool:
    """Monotonic AND evenly stepped: a raster, not a run inside the data."""
    if not _is_axis(a):
        return False
    d = np.diff(a)
    return bool(abs(d.mean()) > 0 and d.std() <= tol * abs(d.mean()))


def _anisotropy(cube: np.ndarray) -> float:
    img = cube.mean(axis=2)
    dv = np.abs(np.diff(img, axis=0)).mean()
    dh = np.abs(np.diff(img, axis=1)).mean()
    return float(dv / dh) if dh > 0 else float("inf")


def _read(path: str) -> bytes:
    with open(path, "rb") as fh:
        raw = fh.read()
    if not raw.startswith(_MAGIC):
        raise ValueError(f"{os.path.basename(path)}: not a LabSpec 6 file")
    return raw


# --------------------------------------------------------------------------
# maps
# --------------------------------------------------------------------------
def read_l6m(path: str) -> RamanMap:
    """Read a gridded LabSpec 6 map. Raises rather than guessing."""
    raw = _read(path)
    blocks = [(o, _f32(raw, o, n)) for o, n in _mat_blocks(raw)]
    if not blocks:
        raise ValueError(f"{os.path.basename(path)}: no matrix records")
    cube_off, payload = max(blocks, key=lambda t: t[1].size)
    if not np.all(np.isfinite(payload)):
        raise ValueError(f"{os.path.basename(path)}: largest matrix is not spectral data")

    axes = [(o, a) for o, a in blocks if o != cube_off and _is_axis(a)]
    spec = [(o, a) for o, a in axes
            if a.size >= 16 and _SHIFT_MIN < min(a[0], a[-1]) and max(a[0], a[-1]) < _SHIFT_MAX
            and payload.size % a.size == 0]
    if not spec:
        raise ValueError(f"{os.path.basename(path)}: no Raman-shift axis fits the data")
    wn_off, wn = max(spec, key=lambda t: t[1].size)
    K = wn.size
    n = payload.size // K
    if n < 4:
        raise ValueError(f"{os.path.basename(path)}: {n} spectra -- not a map")

    # candidate stage axes, in file order (order breaks ties for square maps)
    stage = [(o, a) for o, a in axes if o != wn_off and a.size <= n and _is_stage_axis(a)]
    best = None
    for o_slow, a_slow in stage:
        for o_fast, a_fast in stage:
            if o_slow == o_fast or a_slow.size * a_fast.size != n:
                continue
            cube = payload.reshape(a_slow.size, a_fast.size, K)
            score = abs(np.log(max(_anisotropy(cube), 1e-9)))
            # equal-size axes: the isotropy score cannot tell them apart, so
            # prefer X listed first (as in .l6v/.l6i files)
            key = (round(score, 6), 0 if o_fast < o_slow else 1)
            if best is None or key < best[0]:
                best = (key, cube, a_fast, a_slow)
    if best is None:
        raise ValueError(
            f"{os.path.basename(path)}: {n} spectra but no evenly stepped stage-axis "
            f"pair multiplies to {n} (point list / ParticleFinder maps are not grids)")
    _, cube, x, y = best

    cube = np.asarray(cube, np.float32)
    wn = np.asarray(wn, np.float64)
    if wn[0] > wn[-1]:
        wn, cube = wn[::-1].copy(), cube[..., ::-1].copy()
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    return RamanMap(cube=np.ascontiguousarray(cube), wn=wn, x=x, y=y,
                    name=os.path.splitext(os.path.basename(path))[0], source=path,
                    meta={"format": "l6m", "version": int.from_bytes(raw[12:16], "little")})


TEXT_EXT = (".txt", ".csv", ".tsv", ".dat")
TEXT_LAYOUTS = ("a LabSpec matrix export (first row = Raman shift, then one row per point), "
                "an x / y / shift / intensity table, or a two-column spectrum (shift, intensity)")


def _grid(xs, ys, I, name):
    """Scatter per-point spectra ``I`` (n, K) onto the (y, x) grid they span."""
    ux, uy = np.unique(np.round(xs, 4)), np.unique(np.round(ys, 4))
    if ux.size * uy.size != xs.size:
        raise ValueError(f"{name}: the {xs.size} points do not form a full "
                         f"{uy.size} x {ux.size} grid (point lists cannot be mapped)")
    cube = np.zeros((uy.size, ux.size, I.shape[1]), np.float32)
    cube[np.searchsorted(uy, np.round(ys, 4)), np.searchsorted(ux, np.round(xs, 4))] = I
    return cube, ux.astype(float), uy.astype(float)


def read_text(path: str) -> RamanMap:
    """A map, line scan or single spectrum saved as text by any program.

    Separators: tab, comma, semicolon or spaces. Lines starting with ``#`` and
    a row of column labels are skipped; labels are used to find the x / y /
    shift / intensity columns of a long table when present. Layouts:

    * LabSpec "export as matrix": the first row is the Raman-shift axis, then
      one row per point with two coordinates (LabSpec writes Y first; X is
      taken as the faster-varying column, LabSpec's raster order), one
      coordinate (a line scan) or none.
    * a long table, one row per (x, y, shift, intensity), e.g. Renishaw WiRE.
    * a two-column spectrum (shift, intensity), shown as a single point.
    """
    name = os.path.basename(path)
    rows, labels = [], []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    sep = next(("\t" if "\t" in ln else ";" if ";" in ln else "," if "," in ln else None
                for ln in lines if not ln.startswith("#")), None)
    for ln in lines:
        toks = [t.strip() for t in (ln.split(sep) if sep else ln.split())]
        if sep == ";":
            toks = [t.replace(",", ".") for t in toks]      # decimal commas
        try:
            rows.append(np.array([float(t) for t in toks if t != ""]))
        except ValueError:
            if rows:
                raise ValueError(f"{name}: non-numeric row after the data starts: {ln[:60]!r}") from None
            labels = [t.lstrip("#").strip().lower() for t in toks]
    if len(rows) < 1:
        raise ValueError(f"{name}: no numeric data")

    def done(cube, wn, x, y, layout):
        if wn.size < 10:
            raise ValueError(f"{name}: only {wn.size} spectral channels")
        if wn[0] > wn[-1]:
            wn, cube = wn[::-1].copy(), cube[..., ::-1].copy()
        if not np.all(np.diff(wn) > 0):
            raise ValueError(f"{name}: the Raman-shift axis is not monotonic")
        return RamanMap(cube=np.ascontiguousarray(cube, np.float32), wn=np.asarray(wn, float),
                        x=np.asarray(x, float), y=np.asarray(y, float),
                        name=os.path.splitext(name)[0], source=path,
                        meta={"format": "txt", "layout": layout})

    head, body_widths = rows[0], {r.size for r in rows[1:]}
    if (head.size >= 10 and len(body_widths) == 1 and min(body_widths) - head.size in (0, 1, 2)
            and (np.all(np.diff(head) > 0) or np.all(np.diff(head) < 0))):
        body = np.vstack(rows[1:])
        nc = body.shape[1] - head.size
        coords, I = body[:, :nc], body[:, nc:]
        if nc == 2:
            # the column that changes more often down the file is X
            fast = int(np.count_nonzero(np.diff(coords[:, 1])) > np.count_nonzero(np.diff(coords[:, 0])))
            cube, x, y = _grid(coords[:, fast], coords[:, 1 - fast], I, name)
            return done(cube, head, x, y, "matrix")
        x = coords[:, 0] if nc == 1 else np.arange(I.shape[0], dtype=float)
        return done(I[None], head, x, [0.0], "matrix line" if nc == 1 else "matrix spectra")

    widths = {r.size for r in rows}
    if widths == {4}:
        body = np.vstack(rows)
        col = {"x": 0, "y": 1, "w": 2, "i": 3}
        if len(labels) == 4:
            find = lambda keys: next((i for i, s in enumerate(labels) if any(k in s for k in keys)), None)
            got = {"x": find(("x",)), "y": find(("y",)), "w": find(("wave", "shift", "raman", "cm")),
                   "i": find(("intens", "count", "signal"))}
            if None not in got.values() and len(set(got.values())) == 4:
                col = got
        xs, ys, w, v = (body[:, col[k]] for k in "xywi")
        key = np.round(xs, 4) + 1j * np.round(ys, 4)
        _, first, counts = np.unique(key, return_index=True, return_counts=True)
        K = int(counts[0])
        if np.any(counts != K) or K * first.size != body.shape[0]:
            raise ValueError(f"{name}: points carry different numbers of channels")
        starts = np.sort(first)                         # points in file order
        if np.any(np.diff(starts) != K):
            raise ValueError(f"{name}: each point's channels must be on consecutive rows")
        wn = w[:K]
        if not np.allclose(w.reshape(-1, K), wn, rtol=1e-5, atol=1e-3):
            raise ValueError(f"{name}: points do not share one Raman-shift axis")
        cube, x, y = _grid(xs[starts], ys[starts], v.reshape(-1, K), name)
        return done(cube, wn, x, y, "xy table")
    if widths == {2}:
        body = np.vstack(rows)
        return done(body[None, None, :, 1], body[:, 0], [0.0], [0.0], "spectrum")
    raise ValueError(f"{name}: not a layout this reader knows. Expected {TEXT_LAYOUTS}.")


def read_matrix_txt(path: str) -> RamanMap:
    """Backwards-compatible name for :func:`read_text`."""
    return read_text(path)


def read_map(path: str) -> RamanMap:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".l6m":
        return read_l6m(path)
    if ext in TEXT_EXT:
        return read_text(path)
    raise ValueError(f"unsupported file type {ext}: open a .l6m map or a text export ({', '.join(TEXT_EXT)})")


# --------------------------------------------------------------------------
# white-light video (.l6v) and single-channel result images (.l6i/.l6s)
# --------------------------------------------------------------------------
def read_l6v(path: str) -> VideoImage:
    """The camera frame is stored as uint8 RGB, row-major (H, W, 3), with its
    X axis (W values) listed before its Y axis (H values)."""
    raw = _read(path)
    blocks = _mat_blocks(raw)
    axes = []
    for o, n in blocks:
        a = _f32(raw, o, n)
        if _is_stage_axis(a, tol=0.02):
            axes.append(a.astype(float))
    if len(axes) < 2:
        raise ValueError(f"{os.path.basename(path)}: no stage axes for the video image")
    x, y = axes[0], axes[1]
    need = x.size * y.size * 3
    big = max(blocks, key=lambda t: t[1])
    if big[1] < need:
        raise ValueError(f"{os.path.basename(path)}: image block too small")
    rgb = np.frombuffer(raw, np.uint8, count=need, offset=big[0]).reshape(y.size, x.size, 3)
    return VideoImage(rgb=np.ascontiguousarray(rgb), x=x, y=y)


def read_l6_image(path: str, shape: tuple[int, int]) -> np.ndarray:
    """A single-channel result image (e.g. a vendor peak-area map) saved as
    .l6i/.l6s. Returns ``(ny, nx)`` with X fast. Some files prefix the data
    with a few header words, so the image is taken as the *last* ny*nx values
    of the block."""
    raw = _read(path)
    ny, nx = shape
    need = ny * nx
    for o, n in sorted(_mat_blocks(raw), key=lambda t: -t[1]):
        if n // 4 >= need:
            a = _f32(raw, o, n)[-need:]
            return np.asarray(a, float).reshape(ny, nx)
    raise ValueError(f"{os.path.basename(path)}: no block with {need} values")


def find_video(map_path: str) -> str | None:
    """The .l6v saved next to a map: same stem first, else the only one in the folder."""
    folder = os.path.dirname(map_path)
    stem = os.path.splitext(os.path.basename(map_path))[0]
    same = os.path.join(folder, stem + ".l6v")
    if os.path.exists(same):
        return same
    vids = [f for f in os.listdir(folder) if f.lower().endswith(".l6v")]
    return os.path.join(folder, vids[0]) if len(vids) == 1 else None
