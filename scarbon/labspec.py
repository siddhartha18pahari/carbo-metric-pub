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
import math
import os
import struct
from dataclasses import dataclass, field

import numpy as np

_MAGIC = b"LabSpec6"
_MAT = b"tam"
_TAGS = (b"tam", b"typm", b"uni", b"axi", b"temx")
_RECORD = 24
# spectra recorded through the laser line start in the anti-Stokes region
# (e.g. -1130 to 1939 cm-1); a -300 floor made those maps unreadable
_SHIFT_MIN, _SHIFT_MAX = -4000.0, 4500.0


class PointListError(ValueError):
    """A .l6m holding a list of spectra with no stage grid (ParticleFinder)."""


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


def laser_nm(raw: bytes) -> float | None:
    """The excitation wavelength saved with the acquisition, in nm: a "Laser"
    record whose value is the double 8 bytes after its "pos" tag. Some projects
    save only the laser's state ("Off", value 0): None then."""
    end = len(raw)
    while True:
        i = raw.rfind(b"Laser\x00", 0, end)
        if i < 0:
            return None
        j = raw.find(b"pos", i, i + 80)
        if 0 <= j and j + 16 <= len(raw):
            v = struct.unpack_from("<d", raw, j + 8)[0]
            if 200.0 < v < 1200.0:
                return round(float(v), 3)
        end = i


_T_INT, _T_DOUBLE, _T_INLINE, _T_ARRAY, _T_STRING = 4, 5, 7, 0x1006, 0x1007


def _fill_gap(queue: list, gap: int, recs: list) -> tuple[list, int] | None:
    """The pending payloads that fill one gap between records: the longest leading
    run of the queue whose sizes add up to the gap exactly, allowing strings to be
    skipped (a string shared by two records is written once). None if nothing fits."""
    for cut in range(len(queue), 0, -1):
        head = queue[:cut]
        strings = [i for i, (k, _) in enumerate(head) if recs[k][1] == _T_STRING][-12:]
        need = sum(n for _, n in head) - gap
        if need < 0:
            continue
        best = None
        for mask in range(1 << len(strings)):
            drop = {strings[b] for b in range(len(strings)) if mask >> b & 1}
            if sum(head[i][1] for i in drop) == need and (best is None or len(drop) < len(best)):
                best = drop
        if best is not None:
            return [e for i, e in enumerate(head) if i not in best], cut
    return None


def settings(raw: bytes) -> list[list]:
    """The settings saved with the acquisition (exposure, accumulations, grating,
    objective, filter, hole, autofocus, the calibration's time stamp, ...) as
    [name, value] in file order, repeats kept: a ParticleFinder project holds the
    camera's settings, then a block per acquisition. Each is a name record (tag
    "namm") beside a value record (tag "?al}"): an integer, a double, a short string
    held in the record itself, or a string written into a gap between records.
    Payloads fill the gaps in the order they were declared, so the gaps are filled
    first (checked by exact byte accounting on the lab's files)."""
    word = _pointer_word(raw)
    if not word:
        return []
    recs = []
    for p in sorted(set(_records(raw, word))):
        if p + _RECORD <= len(raw):
            typ, val, extra = struct.unpack_from("<I", raw, p)[0], *struct.unpack_from("<II", raw, p + 16)
            recs.append((p, typ, raw[p + 8:p + 12], val, extra))
    placed, queue = {}, []
    for k, (p, typ, _tag, _val, extra) in enumerate(recs):
        if typ == _T_ARRAY:
            queue.append((k, 4 * extra))
        elif typ == _T_STRING:
            queue.append((k, extra))
        end = p + _RECORD
        gap = (recs[k + 1][0] if k + 1 < len(recs) else len(raw)) - end
        if gap <= 0:
            continue
        fit = _fill_gap(queue, gap, recs) if queue else None
        if fit is None:
            queue = [e for e in queue if recs[e[0]][1] == _T_ARRAY]   # arrays keep waiting; strings are dropped
            continue
        used, cut = fit
        for kk, n in used:
            placed[kk] = (end, n)
            end += n
        queue = queue[cut:]

    def value(k):
        p, typ, _tag, val, extra = recs[k]
        if typ == _T_INT:
            return struct.unpack_from("<i", raw, p + 16)[0]
        if typ == _T_DOUBLE:
            v = struct.unpack_from("<d", raw, p + 16)[0]
            return v if math.isfinite(v) else None
        if typ == _T_INLINE:
            return raw[p + 16:p + 24].split(b"\x00")[0].decode("latin-1")
        if typ == _T_STRING and k in placed:
            off, n = placed[k]
            return raw[off:off + n].split(b"\x00")[0].decode("latin-1")
        return None

    out, used = [], set()
    for k, rec in enumerate(recs):
        if rec[2] != b"namm":
            continue
        name = value(k)
        if not name or not isinstance(name, str) or not name.isprintable():
            continue
        for j in (k - 1, k + 1):
            if 0 <= j < len(recs) and j not in used and recs[j][2][1:] == b"al}":
                used.add(j)
                out.append([name.strip(), value(j)])
                break
    return out


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

    def grid_for(wn_off, K):
        """Best (fast, slow) stage-axis pair for K channels, or None."""
        n = payload.size // K
        # candidate stage axes, in file order (order breaks ties for square maps)
        stage = [(o, a) for o, a in axes if o != wn_off and a.size <= n and _is_stage_axis(a)]
        best = None
        for o_slow, a_slow in stage:
            for o_fast, a_fast in stage:
                if o_slow == o_fast or a_slow.size * a_fast.size != n:
                    continue
                cube = payload.reshape(a_slow.size, a_fast.size, K)
                score = abs(np.log(max(_anisotropy(cube), 1e-9)))
                # Some files store the X axis twice (X, Y, X again), so an
                # identical pair is a last resort; a centred square map can
                # legitimately have X == Y. Equal-size axes: the isotropy score
                # cannot tell them apart, so prefer X listed first (as in .l6v/.l6i).
                same = int(np.array_equal(a_slow, a_fast))
                key = (same, round(score, 6), 0 if o_fast < o_slow else 1)
                if best is None or key < best[0]:
                    best = (key, cube, a_fast, a_slow)
        return best

    # The spectral axis is the one that leaves a consistent X x Y grid. Taking
    # simply the longest "plausible" axis can pick a stage axis instead (a
    # 91-point X axis divides the payload too).
    best = wn = None
    for wn_off, cand in sorted(spec, key=lambda t: -t[1].size):
        best = grid_for(wn_off, cand.size)
        if best is not None:
            wn = cand
            break
    if best is None:
        wn = max(spec, key=lambda t: t[1].size)[1]
        n = payload.size // wn.size
        err = PointListError(
            f"{os.path.basename(path)}: {n} spectra with no stage grid. This is a "
            f"ParticleFinder spectra list: open its .l6pf project, which holds the particle positions")
        err.spectra, err.wn = payload.reshape(n, wn.size), np.asarray(wn, np.float64)
        err.settings = settings(raw)
        raise err
    K = wn.size
    _, cube, x, y = best

    cube = np.asarray(cube, np.float32)
    wn = np.asarray(wn, np.float64)
    if wn[0] > wn[-1]:
        wn, cube = wn[::-1].copy(), cube[..., ::-1].copy()
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    return RamanMap(cube=np.ascontiguousarray(cube), wn=wn, x=x, y=y,
                    name=os.path.splitext(os.path.basename(path))[0], source=path,
                    meta={"format": "l6m", "version": int.from_bytes(raw[12:16], "little"),
                          "laser_nm": laser_nm(raw), "settings": settings(raw)})


# --------------------------------------------------------------------------
# ParticleFinder projects (.l6pf)
# --------------------------------------------------------------------------
# A .l6pf holds the mosaic camera image (uint8 RGB with its X and Y stage
# axes), one spectrum per particle (with a 1..P index axis), and a particle
# table of P records of 24 float32. Field meanings were pinned against the
# instrument (a 4575-particle project: seven rows of LabSpec's own Results table, matched on
# their "Red cursor" value, agree on all four morphology columns at the same
# index, so record k describes spectrum k):
PF_FIELDS = {"x": 0, "y": 1, "area": 2, "ecd": 3, "feret_max": 10, "min_diam": 11, "convexity": 15,
             "elongation": 16, "volume": 18, "xmin": 19, "ymin": 20, "xmax": 21, "ymax": 22,
             "cls": 23}
PF_LABELS = {"ecd": ("Diameter (circle equiv.)", "µm"), "area": ("Area", "µm²"),
             "min_diam": ("Min diameter", "µm"), "convexity": ("Convexity", ""),
             "elongation": ("Elongation", "%"), "volume": ("Volume", "µm³")}
_PF_REC = 24          # floats per particle record


def _pf_angles(pts: dict, video: "VideoImage | None") -> np.ndarray:
    r"""The tilt of each particle's long axis, in radians from +X towards +Y
    (stage coordinates). The table has no angle, but it has the Feret length
    and width and the bounding box: an ellipse of those axes at tilt t has a
    box of half sides sqrt(a² cos² t + b² sin² t) and sqrt(a² sin² t + b² cos² t),
    which gives |t|. Which way it leans (/ or \) comes from the mosaic: the sign
    of the covariance of x and y over the box's pixels, each weighted by how far
    it is from the background around the box. The sums are integers, so both
    readers agree exactly. Round particles (length under 1.05 x width) get 0."""
    P = pts["x"].size
    out = np.zeros(P)
    if video is not None:
        H, W = video.rgb.shape[:2]
        x0, y0 = float(video.x[0]), float(video.y[0])
        dx, dy = float(video.x[1]) - x0, float(video.y[1]) - y0
        col = lambda v: min(W - 1, max(0, math.floor((v - x0) / dx + 0.5)))
        row = lambda v: min(H - 1, max(0, math.floor((v - y0) / dy + 0.5)))
    for p in range(P):
        L, D = float(pts["feret_max"][p]), float(pts["min_diam"][p])
        if not (D > 0 and L > 1.05 * D):
            continue
        a2, b2 = L * L / 4, D * D / 4
        hw, hh = (pts["xmax"][p] - pts["xmin"][p]) / 2, (pts["ymax"][p] - pts["ymin"][p]) / 2
        c2 = min(1.0, max(0.0, ((hw * hw - b2) / (a2 - b2) + 1 - (hh * hh - b2) / (a2 - b2)) / 2))
        t = math.acos(math.sqrt(c2))
        if video is not None:
            qa, qb, ra, rb = col(pts["xmin"][p]), col(pts["xmax"][p]), row(pts["ymin"][p]), row(pts["ymax"][p])
            q0, q1 = max(0, min(qa, qb) - 1), min(W - 1, max(qa, qb) + 1)
            r0, r1 = max(0, min(ra, rb) - 1), min(H - 1, max(ra, rb) + 1)
            R0, R1, Q0, Q1 = max(0, r0 - 2), min(H - 1, r1 + 2), max(0, q0 - 2), min(W - 1, q1 + 2)
            outer = np.ones((R1 - R0 + 1, Q1 - Q0 + 1), bool)          # the background: two pixels around the box
            outer[r0 - R0:r1 - R0 + 1, q0 - Q0:q1 - Q0 + 1] = False
            I = video.rgb[R0:R1 + 1, Q0:Q1 + 1].astype(np.int64).sum(axis=2)
            ring = np.sort(I[outer])
            if ring.size:
                bg = int(ring[(ring.size - 1) // 2])
                w = np.abs(I[r0 - R0:r1 - R0 + 1, q0 - Q0:q1 - Q0 + 1] - bg)
                v, u = np.mgrid[0:r1 - r0 + 1, 0:q1 - q0 + 1]
                sw, sq, sr, sqr = int(w.sum()), int((w * u).sum()), int((w * v).sum()), int((w * u * v).sum())
                c = sw * sqr - sq * sr                # sw² x the covariance, in pixel steps
                t = -t if (c < 0) != (dx * dy < 0) else t
        out[p] = t
    return out


def _pf_seams(pts: dict, video: "VideoImage | None") -> np.ndarray:
    """Objects lying on the seams of a stitched mosaic: the borders between camera tiles
    show as dark lines that ParticleFinder can take for particles. A seam is a column (or
    row) whose median brightness lies at least 10 grey levels (30 in R+G+B) below the
    running median of the 41 lines around it, widened by one line. An object is on a seam
    when its centre is within one pixel of one, or when it runs along one (within 15
    degrees of its direction, at least 2.5 times longer than wide) and the seam passes
    within half its width. Integer medians, so both readers agree exactly."""
    P = pts["x"].size
    out = np.zeros(P, np.uint8)
    if video is None:
        return out
    # medians over every step-th pixel of each line (at most about 2000), as the browser reader does
    H, W = video.rgb.shape[:2]
    sr, sc = max(1, -(-H // 2000)), max(1, -(-W // 2000))
    col_med = np.median(video.rgb[::sr].astype(np.int32).sum(axis=2), axis=0)
    row_med = np.median(video.rgb[:, ::sc].astype(np.int32).sum(axis=2), axis=1)

    def seams_of(med):
        pad = np.pad(med, 20, mode="symmetric")                 # scipy's "reflect"
        run = np.median(np.lib.stride_tricks.sliding_window_view(pad, 41), axis=1)
        flag = med < run - 30
        wide = flag.copy()
        wide[1:] |= flag[:-1]
        wide[:-1] |= flag[1:]
        return wide

    cols, rows = seams_of(col_med), seams_of(row_med)
    dx, dy = float(video.x[1] - video.x[0]), float(video.y[1] - video.y[0])
    xs = float(video.x[0]) + np.flatnonzero(cols) * dx
    ys = float(video.y[0]) + np.flatnonzero(rows) * dy
    if not xs.size and not ys.size:
        return out
    for p in range(P):
        L, D = float(pts["feret_max"][p]), float(pts["min_diam"][p])
        deg = abs(float(pts["angle"][p])) * 180 / math.pi
        dv = float(np.min(np.abs(xs - pts["x"][p]))) if xs.size else math.inf
        dh = float(np.min(np.abs(ys - pts["y"][p]))) if ys.size else math.inf
        reach, long = D / 2 + abs(dx), L >= 2.5 * D
        out[p] = dv <= abs(dx) or dh <= abs(dy) or (long and deg >= 75 and dv <= reach) or (long and deg <= 15 and dh <= reach)
    return out


def _pf_table(raw: bytes, P: int, lo: int, hi: int) -> int | None:
    """Byte offset of P consecutive particle records inside raw[lo:hi].

    Records are recognised by an identity that cannot hold by chance: field 3
    (circle-equivalent diameter) = sqrt(4 * field 2 (area) / pi). Records are
    not 4-byte aligned in the file, so each byte alignment is tried."""
    chunk, overlap = 1 << 24, P * _PF_REC + 64
    for al in range(4):
        start = lo + ((al - lo) % 4)
        n_all = (hi - start) // 4
        for c0 in range(0, n_all, chunk - overlap):
            n = min(chunk, n_all - c0)
            f = np.frombuffer(raw, "<f4", count=n, offset=start + 4 * c0)
            a, e = f[2:-1], f[3:]
            with np.errstate(all="ignore"):
                ok = (a > 0) & (a < 1e9) & (np.abs(e * e - np.float32(4 / np.pi) * a) <= np.float32(1e-3) * a)
            for r in range(_PF_REC):
                col = ok[r::_PF_REC]
                if col.size < P:
                    continue
                cs = np.concatenate([[0], np.cumsum(col, dtype=np.int64)])
                hit = np.flatnonzero(cs[P:] - cs[:-P] == P)
                if hit.size:
                    return start + 4 * (c0 + r + _PF_REC * int(hit[0]))
    return None


def read_l6pf(path: str) -> tuple[RamanMap, VideoImage | None]:
    """A ParticleFinder project: particle spectra, positions, morphology and
    the mosaic image. The map is a 1 x P list of spectra (fit and statistics
    work unchanged); positions and morphology are in ``meta["points"]``."""
    raw = _read(path)
    name = os.path.basename(path)
    blocks = _mat_blocks(raw)
    arrays = [(o, n, _f32(raw, o, n)) for o, n in blocks if n < 64 << 20]
    axes = [(o, a) for o, n, a in arrays if _is_axis(a)]
    index = [(o, a) for o, a in axes if a[0] == 1 and np.all(np.diff(a) == 1)]
    if not index:
        raise ValueError(f"{name}: no particle index in this ParticleFinder project")
    P = int(max(a.size for _, a in index))
    specs = [(o, a) for o, a in axes
             if a.size >= 16 and not (a[0] == 1 and np.all(np.diff(a) == 1))
             and _SHIFT_MIN < min(a[0], a[-1]) and max(a[0], a[-1]) < _SHIFT_MAX]
    found = None
    for wo, wn in sorted(specs, key=lambda t: -t[1].size):
        for o, n in blocks:
            if n // 4 == P * wn.size:
                found = (o, wn, wo)
                break
        if found:
            break
    if found is None:
        raise ValueError(f"{name}: {P} particles but no matching block of spectra "
                         f"(the project may hold images only)")
    s_off, wn, wn_off = found
    K = wn.size
    spectra = np.frombuffer(raw, "<f4", count=P * K, offset=s_off).reshape(P, K)

    # mosaic: two evenly stepped stage axes W, H and W*H*3 bytes of RGB
    video, image_end = None, 0
    stage = [(o, a) for o, a in axes if o != wn_off and a.size > 16 and _is_stage_axis(a, tol=0.02)
             and not (a[0] == 1 and np.all(np.diff(a) == 1))]
    for i in range(len(stage) - 1):
        (xo, xa), (yo, ya) = stage[i], stage[i + 1]
        need = xa.size * ya.size * 3
        img_off = next((o for o, n in blocks if n == need), None)
        # not tagged as a matrix in some versions: it then fills the gap between two records
        # exactly (on every project checked); failing that, it ends just before the X axis
        if img_off is None:
            word = _pointer_word(raw)
            recs = sorted(set(_records(raw, word))) if word else []
            img_off = next((p + _RECORD for p, q in zip(recs, recs[1:]) if q - (p + _RECORD) == need), None)
        if img_off is None:
            rec = xo - _RECORD
            best = None
            for st in range(max(0, rec - need - 1024), rec - need + 1):
                a8 = np.frombuffer(raw, np.uint8, count=need, offset=st).reshape(ya.size, xa.size, 3)[::16, ::16]
                sm = float(np.abs(np.diff(a8.astype(np.int16), axis=1)).mean())
                if best is None or sm < best[0]:
                    best = (sm, st)
            img_off = best[1] if best else None
        if img_off is not None:
            rgb = np.frombuffer(raw, np.uint8, count=need, offset=img_off).reshape(ya.size, xa.size, 3)
            video = VideoImage(rgb=rgb, x=xa.astype(float), y=ya.astype(float))
            image_end = img_off + need
            break

    t_off = _pf_table(raw, P, image_end, len(raw))
    if t_off is None:
        raise ValueError(f"{name}: no particle table with {P} records")
    T = np.frombuffer(raw, "<f4", count=P * _PF_REC, offset=t_off).reshape(P, _PF_REC).astype(float)
    pts = {k: T[:, c].copy() for k, c in PF_FIELDS.items()}
    inside = ((pts["x"] >= pts["xmin"] - 1e-3) & (pts["x"] <= pts["xmax"] + 1e-3)
              & (pts["y"] >= pts["ymin"] - 1e-3) & (pts["y"] <= pts["ymax"] + 1e-3))
    if inside.mean() < 0.95:
        raise ValueError(f"{name}: particle table found but positions fall outside their boxes")
    pts["angle"] = _pf_angles(pts, video)
    pts["seam"] = _pf_seams(pts, video)

    wn = np.asarray(wn, np.float64)
    cube = np.ascontiguousarray(spectra[None].astype(np.float32))
    if wn[0] > wn[-1]:
        wn, cube = wn[::-1].copy(), np.ascontiguousarray(cube[..., ::-1])
    m = RamanMap(cube=cube, wn=wn, x=np.arange(1, P + 1, dtype=float), y=np.zeros(1),
                 name=os.path.splitext(name)[0], source=path,
                 meta={"format": "l6pf", "kind": "particles", "points": pts, "laser_nm": laser_nm(raw),
                       "settings": settings(raw)})
    return m, video


def pf_for_spectra_list(l6m_path: str, n_spectra: int) -> str | None:
    """The .l6pf saved with a ParticleFinder spectra list (.l6m): same folder,
    same particle count. Counting needs only the project's small index axis."""
    folder = os.path.dirname(l6m_path)
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(".l6pf"):
            continue
        p = os.path.join(folder, f)
        try:
            raw = _read(p)
        except ValueError:
            continue
        for o, n in _mat_blocks(raw):
            if n // 4 == n_spectra and n < 64 << 20:
                a = _f32(raw, o, n)
                if a[0] == 1 and np.all(np.diff(a) == 1):
                    return p
    return None


TEXT_EXT = (".txt", ".csv", ".tsv", ".dat")
TEXT_LAYOUTS = ("a LabSpec matrix export (first row = Raman shift, then one row per point), "
                "an x / y / shift / intensity table, or a two-column spectrum (shift, intensity)")


def _grid(xs, ys, I, name):
    """Scatter per-point spectra ``I`` (n, K) onto the (y, x) grid they span.

    A map stopped before its last points (LabSpec writes only what it
    measured) still sits on its grid: the cells never measured are NaN and
    stay out of the fit and the statistics. Points that do not share a grid,
    or fill less than half of it, are refused."""
    ux, uy = np.unique(np.round(xs, 4)), np.unique(np.round(ys, 4))
    ii, jj = np.searchsorted(uy, np.round(ys, 4)), np.searchsorted(ux, np.round(xs, 4))
    cells = ux.size * uy.size
    if cells != xs.size and (np.unique(ii * ux.size + jj).size != xs.size or xs.size < 0.5 * cells):
        raise ValueError(f"{name}: the {xs.size} points do not form a {uy.size} x {ux.size} grid "
                         f"(point lists cannot be mapped)")
    cube = np.full((uy.size, ux.size, I.shape[1]), np.nan, np.float32)
    cube[ii, jj] = I
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
        missing = int((~np.isfinite(cube).all(-1)).sum())
        return RamanMap(cube=np.ascontiguousarray(cube, np.float32), wn=np.asarray(wn, float),
                        x=np.asarray(x, float), y=np.asarray(y, float),
                        name=os.path.splitext(name)[0], source=path,
                        meta={"format": "txt", "layout": layout, "missing": missing})

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
    if ext == ".l6pf":
        return read_l6pf(path)[0]
    if ext in TEXT_EXT:
        return read_text(path)
    raise ValueError(f"unsupported file type {ext}: open a .l6m map, a ParticleFinder .l6pf "
                     f"or a text export ({', '.join(TEXT_EXT)})")


def read_any(path: str) -> tuple[RamanMap, VideoImage | None]:
    """A map with its optical image: the .l6v next to a grid map, or the
    mosaic of a ParticleFinder project. A ParticleFinder spectra list (.l6m)
    takes positions, morphology and mosaic from the .l6pf saved beside it."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".l6pf":
        return read_l6pf(path)
    try:
        m = read_map(path)
    except PointListError as e:
        pf = pf_for_spectra_list(path, e.spectra.shape[0])
        if pf is None:
            raise
        pm, video = read_l6pf(pf)
        wn, cube = e.wn, np.ascontiguousarray(e.spectra[None].astype(np.float32))
        if wn[0] > wn[-1]:
            wn, cube = wn[::-1].copy(), np.ascontiguousarray(cube[..., ::-1])
        m = RamanMap(cube=cube, wn=wn, x=pm.x, y=pm.y,
                     name=os.path.splitext(os.path.basename(path))[0], source=path,
                     meta={**pm.meta, "format": "l6m", "project": pf, "settings": e.settings or pm.meta.get("settings")})
        return m, video
    video = None
    vp = find_video(path) if ext == ".l6m" else None
    if vp:
        try:
            video = read_l6v(vp)
        except Exception:
            video = None
    return m, video


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
        # Blocks can open with NUL-terminated labels: "1/cm x counts\0Peak
        # area\0" (S Carbon result images) or "Max height\0" (red maps). The values start
        # after the last label, off the 4-byte grid when the labels are odd.
        pos = 0
        for _ in range(3):
            nul = raw.find(b"\x00", o + pos, o + pos + 48) - o
            if nul - pos < 3 or not all(32 <= c < 127 for c in raw[o + pos:o + nul]):
                break
            pos = nul + 1
        if (n - pos) // 4 >= need:
            a = np.frombuffer(raw, "<f4", count=need, offset=o + pos) if pos else _f32(raw, o, n)[-need:]
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
