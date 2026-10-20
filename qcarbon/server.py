"""Local web server: the browser UI talks to the fitting engine through this.

Long jobs (map fit, splat fit) run in worker threads and are polled.
Arrays travel as base64 little-endian float32.
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import threading
import time
import traceback

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from . import splat as splat_mod
from .config import Config
from .fitter import fit_spectra, smooth_spectra
from .labspec import PF_LABELS, TEXT_EXT, RamanMap, VideoImage, read_any
from .library import APP_DIR, Library, item_id
from .results import export_table, quality, ratios, statistics, valid_mask

WEB = os.path.join(APP_DIR, "web")
UPLOADS = os.path.join(APP_DIR, "uploads")

app = FastAPI(title="Carbo Metrics")
app.mount("/static", StaticFiles(directory=WEB), name="static")
library = Library()


@app.middleware("http")
async def no_stale_ui(request, call_next):
    # a local app that changes under you: always revalidate the UI files
    response = await call_next(request)
    if request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, "<f4").tobytes()).decode()


class Dataset:
    def __init__(self, path: str):
        self.path = path
        self.id = item_id(path)
        self.map, video = read_any(path)
        self.video: VideoImage | None = None
        self.video_png: bytes | None = None
        if video is not None:
            # a ParticleFinder mosaic can be 8000 px square: keep a copy at
            # most ~4000 px across (and let the file's bytes go)
            f = max(1, int(np.ceil(max(video.rgb.shape[:2]) / 4096)))
            self.video = VideoImage(rgb=np.ascontiguousarray(video.rgb[::f, ::f]),
                                    x=video.x[::f].copy(), y=video.y[::f].copy())
        self.fit = None
        self.fit_cfg: Config | None = None
        self.fit_job = {"state": "idle"}
        self.splat_job = {"state": "idle"}
        self.lock = threading.Lock()

    def info(self) -> dict:
        m = self.map
        d = {
            "id": self.id, "name": m.name, "path": self.path, "nx": int(m.cube.shape[1]),
            "ny": int(m.cube.shape[0]), "k": int(m.wn.size), "wn": b64(m.wn),
            "x": b64(m.x), "y": b64(m.y), "format": m.meta.get("format"),
            "counts": [float(np.nanmin(m.cube)), float(np.nanpercentile(m.cube, 99.9))],
            "missing": int(m.meta.get("missing", 0)),
            "laser_nm": m.meta.get("laser_nm"),
        }
        if self.video is not None:
            v = self.video
            d["video"] = {"w": int(v.rgb.shape[1]), "h": int(v.rgb.shape[0]),
                          "x": [float(v.x[0]), float(v.x[-1])], "y": [float(v.y[0]), float(v.y[-1])]}
        pts = m.meta.get("points")
        if pts is not None:
            d["kind"] = "particles"
            d["points"] = {k: b64(v) for k, v in pts.items()}
            d["point_labels"] = PF_LABELS
        return d


DATASETS: dict[str, Dataset] = {}


def get_ds(ds_id: str) -> Dataset:
    ds = DATASETS.get(ds_id)
    if ds is None:
        raise HTTPException(404, "dataset not open")
    return ds


def open_path(path: str) -> Dataset:
    ds_id = item_id(path)
    if ds_id not in DATASETS:
        try:
            DATASETS[ds_id] = Dataset(path)
        except Exception as e:
            raise HTTPException(422, str(e))
    return DATASETS[ds_id]


# --------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"), headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health():
    ok, why = splat_mod.available()
    return {"ok": True, "splat": ok, "splat_why": why}


@app.get("/api/config/default")
def config_default():
    return Config().to_dict()


@app.post("/api/config/ini")
async def config_ini(request: Request):
    try:
        cfg = Config.from_dict(await request.json())
    except Exception as e:
        raise HTTPException(422, str(e))
    return PlainTextResponse(cfg.to_ini())


@app.post("/api/config/parse")
async def config_parse(request: Request):
    body = await request.json()
    try:
        return Config.from_ini(body.get("text", "")).to_dict()
    except Exception as e:
        raise HTTPException(422, f"not a Carbo Metrics configuration: {e}")


@app.get("/api/library")
def library_list(scan: int = 0):
    items = library.listing(scan=bool(scan))
    return {"items": items, "open": list(DATASETS)}


@app.post("/api/open")
async def open_ds(request: Request):
    body = await request.json()
    path = body.get("path", "")
    if not os.path.isfile(path):
        raise HTTPException(404, "file not found")
    return open_path(path).info()


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = os.path.join(UPLOADS, stamp)
    os.makedirs(folder, exist_ok=True)
    main = None
    for f in files:
        name = os.path.basename(f.filename or "upload")
        dest = os.path.join(folder, name)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        if name.lower().endswith((".l6m",) + TEXT_EXT) and main is None:
            main = dest
    if main is None:
        main = next((os.path.join(folder, os.path.basename(f.filename or "")) for f in files
                     if (f.filename or "").lower().endswith(".l6pf")), None)
    if main is None:
        raise HTTPException(422, f"open a .l6m map (optionally with its .l6v video) or a text export "
                                 f"({', '.join(TEXT_EXT)})")
    return open_path(main).info()


@app.get("/api/ds/{ds_id}/cube")
def cube(ds_id: str):
    ds = get_ds(ds_id)
    return Response(np.ascontiguousarray(ds.map.cube, "<f4").tobytes(),
                    media_type="application/octet-stream")


@app.get("/api/ds/{ds_id}/video.png")
def video_png(ds_id: str):
    ds = get_ds(ds_id)
    if ds.video is None:
        raise HTTPException(404, "no video image saved with this map")
    big = ds.video.rgb.shape[0] * ds.video.rgb.shape[1] > 4_000_000
    if ds.video_png is None:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(ds.video.rgb).save(buf, format="JPEG" if big else "PNG", quality=88)
        ds.video_png = buf.getvalue()
    return Response(ds.video_png, media_type="image/jpeg" if big else "image/png")


# -- fitting ---------------------------------------------------------------
def _fit_payload(ds: Dataset) -> dict:
    r = ds.fit
    return {
        "names": r.names, "shapes": r.shapes, "m": len(r.names),
        "intensity": b64(r.intensity), "position": b64(r.position), "width": b64(r.width),
        "area": b64(r.area), "baseline": b64(r.baseline), "nb": int(r.baseline.shape[1]),
        "std_error": b64(r.std_error), "red_chi2": b64(r.red_chi2), "n_iter": b64(r.n_iter),
        "range": [r.range_lo, r.range_hi], "n_channels": r.n_channels,
        "seconds": round(r.seconds, 3), "config": ds.fit_cfg.to_dict(), "held": r.held or [],
    }


def _fit_worker(ds: Dataset, cfg: Config):
    job = ds.fit_job
    try:
        if ds.map.meta.get("kind") == "particles" and cfg.spatial_kernel:
            # neighbouring particle numbers are not neighbouring places
            cfg = Config.from_dict({**cfg.to_dict(), "spatial_kernel": 0})
        Y = smooth_spectra(ds.map.cube, cfg).reshape(-1, ds.map.wn.size)
        res = fit_spectra(Y, ds.map.wn, cfg, progress=lambda f: job.__setitem__("progress", f))
        with ds.lock:
            ds.fit, ds.fit_cfg = res, cfg
        job.update(state="done", progress=1.0, seconds=res.seconds)
    except Exception as e:
        job.update(state="error", error=str(e), trace=traceback.format_exc()[-1500:])


@app.post("/api/ds/{ds_id}/fit")
async def fit_start(ds_id: str, request: Request):
    ds = get_ds(ds_id)
    try:
        cfg = Config.from_dict(await request.json())
    except Exception as e:
        raise HTTPException(422, str(e))
    if ds.fit_job.get("state") == "running":
        raise HTTPException(409, "a fit is already running")
    ds.fit_job = {"state": "running", "progress": 0.0, "started": time.time()}
    threading.Thread(target=_fit_worker, args=(ds, cfg), daemon=True).start()
    return ds.fit_job


@app.get("/api/ds/{ds_id}/fit")
def fit_status(ds_id: str, result: int = 0):
    ds = get_ds(ds_id)
    out = dict(ds.fit_job)
    if result and ds.fit is not None:
        out["result"] = _fit_payload(ds)
    return JSONResponse(out)


def _with_view(ds: Dataset, body: dict) -> Config:
    """The stored fit, with the view settings (ratio, constraints) from the UI."""
    if ds.fit is None:
        raise HTTPException(409, "fit the map first")
    d = ds.fit_cfg.to_dict()
    for k in ("ratio_num", "ratio_den", "intensity", "ratio_i", "ratio_a"):
        if k in body:
            d[k] = body[k]
    try:
        return Config.from_dict(d)
    except Exception as e:
        raise HTTPException(422, str(e))


@app.post("/api/ds/{ds_id}/summary")
async def summary(ds_id: str, request: Request):
    ds = get_ds(ds_id)
    cfg = _with_view(ds, await request.json())
    v = valid_mask(ds.fit, cfg)
    return {"quality": quality(ds.fit, cfg, v), "statistics": statistics(ds.fit, cfg, v)}


@app.post("/api/ds/{ds_id}/export")
async def export(ds_id: str, request: Request):
    ds = get_ds(ds_id)
    body = await request.json()
    cfg = _with_view(ds, body)
    m = ds.map
    if body.get("format") == "full":
        pts = m.meta.get("points")
        if pts is not None:
            xx, yy = pts["x"], pts["y"]
        else:
            xx, yy = (a.ravel() for a in np.meshgrid(m.x, m.y))
        r = ds.fit
        v = valid_mask(r, cfg)
        rat = ratios(r, cfg)
        cols = [("X (um)", xx), ("Y (um)", yy), ("Valid", v.astype(float))]
        if pts is not None:
            # ASCII units, like "X (um)": Excel misreads a BOM-less UTF-8 "µ"
            ascii_unit = lambda u: u.replace("µm²", "um2").replace("µm³", "um3").replace("µm", "um")
            cols += [(f"{PF_LABELS[k][0]} ({ascii_unit(PF_LABELS[k][1])})" if PF_LABELS[k][1] else PF_LABELS[k][0], pts[k])
                     for k in PF_LABELS] + [("Class", pts["cls"])]
        for j, n in enumerate(r.names):
            cols += [(f"{n} Intensity", r.intensity[:, j]), (f"{n} Position", r.position[:, j]),
                     (f"{n} Width", r.width[:, j]), (f"{n} Area", r.area[:, j])]
        cols += [(f"Ratio Intensity {cfg.ratio_num}/{cfg.ratio_den}", rat["intensity"]),
                 (f"Ratio Width {cfg.ratio_num}/{cfg.ratio_den}", rat["width"]),
                 (f"Ratio Area {cfg.ratio_num}/{cfg.ratio_den}", rat["area"]),
                 ("Std error", r.std_error), ("Red chi2", r.red_chi2)]
        arr = np.stack([c for _, c in cols], 1)
        lines = [",".join(n for n, _ in cols)] + [",".join(f"{x:.6g}" for x in row) for row in arr]
        text, ext = "\n".join(lines) + "\n", "csv"
    else:
        text, ext = export_table(ds.fit, cfg), "txt"
    fname = f"{m.name} CarboMetrics {cfg.ratio_num}-{cfg.ratio_den}.{ext}"
    return PlainTextResponse(text, headers={"Content-Disposition": f'attachment; filename="{fname}"'})


# -- splatting --------------------------------------------------------------
def _splat_worker(ds: Dataset, n_gaussians: int, iters: int, g_name: str):
    job = ds.splat_job

    def progress(frac, hist, snap):
        job["progress"] = frac
        job["last"] = hist
        job.setdefault("history", []).append(hist)
        if snap is not None:
            job["snapshot"] = snap
            job["snap_iter"] = hist["iter"]

    try:
        out = splat_mod.run(ds.map, ds.fit, g_name=g_name, n_gaussians=n_gaussians,
                            iters=iters, progress=progress)
        job.update(state="done", progress=1.0, result=out)
    except Exception as e:
        job.update(state="error", error=str(e), trace=traceback.format_exc()[-1500:])


@app.post("/api/ds/{ds_id}/splat")
async def splat_start(ds_id: str, request: Request):
    ds = get_ds(ds_id)
    ok, why = splat_mod.available()
    if not ok:
        raise HTTPException(501, why)
    if ds.map.meta.get("kind") == "particles":
        raise HTTPException(422, "Splat 3D needs a gridded map; ParticleFinder spectra are one point per particle")
    if ds.fit is None:
        raise HTTPException(409, "fit the map first: the splat uses each pixel's fitted baseline")
    if ds.splat_job.get("state") == "running":
        raise HTTPException(409, "a splat fit is already running")
    body = await request.json()
    n = int(np.clip(body.get("n_gaussians", 1400), 100, 1900))
    iters = int(np.clip(body.get("iters", 150), 10, 600))
    g = body.get("g_name", "G")
    if g not in ds.fit.names:
        g = ds.fit.names[0]
    # the field is fitted on this fit's baselines; the UI flags it if the fit changes later
    ds.splat_job = {"state": "running", "progress": 0.0, "started": time.time(),
                    "n_gaussians": n, "iters": iters, "fit_config": ds.fit_cfg.to_dict()}
    threading.Thread(target=_splat_worker, args=(ds, n, iters, g), daemon=True).start()
    return {k: v for k, v in ds.splat_job.items()}


@app.get("/api/ds/{ds_id}/splat")
def splat_status(ds_id: str, since: int = -1):
    ds = get_ds(ds_id)
    job = ds.splat_job
    out = {k: v for k, v in job.items() if k not in ("history", "snapshot", "result")}
    out["history"] = job.get("history", [])[-400:]
    if job.get("snap_iter", -1) > since and "snapshot" in job:
        out["snapshot"] = job["snapshot"]
    if job.get("state") == "done":
        out["result"] = job["result"]
    return JSONResponse(out)


def warm_up():
    """Compile the fitting kernels once at start-up (numba caches them on disk)."""
    from .config import Config as C
    wn = np.linspace(1000, 1800, 120)
    y = 20 + 50 / (1 + ((wn - 1580) / 10) ** 2)
    fit_spectra(np.tile(y, (4, 1)), wn, C())
