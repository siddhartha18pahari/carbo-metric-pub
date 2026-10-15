"""Find maps on disk.

Roots come from ``data_roots.txt`` next to ``run.py`` (one folder per line,
``#`` for comments). Each candidate is parsed once and the result (grid
shape, range, or why it is not a map) is cached in ``.cache/library.json``
keyed by path, size and mtime, so later scans are instant.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading

from .labspec import TEXT_EXT, find_video, read_map

APP_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CACHE = os.path.join(APP_DIR, ".cache", "library.json")
ROOTS_FILE = os.path.join(APP_DIR, "data_roots.txt")
SKIP_DIRS = {"autosave folder", "autosave data folder", "__pycache__", ".git", ".cache"}


def roots() -> list[str]:
    out = []
    if os.path.exists(ROOTS_FILE):
        for line in open(ROOTS_FILE, encoding="utf-8"):
            line = line.split("#", 1)[0].strip()
            if line:
                p = line if os.path.isabs(line) else os.path.join(APP_DIR, line)
                if os.path.isdir(p):
                    out.append(os.path.normpath(p))
    uploads = os.path.join(APP_DIR, "uploads")
    if os.path.isdir(uploads):
        out.append(uploads)
    return out


def item_id(path: str) -> str:
    return hashlib.sha1(os.path.normcase(os.path.abspath(path)).encode()).hexdigest()[:12]


class Library:
    def __init__(self):
        self._lock = threading.Lock()
        self._cache = {}
        if os.path.exists(CACHE):
            try:
                self._cache = json.load(open(CACHE, encoding="utf-8"))
            except Exception:
                self._cache = {}
        self.scanning = False

    def _save(self):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        tmp = CACHE + ".tmp"
        json.dump(self._cache, open(tmp, "w", encoding="utf-8"))
        os.replace(tmp, CACHE)

    def candidates(self) -> list[str]:
        """Every .l6m under the roots, plus text files people opened (uploads/):
        lab folders hold many .txt that are fit tables or notes, not maps."""
        out = []
        uploads = os.path.normcase(os.path.join(APP_DIR, "uploads"))
        for root in roots():
            texts = os.path.normcase(root) == uploads
            for dp, dns, fns in os.walk(root):
                dns[:] = [d for d in dns if d.lower() not in SKIP_DIRS]
                for f in fns:
                    p = os.path.join(dp, f)
                    if f.lower().endswith(".l6m") and os.path.getsize(p) > 50_000:
                        out.append(p)
                    elif texts and f.lower().endswith(TEXT_EXT):
                        out.append(p)
        return out

    def describe(self, path: str) -> dict:
        st = os.stat(path)
        key = os.path.normcase(os.path.abspath(path))
        stamp = [st.st_size, int(st.st_mtime)]
        with self._lock:
            hit = self._cache.get(key)
        if hit and hit.get("stamp") == stamp:
            return hit
        info = {"stamp": stamp}
        try:
            m = read_map(path)
            info.update(ok=True, nx=int(m.cube.shape[1]), ny=int(m.cube.shape[0]),
                        k=int(m.wn.size), wn=[float(m.wn[0]), float(m.wn[-1])],
                        step=float(abs(m.x[1] - m.x[0])) if m.x.size > 1 else 0.0,
                        video=find_video(path) is not None)
        except Exception as e:
            info.update(ok=False, why=str(e).split(": ", 1)[-1][:160])
        with self._lock:
            self._cache[key] = info
        return info

    def listing(self, scan: bool = False) -> list[dict]:
        items = []
        all_roots = roots()
        for p in self.candidates():
            st = os.stat(p)
            key = os.path.normcase(os.path.abspath(p))
            info = self.describe(p) if scan else self._cache.get(key, {})
            if info.get("stamp") != [st.st_size, int(st.st_mtime)]:
                info = {}
            root = next((r for r in all_roots if key.startswith(os.path.normcase(r))), "")
            items.append({
                "id": item_id(p), "path": p, "name": os.path.splitext(os.path.basename(p))[0],
                "folder": os.path.relpath(os.path.dirname(p), root) if root else os.path.dirname(p),
                "root": os.path.basename(root), "size": st.st_size, "mtime": st.st_mtime,
                **{k: v for k, v in info.items() if k != "stamp"},
            })
        if scan:
            with self._lock:
                self._save()
        items.sort(key=lambda d: -d["mtime"])
        return items
