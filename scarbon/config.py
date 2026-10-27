"""Processing + peak-fitting configuration, laid out like the S Carbon panel.

The default is a typical five-band carbon configuration as set up in
LabSpec 6.8 S Carbon: baseline poly order 2 fitted together
with five carbon bands over 1000-1800 cm-1, ratio D3/G.

Configurations are saved as INI files. The layout is this app's own; LabSpec's
.ini files are not documented and are not read.
"""

from __future__ import annotations

import configparser
import io
from dataclasses import asdict, dataclass, field

SHAPES = ("Lorentz", "Gauss")


@dataclass
class Peak:
    name: str
    pos: float
    pos_min: float
    pos_max: float
    width: float          # FWHM, cm-1
    width_min: float
    width_max: float
    shape: str = "Lorentz"
    fit: bool = True      # False: position and width held at pos/width

    def check(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"peak {self.name}: shape must be one of {SHAPES}")
        if not (self.pos_min <= self.pos <= self.pos_max):
            raise ValueError(f"peak {self.name}: position {self.pos} outside [{self.pos_min}, {self.pos_max}]")
        if not (0 < self.width_min <= self.width <= self.width_max):
            raise ValueError(f"peak {self.name}: width {self.width} outside [{self.width_min}, {self.width_max}]")


@dataclass
class Range:
    lo: float
    hi: float
    use_lo: bool = False
    use_hi: bool = False

    def admits(self, v):
        import numpy as np
        ok = np.ones_like(v, dtype=bool)
        if self.use_lo:
            ok &= v >= self.lo
        if self.use_hi:
            ok &= v <= self.hi
        return ok


@dataclass
class Config:
    baseline_order: int = 2          # -1 = no baseline
    fit_separately: bool = False     # True: baseline first (modpoly), then peaks
    range_lo: float = 1000.0
    range_hi: float = 1800.0
    intensity: Range = field(default_factory=lambda: Range(0, 5000, False, True))
    ratio_i: Range = field(default_factory=lambda: Range(0, 10000, False, False))
    ratio_a: Range = field(default_factory=lambda: Range(0, 10000, False, False))
    peaks: list[Peak] = field(default_factory=lambda: [
        Peak("G", 1580, 1570, 1590, 20, 10, 35, "Lorentz"),
        Peak("D1", 1350, 1340, 1360, 45, 15, 90, "Lorentz"),
        Peak("D2", 1620, 1610, 1630, 25, 10, 60, "Lorentz"),
        Peak("D3", 1500, 1490, 1510, 100, 40, 200, "Gauss"),
        Peak("D4", 1200, 1190, 1210, 50, 20, 120, "Lorentz"),
    ])
    skip_fitting: bool = False       # True: shapes held at their starting values
    auto_shapes: bool = True         # shapes from the mean spectrum; hold those one pixel cannot determine
    smooth_window: int = 0           # Savitzky-Golay along Raman shift before the fit; 0 = off, else odd
    smooth_order: int = 2
    spatial_kernel: int = 0          # Gaussian average of neighbouring spectra: 0 = off, 3, 5 or 7
    ratio_num: str = "D3"
    ratio_den: str = "G"
    max_iter: int = 60

    def check(self) -> "Config":
        if self.range_hi <= self.range_lo:
            raise ValueError("spectral range: high must exceed low")
        if not -1 <= self.baseline_order <= 6:
            raise ValueError("baseline order must be between -1 (none) and 6")
        if self.smooth_window and (self.smooth_window < 3 or self.smooth_window % 2 == 0
                                   or not 0 <= self.smooth_order < self.smooth_window):
            raise ValueError("smoothing window must be odd, at least 3, and larger than the order")
        if self.spatial_kernel not in (0, 3, 5, 7):
            raise ValueError("spatial smoothing must be off, 3, 5 or 7 pixels")
        names = [p.name for p in self.peaks]
        if len(set(names)) != len(names):
            raise ValueError("peak names must be unique")
        for p in self.peaks:
            p.check()
        for r in (self.ratio_num, self.ratio_den):
            if r not in names:
                raise ValueError(f"ratio peak {r!r} is not in the peak table")
        return self

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d)
        for k in ("intensity", "ratio_i", "ratio_a"):
            if k in d and isinstance(d[k], dict):
                d[k] = Range(**d[k])
        if "peaks" in d:
            d["peaks"] = [p if isinstance(p, Peak) else Peak(**p) for p in d["peaks"]]
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known}).check()

    def to_ini(self) -> str:
        cp = configparser.ConfigParser()
        cp["processing"] = {
            "baseline_order": str(self.baseline_order),
            "fit_separately": str(self.fit_separately),
            "range_lo": str(self.range_lo), "range_hi": str(self.range_hi),
            "skip_fitting": str(self.skip_fitting), "auto_shapes": str(self.auto_shapes),
            "smooth_window": str(self.smooth_window), "smooth_order": str(self.smooth_order),
            "spatial_kernel": str(self.spatial_kernel), "max_iter": str(self.max_iter),
            "ratio": f"{self.ratio_num}/{self.ratio_den}",
        }
        for key in ("intensity", "ratio_i", "ratio_a"):
            r: Range = getattr(self, key)
            cp[f"constraint.{key}"] = {"lo": str(r.lo), "hi": str(r.hi),
                                       "use_lo": str(r.use_lo), "use_hi": str(r.use_hi)}
        for i, p in enumerate(self.peaks):
            cp[f"peak.{i + 1}"] = {k: str(v) for k, v in asdict(p).items()}
        buf = io.StringIO()
        buf.write("; Carbo Metrics fitting configuration\n")
        cp.write(buf)
        return buf.getvalue()

    @classmethod
    def from_ini(cls, text: str) -> "Config":
        cp = configparser.ConfigParser()
        cp.read_string(text)
        pr = cp["processing"]
        num, den = pr.get("ratio", "D3/G").split("/")

        def rng(key, default: Range) -> Range:
            if f"constraint.{key}" not in cp:
                return default
            s = cp[f"constraint.{key}"]
            return Range(s.getfloat("lo"), s.getfloat("hi"), s.getboolean("use_lo"), s.getboolean("use_hi"))

        base = cls()
        peaks = []
        for sec in sorted((s for s in cp.sections() if s.startswith("peak.")),
                          key=lambda s: int(s.split(".")[1])):
            s = cp[sec]
            peaks.append(Peak(s["name"], s.getfloat("pos"), s.getfloat("pos_min"), s.getfloat("pos_max"),
                              s.getfloat("width"), s.getfloat("width_min"), s.getfloat("width_max"),
                              s.get("shape", "Lorentz"), s.getboolean("fit", True)))
        return cls(
            baseline_order=pr.getint("baseline_order", 2),
            fit_separately=pr.getboolean("fit_separately", False),
            range_lo=pr.getfloat("range_lo", 1000.0), range_hi=pr.getfloat("range_hi", 1800.0),
            intensity=rng("intensity", base.intensity), ratio_i=rng("ratio_i", base.ratio_i),
            ratio_a=rng("ratio_a", base.ratio_a), peaks=peaks or base.peaks,
            skip_fitting=pr.getboolean("skip_fitting", False),
            auto_shapes=pr.getboolean("auto_shapes", True),
            smooth_window=pr.getint("smooth_window", 0), smooth_order=pr.getint("smooth_order", 2),
            spatial_kernel=pr.getint("spatial_kernel", 0), max_iter=pr.getint("max_iter", 60),
            ratio_num=num.strip(), ratio_den=den.strip(),
        ).check()
