from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "aureth.yaml"


@dataclass(frozen=True)
class NamedStar:
    name: str
    lon: float
    lat: float
    mag: float
    color: str
    label: bool = True
    note: str = ""


@dataclass(frozen=True)
class MilkyWay:
    pole_lon: float = 215.0
    pole_lat: float = 28.0
    centre_lon: float = 258.0
    width: float = 7.0
    bulge_width: float = 9.0
    bulge_length: float = 32.0
    rift_from: float = 335.0
    rift_to: float = 45.0
    rift_strength: float = 0.7
    brightness: float = 1.0
    patches: tuple = ()          # (lon, lat, size_deg, strength)


def ecl_vec(lon_deg, lat_deg) -> np.ndarray:
    lo, la = np.radians(lon_deg), np.radians(lat_deg)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la) * np.ones_like(lo)], axis=-1)


@lru_cache(maxsize=1)
def load_sky(path: str = str(DATA_FILE)) -> tuple[tuple[NamedStar, ...], MilkyWay]:
    d = yaml.safe_load(Path(path).read_text()).get("sky", {})
    stars = tuple(NamedStar(s["name"], float(s["lon"]), float(s["lat"]), float(s["mag"]), s.get("color", "#ffffff"),
                            bool(s.get("label", True)), s.get("note", "")) for s in d.get("stars", []))
    m = d.get("milky_way", {})
    if not m:
        return stars, MilkyWay()
    rift = m.get("rift", {})
    patches = tuple((float(p["lon"]), float(p["lat"]), float(p["size_deg"]), float(p["strength"])) for p in m.get("patches", []))
    mw = MilkyWay(m["pole"]["lon"], m["pole"]["lat"], m.get("centre_lon", 258.0), m.get("width_deg", 7.0),
                  m.get("bulge_width_deg", 9.0), m.get("bulge_length_deg", 32.0), rift.get("from_deg", 335.0),
                  rift.get("to_deg", 45.0), rift.get("strength", 0.7), m.get("brightness", 1.0), patches)
    return stars, mw


def mw_basis(mw: MilkyWay) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vectors: toward the galaxy's centre (on the plane), the other in-plane axis, and the pole."""
    P = ecl_vec(np.array(mw.pole_lon), np.array(mw.pole_lat))
    c0 = ecl_vec(np.array(mw.centre_lon), np.array(0.0))
    C = c0 - np.dot(c0, P) * P
    C /= np.linalg.norm(C)
    return C, np.cross(P, C), P


def _bulge(l_deg: np.ndarray, mw: MilkyWay) -> np.ndarray:
    d = (l_deg + 180) % 360 - 180
    return np.exp(-0.5 * (d / mw.bulge_length) ** 2)


def half_width(l_deg: np.ndarray, mw: MilkyWay) -> np.ndarray:
    return mw.width + mw.bulge_width * _bulge(l_deg, mw)


def band_edges(mw: MilkyWay, scale: float, n: int = 72, l0: float = 0.0, l1: float = 360.0) -> tuple[np.ndarray, np.ndarray]:
    """Upper and lower edge of the band, `scale` times its natural half-width, from l0 to l1 degrees along the plane."""
    C, E2, P = mw_basis(mw)
    l = np.linspace(l0, l1, n)
    u = np.cos(np.radians(l))[:, None] * C + np.sin(np.radians(l))[:, None] * E2
    w = np.radians(half_width(l, mw) * scale)[:, None]
    return u * np.cos(w) + P * np.sin(w), u * np.cos(w) - P * np.sin(w)


def circle_on_sky(lon: float, lat: float, radius_deg: float, n: int = 20) -> np.ndarray:
    c = ecl_vec(np.array(lon), np.array(lat))
    t1 = np.cross(c, [0.0, 0.0, 1.0])
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.cross(c, [1.0, 0.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(c, t1)
    a = np.linspace(0, 2 * np.pi, n)
    r = math.radians(radius_deg)
    return c * math.cos(r) + (np.cos(a)[:, None] * t1 + np.sin(a)[:, None] * t2) * math.sin(r)


def faint_stars(mw: MilkyWay, n_band: int = 380, n_field: int = 160, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Faint stars: most crowded along the galaxy, a thin scatter elsewhere. Returns unit vectors and magnitudes."""
    rng = np.random.default_rng(seed)
    C, E2, P = mw_basis(mw)
    pts = []
    while len(pts) < n_band:
        l = rng.uniform(0, 360)
        if rng.random() > (0.35 + 0.65 * _bulge(np.array(l), mw)):
            continue
        b = rng.normal(0, 0.55 * half_width(np.array(l), mw))
        u = math.cos(math.radians(l)) * C + math.sin(math.radians(l)) * E2
        v = u * math.cos(math.radians(b)) + P * math.sin(math.radians(b))
        pts.append(v / np.linalg.norm(v))
    v = rng.normal(size=(n_field, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    allv = np.vstack([np.array(pts), v])
    mag = rng.uniform(3.2, 5.6, len(allv))
    return allv, mag


def load_faint_settings(path: str = str(DATA_FILE)) -> dict:
    """The seed and counts for the generated faint stars (cached, so call clear_sky_cache after saving)."""
    d = yaml.safe_load(Path(path).read_text()).get("sky", {}).get("faint_stars", {})
    return {"seed": int(d.get("seed", 5)), "n_band": int(d.get("count_along_galaxy", 380)),
            "n_field": int(d.get("count_elsewhere", 160))}


def save_faint_seed(seed: int, path: Path = DATA_FILE) -> None:
    """Write a new seed into the data file, leaving the rest of the file (and its comments) alone."""
    import re
    text = path.read_text()
    new, n = re.subn(r"(faint_stars:[^\n]*\n\s+seed:\s*)\d+", rf"\g<1>{int(seed)}", text, count=1)
    if n != 1:
        raise ValueError("could not find sky.faint_stars.seed in the data file")
    path.write_text(new)
    load_sky.cache_clear()
