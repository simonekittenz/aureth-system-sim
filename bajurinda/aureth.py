from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from .constants import AU_KM, G, G_EARTH, M_EARTH, M_SUN
from .model import canon_planet, near_resonances

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "aureth.yaml"
K2 = 2.959122082855911e-4            # G in AU^3 / (solar mass day^2)
AU_PER_DAY_KMS = AU_KM / 86400.0     # 1 AU/day in km/s
EARTH_IN_SUN = M_EARTH / M_SUN
COLORS = {"Bayurinda": "#4cc9f0", "Ashara": "#ffb703", "Tarnic": "#e5383b"}


@dataclass(frozen=True)
class Star:
    name: str
    mass_msun: float
    lum_lsun: float
    radius_rsun: float


@dataclass(frozen=True)
class Body:
    name: str
    a_au: float
    ecc: float
    mass_earth: float
    radius_km: float | None
    day_hours: float | None
    phase_deg: float = 0.0           # mean anomaly at the founding moment
    peri_deg: float = 0.0
    node_deg: float = 0.0
    inc_deg: float = 0.0
    placeholder_note: str = ""

    @property
    def mass_msun(self) -> float:
        return self.mass_earth * EARTH_IN_SUN

    @property
    def escape_kms(self) -> float | None:
        if not self.radius_km:
            return None
        return math.sqrt(2 * G * self.mass_earth * M_EARTH / (self.radius_km * 1e3)) / 1e3


def load_system(path: Path = DATA_FILE) -> tuple[Star, list[Body]]:
    """Star and the planets that have a distance (Bayurinda, Ashara, Tarnic for now)."""
    d = yaml.safe_load(Path(path).read_text())
    pl = canon_planet()
    star = Star(d["star"]["name"], pl.star_mass_msun, d["star"]["luminosity_lsun"], d["star"]["radius_rsun"])
    start = d.get("system_start", {})
    phases = start.get("phase_deg", {})
    inc = float(start.get("inclination_deg", 0.0))
    bodies: list[Body] = []
    for p in d["planets"]:
        if p.get("distance_au") is None:
            continue
        name = p["name"]
        radius = p["diameter_km"] / 2 if p.get("diameter_km") else None
        if name == "Bayurinda":
            a, mass = pl.a_au, pl.mass_earth                  # exact values behind the 440-day year
        else:
            a = float(p["distance_au"])
            mass = p.get("mass_earth")
            if mass is None and radius and p.get("gravity_g"):
                mass = p["gravity_g"] * G_EARTH * (radius * 1e3) ** 2 / G / M_EARTH
        note = ""
        if name == "Ashara":
            note = "eccentricity and mass are placeholders"
        bodies.append(Body(name, a, float(p.get("eccentricity", 0.0)), float(mass), radius, p.get("day_hours"),
                           float(phases.get(name, 0.0)), 0.0, 0.0, inc, note))
    return star, bodies


def mu_of(star: Star, body: Body) -> float:
    return K2 * (star.mass_msun + body.mass_msun)


def period_days(star: Star, body: Body) -> float:
    return 2 * math.pi * math.sqrt(body.a_au**3 / mu_of(star, body))


def synodic_days(p1: float, p2: float) -> float:
    return 1 / abs(1 / p1 - 1 / p2)


def kepler_E(M: np.ndarray, e: float) -> np.ndarray:
    E = M + e * np.sin(M)
    for _ in range(40):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - dE
        if np.max(np.abs(dE)) < 1e-14:
            break
    return E


def _rotation(body: Body) -> np.ndarray:
    w, i, n = (math.radians(x) for x in (body.peri_deg, body.inc_deg, body.node_deg))
    cw, sw, ci, si, cn, sn = math.cos(w), math.sin(w), math.cos(i), math.sin(i), math.cos(n), math.sin(n)
    return np.array([[cn * cw - sn * sw * ci, -cn * sw - sn * cw * ci, sn * si],
                     [sn * cw + cn * sw * ci, -sn * sw + cn * cw * ci, -cn * si],
                     [sw * si, cw * si, ci]])


def state(star: Star, body: Body, t_days) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric position (AU) and velocity (AU/day) on the exact two-body orbit; t = 0 is the founding moment."""
    t = np.atleast_1d(np.asarray(t_days, float))
    mu = mu_of(star, body)
    n = math.sqrt(mu / body.a_au**3)
    M = math.radians(body.phase_deg) + n * t
    E = kepler_E(np.mod(M, 2 * math.pi), body.ecc)
    a, e = body.a_au, body.ecc
    b = a * math.sqrt(1 - e * e)
    x, y = a * (np.cos(E) - e), b * np.sin(E)
    d = 1 - e * np.cos(E)
    vx, vy = -a * n * np.sin(E) / d, b * n * np.cos(E) / d
    R = _rotation(body)
    pos = np.stack([x, y, np.zeros_like(x)], axis=1) @ R.T
    vel = np.stack([vx, vy, np.zeros_like(x)], axis=1) @ R.T
    return pos, vel


def all_positions(star: Star, bodies: list[Body], t_days) -> np.ndarray:
    return np.stack([state(star, b, t_days)[0] for b in bodies], axis=1)      # (T, nbodies, 3)


# ------------------------------------------------------------------ N-body

def build_nbody(star: Star, bodies: list[Body]):
    import rebound

    sim = rebound.Simulation()
    sim.units = ("day", "AU", "Msun")
    sim.add(m=star.mass_msun)
    for b in bodies:
        sim.add(m=b.mass_msun, a=b.a_au, e=b.ecc, inc=math.radians(b.inc_deg), Omega=math.radians(b.node_deg),
                omega=math.radians(b.peri_deg), M=math.radians(b.phase_deg), primary=sim.particles[0])
    sim.move_to_com()
    return sim


def nbody_track(star: Star, bodies: list[Body], times_days: np.ndarray, integrator: str = "whfast") -> dict:
    sim = build_nbody(star, bodies)
    sim.integrator = integrator
    if integrator == "whfast":
        sim.dt = 0.04 * min(period_days(star, b) for b in bodies)
    pos = np.zeros((len(times_days), len(bodies), 3))
    a = np.zeros((len(times_days), len(bodies)))
    e = np.zeros_like(a)
    for i, t in enumerate(times_days):
        sim.integrate(float(t))
        ps = sim.particles
        s = np.array([ps[0].x, ps[0].y, ps[0].z])
        for k in range(len(bodies)):
            p = ps[k + 1]
            pos[i, k] = np.array([p.x, p.y, p.z]) - s
            o = p.orbit(primary=ps[0])
            a[i, k], e[i, k] = o.a, o.e
    return {"t": np.asarray(times_days, float), "pos": pos, "a": a, "e": e}


# ------------------------------------------------------------------ cycles, sunlight, sky

def planet_table(star: Star, bodies: list[Body], day_ref_hours: float = 19.0) -> list[dict]:
    rows = []
    for b in bodies:
        P = period_days(star, b)
        peri, apo = b.a_au * (1 - b.ecc), b.a_au * (1 + b.ecc)
        rows.append({
            "name": b.name, "a_au": b.a_au, "ecc": b.ecc, "period_days": P,
            "period_years": P / 365.25, "period_bayurinda_days": P / (day_ref_hours / 24),
            "period_local_days": P / (b.day_hours / 24) if b.day_hours else None,
            "flux": star.lum_lsun / b.a_au**2,
            "flux_range": (star.lum_lsun / apo**2, star.lum_lsun / peri**2),
            "sun_deg": 2 * math.degrees(math.atan(star.radius_rsun * 6.957e5 / (b.a_au * AU_KM))),
            "mass_earth": b.mass_earth, "note": b.placeholder_note,
        })
    return rows


def pair_table(star: Star, bodies: list[Body]) -> list[dict]:
    rows = []
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            p1, p2 = period_days(star, bodies[i]), period_days(star, bodies[j])
            inner, outer = (p1, p2) if p1 < p2 else (p2, p1)
            ratio = outer / inner
            rows.append({"pair": f"{bodies[i].name} - {bodies[j].name}", "ratio": ratio,
                         "synodic_days": synodic_days(p1, p2),
                         "resonances": near_resonances(ratio, max_order=3, tol=0.03)[:2]})
    return rows


def _angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    c = np.einsum("ij,ij->i", v1, v2) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def sky_from(pos: np.ndarray, names: list[str], observer: str) -> dict:
    """Elongation from the star, distance and signed longitude offset of each other planet, seen from `observer`."""
    o = names.index(observer)
    sun = -pos[:, o]
    out = {}
    for k, n in enumerate(names):
        if k == o:
            continue
        v = pos[:, k] - pos[:, o]
        lon = np.unwrap(np.arctan2(v[:, 1], v[:, 0]) - np.arctan2(sun[:, 1], sun[:, 0]))
        out[n] = {"elongation_deg": _angle(v, sun), "distance_au": np.linalg.norm(v, axis=1), "lon_offset": lon, "vec": v}
    return out


def crossing_times(t: np.ndarray, angle: np.ndarray, target: float) -> np.ndarray:
    x = (angle - target) / (2 * np.pi)
    f = np.floor(x)
    idx = np.where(f[:-1] != f[1:])[0]
    n = np.maximum(f[idx], f[idx + 1])
    frac = (n - x[idx]) / (x[idx + 1] - x[idx])
    return t[idx] + frac * (t[idx + 1] - t[idx])


def conjunction_minima(t: np.ndarray, sep_deg: np.ndarray, limit_deg: float) -> list[tuple[float, float]]:
    """Local minima of an angular separation below `limit_deg`, refined with a parabola."""
    out = []
    for i in range(1, len(sep_deg) - 1):
        if sep_deg[i] <= sep_deg[i - 1] and sep_deg[i] < sep_deg[i + 1] and sep_deg[i] < limit_deg:
            y0, y1, y2 = sep_deg[i - 1] ** 2, sep_deg[i] ** 2, sep_deg[i + 1] ** 2
            den = y0 - 2 * y1 + y2
            shift = 0.5 * (y0 - y2) / den if den > 0 else 0.0
            val = y1 - (y2 - y0) ** 2 / (8 * den) if den > 0 else y1
            out.append((float(t[i] + shift * (t[1] - t[0])), float(math.sqrt(max(val, 0.0)))))
    return out


def grand_alignments(star: Star, bodies: list[Body], years: float, tol_deg: float = 10.0, step: float = 2.0) -> list[dict]:
    """Times when all planets lie within `tol_deg` of the same heliocentric direction."""
    t = np.arange(0, years * 365.25, step)
    pos = all_positions(star, bodies, t)
    lon = np.degrees(np.arctan2(pos[:, :, 1], pos[:, :, 0]))
    ref = lon[:, :1]
    off = (lon - ref + 180) % 360 - 180
    spread = off.max(axis=1) - off.min(axis=1)
    ok = spread < tol_deg
    events, start = [], None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = i
        if (not flag or i == len(ok) - 1) and start is not None:
            seg = slice(start, i if not flag else i + 1)
            best = start + int(np.argmin(spread[seg]))
            events.append({"t": float(t[best]), "spread_deg": float(spread[best]), "length_days": float(t[min(i, len(t) - 1)] - t[start])})
            start = None
    return events
