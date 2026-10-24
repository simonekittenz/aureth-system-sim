from __future__ import annotations

import math
from dataclasses import dataclass, replace
import warnings

import numpy as np
import rebound

from .constants import YEAR_DAYS
from .model import System, stable_limit_km


def _orbit_state(host_mass_kg: float, body_mass_kg: float, **orbit) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity (km, km/day) of a body on a two-body orbit around a host, relative to the host."""
    tmp = rebound.Simulation()
    tmp.units = ("day", "km", "kg")
    tmp.add(m=host_mass_kg)
    tmp.add(m=body_mass_kg, **orbit)
    p = tmp.particles[1]
    return np.array([p.x, p.y, p.z]), np.array([p.vx, p.vy, p.vz])


def build_simulation(system: System, megno: bool = False) -> rebound.Simulation:
    """The star, the chosen planet with its moons, and the other planets, all in one simulation. Units: day, km, kg.

    Particle order is star, planet, moons, other planets. The planet and its moons are placed so that their shared centre
    of mass (not the planet alone) follows the planet's orbit around the star; otherwise the moons' pull would stretch the
    year by about 0.16%. Moon inclinations are measured from the orbital plane (the Laplace plane at these distances)."""
    pl = system.planet
    sim = rebound.Simulation()
    sim.units = ("day", "km", "kg")
    sim.integrator = "ias15"
    sim.add(m=pl.star_mass_kg)

    moon_rel = [_orbit_state(pl.mass_kg, mn.mass_kg, a=mn.a_km, e=mn.ecc, inc=math.radians(mn.inc_deg),
                             Omega=math.radians(mn.asc_node_deg), omega=math.radians(mn.arg_peri_deg),
                             M=math.radians(mn.mean_anomaly_deg)) for mn in system.moons]
    total = pl.mass_kg + sum(mn.mass_kg for mn in system.moons)
    r_c, v_c = _orbit_state(pl.star_mass_kg, total, a=pl.a_km, e=pl.ecc, M=math.radians(pl.mean_anomaly_deg))
    if system.moons:
        shift_r = sum(mn.mass_kg * r for mn, (r, _) in zip(system.moons, moon_rel)) / total
        shift_v = sum(mn.mass_kg * v for mn, (_, v) in zip(system.moons, moon_rel)) / total
    else:
        shift_r = shift_v = np.zeros(3)
    r_p, v_p = r_c - shift_r, v_c - shift_v
    sim.add(m=pl.mass_kg, x=r_p[0], y=r_p[1], z=r_p[2], vx=v_p[0], vy=v_p[1], vz=v_p[2])
    for mn, (r, v) in zip(system.moons, moon_rel):
        sim.add(m=mn.mass_kg, x=r_p[0] + r[0], y=r_p[1] + r[1], z=r_p[2] + r[2], vx=v_p[0] + v[0], vy=v_p[1] + v[1], vz=v_p[2] + v[2])
    for nb in system.neighbors:
        sim.add(m=nb.mass_kg, a=nb.a_km, e=nb.ecc, M=math.radians(nb.mean_anomaly_deg), primary=sim.particles[0])
    sim.move_to_com()
    if megno:
        sim.init_megno()
    return sim


@dataclass
class Track:
    t: np.ndarray                 # days
    moon_pos: np.ndarray          # (n, 2, 3) km, relative to planet
    star_pos: np.ndarray          # (n, 3) km, relative to planet
    a: np.ndarray                 # (n, 2) km, relative to planet
    e: np.ndarray                 # (n, 2)
    inc: np.ndarray               # (n, 2) degrees
    megno: float | None = None
    others: np.ndarray | None = None   # (n, k, 3) km, the other planets relative to this planet


def _lost_flags(system: System, moon_pos: np.ndarray, e: np.ndarray) -> np.ndarray:
    """(n, 2) booleans: moon escaped the Hill sphere, left a bound orbit, or hit the planet."""
    pl = system.planet
    dist = np.linalg.norm(moon_pos, axis=2)
    out = (dist > pl.hill_radius_km) | (e >= 1) | ~np.isfinite(e)
    for k, m in enumerate(system.moons):
        out[:, k] |= dist[:, k] < pl.radius_km + m.radius_km
    return out


def integrate(system: System, times_days: np.ndarray, megno: bool = False,
              stop_when_lost: bool = False, elements: bool = True) -> Track:
    sim = build_simulation(system, megno=megno)
    n = len(times_days)
    nm, nn = len(system.moons), len(system.neighbors)
    moon_pos = np.zeros((n, nm, 3))
    star_pos = np.zeros((n, 3))
    other_pos = np.zeros((n, nn, 3))
    a = np.zeros((n, nm))
    e = np.zeros((n, nm))
    inc = np.zeros((n, nm))
    last = n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, t in enumerate(times_days):
            sim.integrate(float(t))
            ps = sim.particles
            planet = ps[1]
            pp = np.array([planet.x, planet.y, planet.z])
            star_pos[i] = np.array([ps[0].x, ps[0].y, ps[0].z]) - pp
            for k in range(nm):
                m = ps[2 + k]
                moon_pos[i, k] = np.array([m.x, m.y, m.z]) - pp
                if elements:
                    o = m.orbit(primary=planet)
                    a[i, k], e[i, k], inc[i, k] = o.a, o.e, math.degrees(o.inc)
            for k in range(nn):
                q = ps[2 + nm + k]
                other_pos[i, k] = np.array([q.x, q.y, q.z]) - pp
            if stop_when_lost and nm and _lost_flags(system, moon_pos[i:i + 1], e[i:i + 1]).any():
                last = i + 1
                break
    meg = sim.megno() if megno and last == n else None
    sl = slice(0, last)
    return Track(np.asarray(times_days, float)[sl], moon_pos[sl], star_pos[sl], a[sl], e[sl], inc[sl], meg, other_pos[sl])


def mean_rates(system: System, days: float = 4000.0, step: float = 1.0) -> dict:
    """Mean sidereal and synodic periods (Earth days) and mean distance (km) of each moon, measured
    from a linear fit to the simulated longitudes. These include the Sun's perturbation, unlike Kepler's law."""
    t = np.arange(0.0, days, step)
    tr = integrate(system, t)
    star_lon = np.unwrap(np.arctan2(tr.star_pos[:, 1], tr.star_pos[:, 0]))
    out = {"sidereal": [], "synodic": [], "distance": []}
    for k in range(len(system.moons)):
        lon = np.unwrap(np.arctan2(tr.moon_pos[:, k, 1], tr.moon_pos[:, k, 0]))
        out["sidereal"].append(2 * math.pi / np.polyfit(t, lon, 1)[0])
        out["synodic"].append(2 * math.pi / np.polyfit(t, lon - star_lon, 1)[0])
        out["distance"].append(float(np.linalg.norm(tr.moon_pos[:, k], axis=1).mean()))
    return out


@dataclass
class StabilityResult:
    track: Track
    years: float
    survived: bool
    reason: str
    lost_year: float | None
    a_min: np.ndarray
    a_max: np.ndarray
    e_max: np.ndarray
    inc_max: np.ndarray
    min_moon_sep_km: float
    max_dist_frac_of_limit: np.ndarray


def stability_run(system: System, years: float, n_samples: int = 600, megno: bool = False) -> StabilityResult:
    times = np.linspace(0, years * YEAR_DAYS, n_samples + 1)[1:]
    pl = system.planet
    tr = integrate(system, times, megno=megno, stop_when_lost=True)
    dist = np.linalg.norm(tr.moon_pos, axis=2)
    sep = np.linalg.norm(tr.moon_pos[:, 0] - tr.moon_pos[:, 1], axis=1)
    limits = np.array([stable_limit_km(pl, m.ecc) for m in system.moons])
    flags = _lost_flags(system, tr.moon_pos, tr.e)
    survived, reason, lost_year = True, "both moons stayed bound for the whole run", None
    if flags.any():
        i, k = np.argwhere(flags)[0]
        m = system.moons[k]
        lost_year = float(tr.t[i] / YEAR_DAYS)
        hit = dist[i, k] < pl.radius_km + m.radius_km
        survived = False
        reason = f"{m.name} {'hit the planet' if hit else 'was ejected from the planet system'} at ~{lost_year:.0f} yr"
    elif sep.min() < sum(m.radius_km for m in system.moons):
        survived, reason = False, "the moons collided"
    return StabilityResult(
        tr, years, survived, reason, lost_year,
        np.nanmin(tr.a, axis=0), np.nanmax(tr.a, axis=0), np.nanmax(tr.e, axis=0), np.nanmax(tr.inc, axis=0),
        float(sep.min()), np.nanmax(dist, axis=0) / limits,
    )


def phase_ensemble(system: System, years: float, n: int = 8, seed: int = 0, progress=None) -> list[StabilityResult]:
    """Re-run with random starting phases (mean anomaly, node, periapsis) for both moons."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        moons = tuple(
            replace(m, mean_anomaly_deg=rng.uniform(0, 360), asc_node_deg=rng.uniform(0, 360),
                    arg_peri_deg=rng.uniform(0, 360))
            for m in system.moons
        )
        out.append(stability_run(System(system.planet, moons, system.neighbors), years, n_samples=200))
        if progress:
            progress((i + 1) / n)
    return out
