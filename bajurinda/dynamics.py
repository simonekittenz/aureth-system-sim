from __future__ import annotations

import math
from dataclasses import dataclass, replace
import warnings

import numpy as np
import rebound

from .constants import YEAR_DAYS
from .model import System, stable_limit_km


def build_simulation(system: System, megno: bool = False) -> rebound.Simulation:
    """Star + planet + two moons. Units: day, km, kg. Moon inclinations are
    measured from the planet's orbital plane (the Laplace plane at these distances)."""
    pl = system.planet
    sim = rebound.Simulation()
    sim.units = ("day", "km", "kg")
    sim.integrator = "ias15"
    sim.add(m=pl.star_mass_kg)
    sim.add(m=pl.mass_kg, a=pl.a_km, e=pl.ecc, primary=sim.particles[0])
    for mn in system.moons:
        sim.add(
            m=mn.mass_kg, a=mn.a_km, e=mn.ecc,
            inc=math.radians(mn.inc_deg),
            Omega=math.radians(mn.asc_node_deg),
            omega=math.radians(mn.arg_peri_deg),
            M=math.radians(mn.mean_anomaly_deg),
            primary=sim.particles[1],
        )
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


def _lost_flags(system: System, moon_pos: np.ndarray, e: np.ndarray) -> np.ndarray:
    """(n, 2) booleans: moon escaped the Hill sphere, left a bound orbit, or hit the planet."""
    pl = system.planet
    dist = np.linalg.norm(moon_pos, axis=2)
    out = (dist > pl.hill_radius_km) | (e >= 1) | ~np.isfinite(e)
    for k, m in enumerate(system.moons):
        out[:, k] |= dist[:, k] < pl.radius_km + m.radius_km
    return out


def integrate(system: System, times_days: np.ndarray, megno: bool = False,
              stop_when_lost: bool = False) -> Track:
    sim = build_simulation(system, megno=megno)
    n = len(times_days)
    moon_pos = np.zeros((n, 2, 3))
    star_pos = np.zeros((n, 3))
    a = np.zeros((n, 2))
    e = np.zeros((n, 2))
    inc = np.zeros((n, 2))
    last = n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, t in enumerate(times_days):
            sim.integrate(float(t))
            ps = sim.particles
            planet = ps[1]
            pp = np.array([planet.x, planet.y, planet.z])
            star_pos[i] = np.array([ps[0].x, ps[0].y, ps[0].z]) - pp
            for k in range(2):
                m = ps[2 + k]
                moon_pos[i, k] = np.array([m.x, m.y, m.z]) - pp
                o = m.orbit(primary=planet)
                a[i, k], e[i, k], inc[i, k] = o.a, o.e, math.degrees(o.inc)
            if stop_when_lost and _lost_flags(system, moon_pos[i:i + 1], e[i:i + 1]).any():
                last = i + 1
                break
    meg = sim.megno() if megno and last == n else None
    sl = slice(0, last)
    return Track(np.asarray(times_days, float)[sl], moon_pos[sl], star_pos[sl], a[sl], e[sl], inc[sl], meg)


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
        out.append(stability_run(System(system.planet, moons), years, n_samples=200))
        if progress:
            progress((i + 1) / n)
    return out
