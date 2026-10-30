from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from .dynamics import Track
from .model import KM, Planet, System, moon_period_days, synodic_period_days


def spin_axis(planet: Planet) -> np.ndarray:
    eps = math.radians(planet.obliquity_deg)
    az = math.radians(planet.axis_azimuth_deg)
    return np.array([math.sin(eps) * math.cos(az), math.sin(eps) * math.sin(az), math.cos(eps)])


def start_direction(planet: Planet) -> np.ndarray:
    """The direction from the star to the planet on day 1 (the planet's orbit starts closest to the star along +x, then moves on by its starting position)."""
    M, e = math.radians(planet.mean_anomaly_deg), planet.ecc
    E = M + e * math.sin(M)
    for _ in range(30):
        E -= (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
    nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))
    return np.array([math.cos(nu), math.sin(nu), 0.0])


def surface_normal(planet: Planet, lat_deg: float, lon_deg: float, t_days: np.ndarray) -> np.ndarray:
    """Straight up at a point on the surface. Longitude 0 faces away from the star at t = 0, so the clock reads midnight there on day 1."""
    s = spin_axis(planet)
    u0 = start_direction(planet)
    e1 = u0 - s * (s @ u0)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s, e1)
    theta = 2 * np.pi * t_days / (planet.sidereal_rotation_hours / 24) + math.radians(lon_deg)
    lat = math.radians(lat_deg)
    return math.cos(lat) * (np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2) + math.sin(lat) * s


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def illuminated_fraction(moon_pos: np.ndarray, star_pos: np.ndarray) -> np.ndarray:
    to_sun = star_pos - moon_pos
    to_planet = -moon_pos
    cos_i = np.einsum("ij,ij->i", _unit(to_sun), _unit(to_planet))
    return (1 + cos_i) / 2


def elongation(moon_pos: np.ndarray, star_pos: np.ndarray) -> np.ndarray:
    """Unwrapped ecliptic angle of the moon east of the Sun (0 = new, pi = full)."""
    ang = np.arctan2(moon_pos[:, 1], moon_pos[:, 0]) - np.arctan2(star_pos[:, 1], star_pos[:, 0])
    return np.unwrap(ang)


@dataclass
class TideSeries:
    liwaya: np.ndarray
    nineya: np.ndarray
    sun: np.ndarray
    site_total: np.ndarray
    site_range: np.ndarray
    global_peak: np.ndarray
    global_trough: np.ndarray
    global_peak_moons_only: np.ndarray


def _tide_amplitudes(planet: Planet, masses: list[float], vecs: list[np.ndarray]):
    R = planet.radius_km * KM
    amps, units = [], []
    for m, v in zip(masses, vecs):
        d = np.linalg.norm(v, axis=1)
        amps.append(m / planet.mass_kg * (R / (d * KM)) ** 3 * R)
        units.append(v / d[:, None])
    return amps, units


def _tensor_extrema(amps, units):
    n = len(amps[0])
    T = np.zeros((n, 3, 3))
    for A, u in zip(amps, units):
        T += A[:, None, None] * (3 * u[:, :, None] * u[:, None, :] - np.eye(3)) / 2
    ev = np.linalg.eigvalsh(T)
    return ev[:, -1], ev[:, 0]


def tide_series(system: System, track: Track, lat_deg: float, lon_deg: float) -> TideSeries:
    pl = system.planet
    masses = [system.moons[0].mass_kg, system.moons[1].mass_kg, pl.star_mass_kg]
    vecs = [track.moon_pos[:, 0], track.moon_pos[:, 1], track.star_pos]
    amps, units = _tide_amplitudes(pl, masses, vecs)
    r_hat = surface_normal(pl, lat_deg, lon_deg, track.t)
    zetas = []
    for u, A in zip(units, amps):
        c = np.einsum("ij,ij->i", u, r_hat)
        zetas.append(A * (3 * c**2 - 1) / 2)
    total = sum(zetas)
    dt = float(track.t[1] - track.t[0]) if len(track.t) > 1 else 1.0
    width = max(3, int(round(pl.sidereal_rotation_hours / 24 / dt)))
    rng = maximum_filter1d(total, width, mode="nearest") - minimum_filter1d(total, width, mode="nearest")
    peak, trough = _tensor_extrema(amps, units)
    peak_m, _ = _tensor_extrema(amps[:2], units[:2])
    return TideSeries(zetas[0], zetas[1], zetas[2], total, rng, peak, trough, peak_m)


def altitude_deg(planet: Planet, track: Track, body_vec: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    r_hat = surface_normal(planet, lat_deg, lon_deg, track.t)
    return np.degrees(np.arcsin(np.clip(np.einsum("ij,ij->i", r_hat, _unit(body_vec)), -1, 1)))


def crossings(t: np.ndarray, y: np.ndarray, level: float = 0.0):
    """Linearly-interpolated times where y crosses `level`; returns (rising, falling)."""
    s = y - level
    idx = np.where(np.sign(s[:-1]) * np.sign(s[1:]) < 0)[0]
    times = t[idx] - s[idx] * (t[idx + 1] - t[idx]) / (s[idx + 1] - s[idx])
    rising = s[idx + 1] > s[idx]
    return times[rising], times[~rising]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def angle_crossings(t: np.ndarray, angle: np.ndarray, target: float) -> np.ndarray:
    """Times at which a continuous (unwrapped) angle passes target + 2*pi*n."""
    x = (angle - target) / (2 * np.pi)
    f = np.floor(x)
    idx = np.where(f[:-1] != f[1:])[0]
    n = np.maximum(f[idx], f[idx + 1])
    frac = (n - x[idx]) / (x[idx + 1] - x[idx])
    return t[idx] + frac * (t[idx + 1] - t[idx])


def alignment_events(track: Track, phase_window_deg: float = 15.0) -> list[dict]:
    """Notable alignments in the sampled window, sorted by time."""
    t = track.t
    el = [elongation(track.moon_pos[:, k], track.star_pos) for k in range(2)]
    events: list[dict] = []

    def add(times, label):
        events.extend({"day": float(x), "event": label} for x in times)

    for k, name in enumerate(("Liwaya", "Nineya")):
        add(angle_crossings(t, el[k], 0.0), f"{name} new")
        add(angle_crossings(t, el[k], np.pi), f"{name} full")
    diff = el[0] - el[1]
    add(angle_crossings(t, diff, 0.0), "Liwaya-Nineya conjunction")
    add(angle_crossings(t, diff, np.pi), "Liwaya-Nineya opposition")

    w = math.radians(phase_window_deg)
    for label, centre in (("Double full", np.pi), ("Double new", 0.0)):
        near = (np.abs(_wrap(el[0] - centre)) < w) & (np.abs(_wrap(el[1] - centre)) < w)
        edges = np.diff(near.astype(int))
        starts, ends = np.where(edges == 1)[0] + 1, np.where(edges == -1)[0] + 1
        if near[0]:
            starts = np.insert(starts, 0, 0)
        for s0 in starts:
            e0 = ends[ends > s0]
            e0 = e0[0] if len(e0) else len(t) - 1
            events.append({"day": float(t[(s0 + e0) // 2]), "event": f"{label} (within {phase_window_deg:.0f} deg)"})
    return sorted(events, key=lambda e: e["day"])


def pattern_periods(system: System) -> dict:
    pl = system.planet
    year = pl.year_days
    Pl, Pn = (moon_period_days(pl, m) for m in system.moons)
    Sl, Sn = synodic_period_days(Pl, year), synodic_period_days(Pn, year)
    return {
        "liwaya_sidereal": Pl,
        "nineya_sidereal": Pn,
        "liwaya_synodic": Sl,
        "nineya_synodic": Sn,
        "moon_moon_conjunction": synodic_period_days(Pl, Pn),
        "phase_beat": synodic_period_days(Sl, Sn),
        "liwaya_spring_neap": Sl / 2,
        "nineya_spring_neap": Sn / 2,
        "moon_moon_tidal_alignment": synodic_period_days(Pl, Pn) / 2,
    }


def phase_disc(k: float, waxing: bool, radius: float, cx: float, cy: float, n: int = 40):
    """Polygon (x, y) for the lit part of a lunar disc."""
    phi = np.linspace(-np.pi / 2, np.pi / 2, n)
    sgn = 1 if waxing else -1
    limb_x = cx + sgn * radius * np.cos(phi)
    limb_y = cy + radius * np.sin(phi)
    term_x = cx + sgn * radius * np.cos(phi[::-1]) * (1 - 2 * k)
    term_y = cy + radius * np.sin(phi[::-1])
    return np.concatenate([limb_x, term_x]), np.concatenate([limb_y, term_y])
