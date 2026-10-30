from __future__ import annotations

import math

import numpy as np

from .dynamics import integrate
from .model import System
from .sky import spin_axis


def rise_offset_deg(dec_deg: float, lat_deg: float) -> float | None:
    """Where a body at declination `dec_deg` rises, in degrees north (+) or south (-) of due east, for an observer at
    latitude `lat_deg` and a flat horizon. None if it never rises or never sets there. Refraction, disc size and
    parallax are ignored."""
    x = math.sin(math.radians(dec_deg)) / math.cos(math.radians(lat_deg))
    if abs(x) >= 1:
        return None
    return math.degrees(math.asin(x))


def sun_markers(system: System, year_days_solar: int = 440, step: float = 0.25) -> dict:
    """Solstices and equinoxes of Bayurinda's year, counted in solar days from year 1, day 1.

    The direction the spin axis tilts (relative to perihelion) is a free choice, set with the rest of the day 1 start
    (`axis_azimuth_deg` in data/aureth.yaml)."""
    pl = system.planet
    D = pl.local_day_days
    t = np.arange(0, pl.year_days + 5, step)
    from .aureth import body_from_planet, star_of, state
    pos, _ = state(star_of(pl), body_from_planet(pl), t)
    sun = -pos / np.linalg.norm(pos, axis=1, keepdims=True)
    dec = np.degrees(np.arcsin(np.clip(sun @ spin_axis(pl), -1, 1)))
    n = int(pl.year_days / step)
    i_s, i_w = int(np.argmax(dec[:n])), int(np.argmin(dec[:n]))
    up = np.where((dec[:n - 1] < 0) & (dec[1:n] >= 0))[0]
    dn = np.where((dec[:n - 1] >= 0) & (dec[1:n] < 0))[0]
    day = lambda i: float(t[i] / D)
    out = {"max_dec": float(dec.max()), "min_dec": float(dec.min()),
           "north_summer_day": day(i_s), "north_winter_day": day(i_w),
           "spring_equinox_day": day(up[0]) if len(up) else None, "autumn_equinox_day": day(dn[0]) if len(dn) else None,
           "year_days_solar": pl.year_solar_days}
    return out


def moon_declination_envelope(system: System, years: int = 130, step: float = 0.5) -> dict:
    """For each moon, the biggest declination reached each month (north or south), and the standstills."""
    pl = system.planet
    Y = pl.year_days
    t = np.arange(0, years * Y, step)
    tr = integrate(system, t, elements=False)
    ax = spin_axis(pl)
    out = {}
    for k, moon in enumerate(system.moons):
        u = tr.moon_pos[:, k] / np.linalg.norm(tr.moon_pos[:, k], axis=1, keepdims=True)
        dec = np.degrees(np.arcsin(np.clip(u @ ax, -1, 1)))
        P = 1 / (1 / 28.03 if k == 0 else 1 / 7.51)
        w = max(int(P / step), 4)
        nwin = len(dec) // w
        hi = np.array([dec[i * w:(i + 1) * w].max() for i in range(nwin)])
        lo = np.array([dec[i * w:(i + 1) * w].min() for i in range(nwin)])
        tm = np.array([t[i * w + w // 2] for i in range(nwin)])
        amp = np.maximum(hi, -lo)
        sm = np.convolve(amp, np.ones(5) / 5, mode="same")
        mean = sm[5:-5].mean()
        maxs = [i for i in range(8, nwin - 8) if sm[i] > sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] > mean]
        mins = [i for i in range(8, nwin - 8) if sm[i] < sm[i - 1] and sm[i] <= sm[i + 1] and sm[i] < mean]
        thin = lambda ids: [i for j, i in enumerate(ids) if j == 0 or tm[i] - tm[ids[j - 1]] > 3 * Y]
        out[moon.name] = {
            "t": tm, "amp": amp,
            "major": [(float(tm[i] / Y), float(amp[max(i - 2, 0):i + 3].max())) for i in thin(maxs)],
            "minor": [(float(tm[i] / Y), float(amp[max(i - 2, 0):i + 3].min())) for i in thin(mins)],
            "range": (float(amp.min()), float(amp.max())),
        }
    return out


def return_cycles(measured: dict, day_len: float, year_len: float, moon: str, max_months: int = 12000) -> dict:
    """Whole numbers of months after which several things repeat at once.

    `measured` is the dict from cycles.moon_cycles. Profiles:
      calendar - same phase, hour of the day and date in the year
      eclipse  - same phase, node (so the same declination range) and distance
      sky      - all five within about a day
    """
    m = measured[moon]
    S, Dr, An = m["synodic"], m["draconic"], m["anomalistic"]
    near = lambda x, p: abs(x - round(x / p) * p)
    rows = {"calendar": [], "eclipse": [], "sky": []}
    for N in range(2, max_months):
        T = N * S
        e = {"node": near(T, Dr), "dist": near(T, An), "hour": near(T, day_len) * 24, "date": near(T, year_len)}
        rec = {"months": N, "years": T / year_len, "solar_days": T / day_len, **e}
        if e["hour"] <= 1.5 and e["date"] <= 1.0 and len(rows["calendar"]) < 4:
            rows["calendar"].append(rec)
        if e["node"] <= 0.25 and e["dist"] <= 0.25 and len(rows["eclipse"]) < 4:
            rows["eclipse"].append(rec)
        if e["node"] <= 1.0 and e["dist"] <= 1.0 and e["hour"] <= 1.5 and e["date"] <= 1.5 and len(rows["sky"]) < 3:
            rows["sky"].append(rec)
    return rows


def star_alignment_life(precession_years: float, obliquity_deg: float, tolerances_deg=(0.25, 0.5, 1.0)) -> dict:
    """How long a monument aligned to a fixed star stays aligned as the spin axis wobbles.
    The star's declination changes at up to (2*pi / P) * sin(obliquity) per year, fastest for stars on the 0h/12h meridian
    of right ascension and not at all for stars near 6h or 18h."""
    rate = math.degrees(2 * math.pi / precession_years) * math.sin(math.radians(obliquity_deg))      # deg per year, worst case
    return {"max_drift_deg_per_year": rate, "per_century": rate * 100,
            "years_to_drift": {tol: 2 * tol / rate for tol in tolerances_deg}}
