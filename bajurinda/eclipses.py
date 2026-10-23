from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .constants import R_SUN_KM
from .dynamics import Track
from .model import System
from .sky import angle_crossings, elongation

YEAR = 374


def eclipse_events(system: System, track: Track, star_radius_rsun: float | None = None) -> list[dict]:
    """Solar eclipses (at new moon) and lunar eclipses (at full moon) seen from anywhere on the planet.

    Geometry is evaluated at the moment of new or full moon, with the planet's centre as the reference.
    Solar: the shadow axis reaches the planet if the Sun-moon offset is below asin(R_planet / d). A central
    eclipse is 'total' if the moon looks bigger than the Sun even at the edge of the visible disc, 'hybrid' if
    it does only near the sub-moon point, otherwise 'annular'. Lunar: the moon is in the umbra, whose radius
    is asin(R_planet / d) minus the Sun's angular radius. Penumbral lunar eclipses are not counted.
    """
    pl = system.planet
    D = pl.local_day_days
    rs_km = (star_radius_rsun if star_radius_rsun is not None else pl.star_radius_rsun) * R_SUN_KM
    t = track.t
    d_star = np.linalg.norm(track.star_pos, axis=1)
    lat_star = np.arcsin(track.star_pos[:, 2] / d_star)
    out: list[dict] = []
    for k, moon in enumerate(system.moons):
        d = np.linalg.norm(track.moon_pos[:, k], axis=1)
        lat = np.arcsin(track.moon_pos[:, k, 2] / d)
        el = elongation(track.moon_pos[:, k], track.star_pos)
        for kind, target in (("solar", 0.0), ("lunar", math.pi)):
            tc = angle_crossings(t, el, target)
            dm, ds = np.interp(tc, t, d), np.interp(tc, t, d_star)
            sun_lat = np.interp(tc, t, lat_star)
            sep = np.abs(np.interp(tc, t, lat) - (sun_lat if kind == "solar" else -sun_lat))
            sun_r = np.arctan(rs_km / ds)
            moon_r = np.arctan(moon.radius_km / dm)
            reach = np.arcsin(pl.radius_km / dm)
            if kind == "solar":
                near = np.arctan(moon.radius_km / (dm - pl.radius_km))
                for i in range(len(tc)):
                    if sep[i] < reach[i]:
                        typ = "total" if moon_r[i] >= sun_r[i] else ("hybrid" if near[i] >= sun_r[i] else "annular")
                    elif sep[i] < reach[i] + sun_r[i] + moon_r[i]:
                        typ = "partial"
                    else:
                        typ = "none"
                    out.append(_row(moon.name, kind, typ, tc[i], D, sep[i], sun_r[i], moon_r[i], near[i]))
            else:
                umbra = reach - sun_r
                for i in range(len(tc)):
                    if sep[i] <= umbra[i] - moon_r[i]:
                        typ = "total"
                    elif sep[i] < umbra[i] + moon_r[i]:
                        typ = "partial"
                    else:
                        typ = "none"
                    out.append(_row(moon.name, kind, typ, tc[i], D, sep[i], sun_r[i], moon_r[i], moon_r[i]))
    out.sort(key=lambda e: e["t"])
    return out


def _row(moon, kind, typ, t, D, sep, sun_r, moon_r, near):
    return {"moon": moon, "kind": kind, "type": typ, "t": float(t), "day": float(t / D), "year": float(t / D / YEAR),
            "offset_deg": math.degrees(sep), "sun_deg": 2 * math.degrees(sun_r),
            "moon_deg": 2 * math.degrees(moon_r), "moon_near_deg": 2 * math.degrees(near)}


def summarise(events: list[dict], moon: str, kind: str, years: float) -> dict:
    sel = [e for e in events if e["moon"] == moon and e["kind"] == kind]
    n = len(sel)
    c = Counter(e["type"] for e in sel)
    central = c["total"] + c["hybrid"] + c["annular"]
    return {"events": n, "counts": c, "per_year": {k: c[k] / years for k in c},
            "any": n - c["none"], "central": central,
            "share_total_of_central": (c["total"] + c["hybrid"]) / central if central else 0.0,
            "share_any": (n - c["none"]) / n if n else 0.0}


# ------------------------------------------------------------------ what a single place on the planet sees

SITE_NONE, SITE_PARTIAL, SITE_ANNULAR, SITE_TOTAL = 0, 1, 2, 3


def site_eclipse_types(system: System, track: Track, lats, lons, star_radius_rsun: float | None = None,
                       window_hours: float = 14.0, step_minutes: float = 15.0):
    """Best solar-eclipse type seen at each (lat, lon) for every new moon.

    Returns (events, types): events is a list of (moon_index, time in Earth days) and types is an
    int array (events, points) holding 0 none, 1 partial, 2 annular, 3 total. The observer's own
    position matters (parallax is large for Nineya), and the Sun must be above the horizon.
    """
    from scipy.interpolate import CubicSpline

    pl = system.planet
    lats, lons = np.atleast_1d(np.asarray(lats, float)), np.atleast_1d(np.asarray(lons, float))
    R = pl.radius_km
    rs_km = (star_radius_rsun if star_radius_rsun is not None else pl.star_radius_rsun) * R_SUN_KM
    spin = np.array([math.sin(math.radians(pl.obliquity_deg)), 0.0, math.cos(math.radians(pl.obliquity_deg))])
    e1 = np.array([1.0, 0.0, 0.0]) - spin * spin[0]
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(spin, e1)
    p_sid = pl.sidereal_rotation_hours / 24
    la, lo = np.radians(lats), np.radians(lons)
    star_spline = CubicSpline(track.t, track.star_pos, axis=0)
    offsets = np.arange(-window_hours, window_hours + 1e-9, step_minutes / 60) / 24
    events, rows = [], []
    for k, moon in enumerate(system.moons):
        el = elongation(track.moon_pos[:, k], track.star_pos)
        moon_spline = CubicSpline(track.t, track.moon_pos[:, k], axis=0)
        for tc in angle_crossings(track.t, el, 0.0):
            tt = tc + offsets
            if tt[0] < track.t[0] or tt[-1] > track.t[-1]:
                continue
            theta = 2 * np.pi * tt[:, None] / p_sid + lo[None, :]
            n = (np.cos(la)[None, :, None] * (np.cos(theta)[..., None] * e1 + np.sin(theta)[..., None] * e2)
                 + np.sin(la)[None, :, None] * spin)
            obs = R * n
            sv = star_spline(tt)[:, None, :] - obs
            mv = moon_spline(tt)[:, None, :] - obs
            sd, md = np.linalg.norm(sv, axis=2), np.linalg.norm(mv, axis=2)
            cos_sep = np.einsum("tpc,tpc->tp", sv, mv) / (sd * md)
            sep = np.arccos(np.clip(cos_sep, -1, 1))
            sun_r, moon_r = np.arctan(rs_km / sd), np.arctan(moon.radius_km / md)
            up = np.einsum("tpc,tpc->tp", n, sv) / sd > 0
            partial = up & (sep < sun_r + moon_r)
            total = up & (sep + sun_r <= moon_r)
            annular = up & (sep + moon_r <= sun_r)
            best = np.where(total.any(0), SITE_TOTAL, np.where(annular.any(0), SITE_ANNULAR,
                                                               np.where(partial.any(0), SITE_PARTIAL, SITE_NONE)))
            events.append((k, float(tc)))
            rows.append(best)
    return events, np.array(rows)


def site_rates(events, types, moon_index: int, years: float) -> dict:
    sel = np.array([e[0] == moon_index for e in events])
    t = types[sel]
    return {"any": (t > 0).sum(0) / years, "total": (t == SITE_TOTAL).sum(0) / years,
            "annular": (t == SITE_ANNULAR).sum(0) / years,
            "partial": (t == SITE_PARTIAL).sum(0) / years}
