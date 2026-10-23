from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .dynamics import integrate
from .lunar_calendar import CalendarData, build_calendar
from .model import System
from .sky import angle_crossings

YEAR = 374


@dataclass
class ErasData:
    years: float
    plane_years: np.ndarray          # coarse time axis (Bajurindan years)
    mutual_deg: np.ndarray           # angle between the two orbital planes
    overlap_years: np.ndarray        # time of each overlap
    overlap_sep_deg: np.ndarray      # closest approach of the two moon centres at that pass
    passes: int
    calendar: CalendarData


def eras_data(system: System, years: int = 140, step: float = 0.25) -> ErasData:
    pl = system.planet
    D = pl.local_day_days
    t = np.arange(0.0, years * YEAR * D + 120, step)
    tr = integrate(system, t, elements=False)
    pos = tr.moon_pos

    lam = [np.unwrap(np.arctan2(pos[:, k, 1], pos[:, k, 0])) for k in range(2)]
    beta = [np.degrees(np.arcsin(pos[:, k, 2] / np.linalg.norm(pos[:, k], axis=1))) for k in range(2)]
    rad = [math.degrees(math.atan(m.radius_km / np.linalg.norm(pos[:, k], axis=1).mean()))
           for k, m in enumerate(system.moons)]
    tc = angle_crossings(t, lam[0] - lam[1], 0.0)
    tc = tc[tc < years * YEAR * D]
    db = np.abs(np.interp(tc, t, beta[0]) - np.interp(tc, t, beta[1]))
    hit = db < sum(rad)

    stride = int(round(2.0 / step))
    tc_s = t[::stride]
    h = []
    for k in range(2):
        p = pos[::stride, k]
        v = np.gradient(p, tc_s, axis=0)
        hk = np.cross(p, v)
        hk /= np.linalg.norm(hk, axis=1, keepdims=True)
        h.append(np.apply_along_axis(lambda c: np.convolve(c, np.ones(15) / 15, mode="same"), 0, hk))
    h = [x / np.linalg.norm(x, axis=1, keepdims=True) for x in h]
    mutual = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", h[0], h[1]), -1, 1)))
    keep = (tc_s >= 30) & (tc_s <= years * YEAR * D - 30)
    plane_years = tc_s / D / YEAR
    mutual = np.where(keep, mutual, np.nan)
    mutual[: 15] = mutual[15]                                    # edge of the smoothing window is not meaningful
    mutual[0] = 0.0 if system.moons[0].asc_node_deg == system.moons[1].asc_node_deg else mutual[0]

    cal = build_calendar(system, tr, "full", 0.5, span_days=years * YEAR * D)
    return ErasData(years, plane_years, mutual, tc[hit] / D / YEAR, db[hit], len(tc), cal)


def era_bands(data: ErasData, threshold_deg: float) -> list[tuple[float, float]]:
    """Stretches of time (start, end in years) when the two orbital planes are within `threshold_deg`."""
    below = np.nan_to_num(data.mutual_deg, nan=99.0) < threshold_deg
    bands, start = [], None
    for i, b in enumerate(below):
        if b and start is None:
            start = data.plane_years[i]
        if not b and start is not None:
            bands.append((float(start), float(data.plane_years[i])))
            start = None
    if start is not None:
        bands.append((float(start), float(data.plane_years[-1])))
    return bands


def era_summary(data: ErasData, bands: list[tuple[float, float]]) -> dict:
    inside = lambda y: any(a <= y <= b for a, b in bands)
    in_era = int(sum(inside(y) for y in data.overlap_years))
    total = len(data.overlap_years)
    mut = np.nan_to_num(data.mutual_deg, nan=99.0)
    centres = []
    for a, b in bands:
        sel = (data.plane_years >= a) & (data.plane_years <= b)
        if b - a > 1:
            centres.append(float(data.plane_years[sel][np.argmin(mut[sel])]))
    cal = data.calendar
    D = cal.day_len
    runs_in = 0
    for r in cal.runs:
        if inside(r["start_t"] / D / YEAR):
            runs_in += 1
    era_years = sum(b - a for a, b in bands)
    return {
        "overlaps": total, "per_year": total / data.years, "in_era": in_era,
        "share_in_era": in_era / total if total else 0.0,
        "era_share_of_time": era_years / data.years,
        "spacing": float(np.median(np.diff(centres))) if len(centres) > 1 else None,
        "n_eras": len(bands), "runs": len(cal.runs), "runs_in_era": runs_in,
        "touching": int((data.overlap_sep_deg < 0.05).sum()),
    }
