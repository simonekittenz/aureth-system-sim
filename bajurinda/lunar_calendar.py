from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .dynamics import Track
from .model import System
from .sky import angle_crossings, elongation


@dataclass
class CalendarData:
    day_len: float                     # Earth days per Bajurindan solar day
    year_len: int                      # whole Bajurindan days per year (rounded)
    events: dict                       # {(moon_index, "full"|"new"): array of Earth-day times}
    months: list[dict]
    weeks: list[dict]
    runs: list[dict]
    start_kind: str


def phase_events(track: Track, moon_index: int) -> dict[str, np.ndarray]:
    """Times (Earth days) of full and new moon, including one at the epoch if the run starts on a phase."""
    t = track.t
    el = elongation(track.moon_pos[:, moon_index], track.star_pos)
    rate = (el[5] - el[0]) / (t[5] - t[0])
    out = {}
    for kind, target in (("full", math.pi), ("new", 0.0)):
        times = angle_crossings(t, el, target)
        r = (el[0] - target + math.pi) % (2 * math.pi) - math.pi
        t_ev = -r / rate
        if abs(r) < 0.35 and -0.5 <= t_ev <= 0.5 and (len(times) == 0 or times[0] - t_ev > 1):
            times = np.insert(times, 0, max(t_ev, 0.0))
        out[kind] = times
    return out


def _day_number(t: float, day_len: float) -> int:
    return int(math.floor(t / day_len + 1e-9)) + 1


def _ymd(day: int, year_len: int) -> tuple[int, int]:
    return (day - 1) // year_len + 1, (day - 1) % year_len + 1


def build_calendar(system: System, track: Track, start_kind: str = "full",
                   tolerance_days: float = 0.5, span_days: float | None = None) -> CalendarData:
    """Months follow Liwaya's phase cycle, weeks follow Nineya's. A 'double' is a Liwaya phase event
    with a Nineya event of the same kind within `tolerance_days` (Bajurindan days)."""
    pl = system.planet
    D = pl.local_day_days
    year_days = int(round(pl.year_solar_days))
    ev = {(k, kind): phase_events(track, k)[kind] for k in range(2) for kind in ("full", "new")}
    end = span_days if span_days is not None else float(track.t[-1])
    tol = tolerance_days * D

    L_start = ev[(0, start_kind)]
    N_start = ev[(1, start_kind)]

    def nearest(kind: str, t: float) -> float:
        arr = ev[(1, kind)]
        return float(arr[np.argmin(np.abs(arr - t))] - t)

    cut_m = end - 0.5 * float(np.median(np.diff(L_start)))     # a cycle holds the starts before half a month from its end
    cut_w = end - 0.5 * float(np.median(np.diff(N_start)))
    months = []
    for i in range(len(L_start) - 1):
        a, b = L_start[i], L_start[i + 1]
        if a >= cut_m:
            break
        row = {"month": i + 1, "start_t": float(a), "start_day": _day_number(a, D),
               "length_days": _day_number(b, D) - _day_number(a, D), "exact_days": float((b - a) / D)}
        row["year"], row["day_of_year"] = _ymd(row["start_day"], year_days)
        for kind in ("full", "new"):
            inside = ev[(0, kind)][(ev[(0, kind)] >= a - 1e-6) & (ev[(0, kind)] < b - 1e-6)]
            if len(inside):
                t0 = float(inside[0])
                off = nearest(kind, t0) / D
                row[f"{kind}_day"] = _day_number(t0, D)
                row[f"{kind}_t"] = t0
                row[f"{kind}_offset"] = off
                row[f"double_{kind}"] = abs(off) <= tolerance_days
            else:
                row.update({f"{kind}_day": None, f"{kind}_t": None, f"{kind}_offset": None, f"double_{kind}": False})
        months.append(row)

    weeks = []
    month_starts = [m["start_day"] for m in months]
    for i in range(len(N_start) - 1):
        a, b = N_start[i], N_start[i + 1]
        if a >= cut_w:
            break
        sd = _day_number(a, D)
        mi = max([j for j, s in enumerate(month_starts) if s <= sd], default=None)
        row = {"week": i + 1, "start_t": float(a), "start_day": sd,
               "length_days": _day_number(b, D) - sd, "exact_days": float((b - a) / D),
               "month": months[mi]["month"] if mi is not None else None,
               "day_of_month": sd - month_starts[mi] + 1 if mi is not None else None}
        row["year"], row["day_of_year"] = _ymd(sd, year_days)
        weeks.append(row)

    runs = []
    for kind in ("full", "new"):
        flagged = [m for m in months if m[f"double_{kind}"]]
        group: list[dict] = []
        for m in flagged:
            if group and m["month"] != group[-1]["month"] + 1:
                runs.append(_run(kind, group))
                group = []
            group.append(m)
        if group:
            runs.append(_run(kind, group))
    runs.sort(key=lambda r: r["start_t"])
    return CalendarData(D, year_days, ev, months, weeks, runs, start_kind)


def _run(kind: str, group: list[dict]) -> dict:
    best = min(group, key=lambda m: abs(m[f"{kind}_offset"]))
    return {
        "kind": kind, "first_month": group[0]["month"], "last_month": group[-1]["month"], "count": len(group),
        "start_t": group[0][f"{kind}_t"], "end_t": group[-1][f"{kind}_t"],
        "start_day": group[0][f"{kind}_day"], "end_day": group[-1][f"{kind}_day"],
        "best_month": best["month"], "best_day": best[f"{kind}_day"], "best_offset": best[f"{kind}_offset"],
    }
