from __future__ import annotations

import math

import numpy as np

from .aureth import Body, Star, build_nbody, grand_alignments, period_days, synodic_days
from .constants import G
from .dynamics import integrate
from .eras import era_bands, eras_data
from .model import System
from .sky import elongation


def moon_cycles(system: System, years: int = 250, step: float = 0.25) -> dict:
    """Synodic, draconic (node to node) and anomalistic (perigee to perigee) months for each moon, measured from the
    N-body run, plus the eclipse year, node regression period and the best eclipse-style cycle (whole numbers of all three)."""
    pl = system.planet
    Y = pl.year_days
    t = np.arange(0, years * Y, step)
    tr = integrate(system, t, elements=False)
    out = {}
    for k, moon in enumerate(system.moons):
        pos = tr.moon_pos[:, k]
        z, r = pos[:, 2], np.linalg.norm(pos, axis=1)
        up = np.where((z[:-1] < 0) & (z[1:] >= 0))[0]
        tn = t[up] - z[up] * step / (z[up + 1] - z[up])
        draconic = float(np.diff(tn).mean())
        w = max(int(2 / step), 1)
        rs = np.convolve(r, np.ones(w) / w, mode="same")
        mins = np.where((rs[1:-1] < rs[:-2]) & (rs[1:-1] <= rs[2:]))[0] + 1
        anomalistic = float(np.diff(t[mins]).mean())
        syn = float(2 * math.pi / np.polyfit(t, elongation(pos, tr.star_pos), 1)[0])
        sidereal = 1 / (1 / syn + 1 / Y)
        regress = 1 / draconic - 1 / sidereal                       # cycles/day the nodes move backwards
        eclipse_year = 1 / (1 / Y + regress)
        cands = []
        for n in range(12, 900):
            m, a = round(n * syn / draconic), round(n * syn / anomalistic)
            cands.append((n, m, a, abs(n * syn - m * draconic), abs(n * syn - a * anomalistic)))
        cands.sort(key=lambda c: c[3] + 0.5 * c[4])
        best = cands[0]
        out[moon.name] = {"synodic": syn, "draconic": draconic, "anomalistic": anomalistic, "eclipse_year": eclipse_year,
                          "node_regression_years": 1 / regress / Y, "cycle_months": best[0], "cycle_draconic": best[1],
                          "cycle_anomalistic": best[2], "cycle_off_days": best[3], "cycle_years": best[0] * syn / Y,
                          "cycle_alt": [(c[0], c[0] * syn / Y, c[3]) for c in cands[1:60] if abs(c[0] - best[0]) > 5][:3]}
    return out


def plane_cycle_years(system: System, years: int = 160) -> float | None:
    """Spacing between the times the two moons' orbital planes line up again."""
    ed = eras_data(system, years)
    bands = era_bands(ed, 1.0)
    mut = np.nan_to_num(ed.mutual_deg, nan=99.0)
    mins = []
    for a, b in bands:
        sel = (ed.plane_years >= a) & (ed.plane_years <= b)
        mins.append(float(ed.plane_years[sel][np.argmin(mut[sel])]))
    return float((mins[-1] - mins[0]) / (len(mins) - 1)) if len(mins) > 1 else None


def planet_repeat(star: Star, bodies: list[Body], max_cycles: int = 120, tol: float = 0.03) -> dict | None:
    """Smallest span after which the Bayurinda-Ashara, Bayurinda-Tarnic and Ashara-Tarnic phases all come back (whole synodic cycles)."""
    by = {b.name: b for b in bodies}
    if not all(n in by for n in ("Bayurinda", "Ashara", "Tarnic")):
        return None
    P = {n: period_days(star, by[n]) for n in by}
    s_ba, s_bt, s_at = synodic_days(P["Bayurinda"], P["Ashara"]), synodic_days(P["Bayurinda"], P["Tarnic"]), synodic_days(P["Ashara"], P["Tarnic"])
    for k in range(1, max_cycles):
        T = k * s_ba
        n_bt, n_at = T / s_bt, T / s_at
        if abs(n_bt - round(n_bt)) < tol and abs(n_at - round(n_at)) < tol * 3:
            return {"years": T / P["Bayurinda"], "ba": k, "bt": round(n_bt), "at": round(n_at),
                    "off": (abs(n_bt - round(n_bt)), abs(n_at - round(n_at)))}
    return None


def great_cycle(periods: dict[str, float], tol: float = 0.0025, tmax: float = 1500.0) -> dict | None:
    """Smallest span that is a whole number of every given cycle, to within `tol` (as a fraction)."""
    names = list(periods)
    best = None
    t = 0.0
    for T in np.arange(min(periods.values()), tmax, 0.05):
        counts = [max(round(T / p), 1) for p in periods.values()]
        spans = [c * p for c, p in zip(counts, periods.values())]
        spread = max(spans) / min(spans) - 1
        if spread < tol and min(counts) >= 1:
            return {"years": float(np.mean(spans)), "counts": dict(zip(names, counts)), "spread": float(spread)}
    return best


def axial_precession(system: System, cm: float = 0.33) -> dict:
    """Lunisolar precession of the spin axis. Dynamical ellipticity from the rotation rate (J2 ~ 0.314 q) and a
    moment-of-inertia factor C/(M R^2) = cm. Moons are assumed to orbit near the ecliptic."""
    pl = system.planet
    w = 2 * math.pi / (pl.sidereal_rotation_hours * 3600)
    q = w**2 * (pl.radius_km * 1e3) ** 3 / (G * pl.mass_kg)
    H = 0.314 * q / cm
    n_p = 2 * math.pi / (pl.year_days * 86400)
    terms = {"Aureth": n_p**2}
    for m in system.moons:
        terms[m.name] = G * m.mass_kg / (m.a_km * 1e3) ** 3
    rate = 1.5 * H * math.cos(math.radians(pl.obliquity_deg)) / w * sum(terms.values())
    years = 2 * math.pi / rate / 3.15576e7
    return {"earth_years": years, "years": years * 365.25 / pl.year_days, "shares": {k: v / sum(terms.values()) for k, v in terms.items()}}


def orbit_secular(star: Star, bodies: list[Body], years: int = 400_000, n: int = 3000) -> dict:
    """Slow swelling and shrinking of Bayurinda's orbit (eccentricity) caused by the other planets."""
    sim = build_nbody(star, bodies)
    sim.integrator = "whfast"
    sim.dt = 0.04 * min(period_days(star, b) for b in bodies)
    ts = np.linspace(0, years * 365.25, n)
    e = np.zeros(n)
    for i, t in enumerate(ts):
        sim.integrate(float(t))
        e[i] = sim.particles[1].orbit(primary=sim.particles[0]).e
    dt_y = years / n
    F = np.abs(np.fft.rfft((e - e.mean()) * np.hanning(n), 16 * n))
    f = np.fft.rfftfreq(16 * n, d=dt_y)
    j = int(np.argmax(F[1:]) + 1)
    return {"e_min": float(e.min()), "e_max": float(e.max()), "period_earth_years": float(1 / f[j]), "span_years": years}
