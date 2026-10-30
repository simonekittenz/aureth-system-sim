"""Year 1, day 1: where every planet and moon starts, and which way each axis leans.

A start is generated from a seed. Draws that look contrived are thrown away and redrawn: planets lined up with each other, moons on top of each
other or both full or both new. The moons are put at random places on their orbits and the whole moon system is turned at random, but the two
moons' orbits keep the same relation to each other, because that is the arrangement the stability tests covered; a fully random relation would give
Nineya a large extra eccentricity and tilt.

Where a moon starts also changes how its starting distance relates to its average distance, so every new start needs the distances retuned to keep the
phase cycles at 38.5 and 9.70 days (`retune_distances`). The numbers are written into data/aureth.yaml (`system_start`), and can be edited by hand.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from dataclasses import replace

from .dynamics import mean_rates
from .model import System
from .sky import start_direction
from .world import DATA_FILE, World


def _moon_vector(m, rng_vals) -> np.ndarray:
    """Direction of a moon from its planet at day 1, from its starting orbit (position on the orbit and orientation of the orbit)."""
    M, Om, w = (math.radians(v) for v in rng_vals)
    e, inc = m.ecc, math.radians(m.inc_deg)
    E = M + e * math.sin(M)
    for _ in range(30):
        E -= (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
    x, y = math.cos(E) - e, math.sqrt(1 - e * e) * math.sin(E)
    cw, sw, ci, si, cn, sn = math.cos(w), math.sin(w), math.cos(inc), math.sin(inc), math.cos(Om), math.sin(Om)
    v = np.array([(cn * cw - sn * sw * ci) * x + (-cn * sw - sn * cw * ci) * y, (sn * cw + cn * sw * ci) * x + (-sn * sw + cn * cw * ci) * y, (sw * si) * x + (cw * si) * y])
    return v / np.linalg.norm(v)


def random_start(seed: int, base: World) -> dict[str, float]:
    """A natural-looking day 1, as widget-key -> value (the keys the sidebar uses)."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    for _ in range(5000):
        phase = {p.name: float(rng.uniform(0, 360)) for p in base.planets}
        lon = {p.name: math.degrees(math.atan2(*start_direction(_with_phase(p, phase[p.name]))[1::-1])) for p in base.planets}
        names = list(lon)
        if any(abs((lon[a] - lon[b] + 180) % 360 - 180) < 25 for i, a in enumerate(names) for b in names[i + 1:]):
            continue                                                  # two planets almost in line seen from the star
        axis = {p.name: float(rng.uniform(0, 360)) for p in base.planets}
        moons, ok = {}, True
        for host, ms in base.moons:
            sun = -start_direction(_with_phase(base.planet(host), phase[host]))
            node, peri = float(rng.uniform(0, 360)), float(rng.uniform(0, 360))   # one turn for the whole moon system, so the moons' orbits keep their relation to each other
            vecs, ks = [], []
            for m in ms:
                vals = (float(rng.uniform(0, 360)), node, peri)
                moons[m.name] = vals
                v = _moon_vector(m, vals)
                vecs.append(v)
                ks.append((1 - float(v @ sun)) / 2)                   # lit fraction: 0 when the moon is in the Sun's direction (new), 1 opposite (full)
            if len(vecs) > 1:
                sep = math.degrees(math.acos(float(np.clip(vecs[0] @ vecs[1], -1, 1))))
                if sep < 25 or all(k > 0.85 for k in ks) or all(k < 0.15 for k in ks):
                    ok = False
        if not ok:
            continue
        for n in phase:
            out[f"w_{n}_ph"] = round(phase[n], 1)
            out[f"w_{n}_axis"] = round(axis[n], 1)
        for n, (M, Om, w) in moons.items():
            out.update({f"w_{n}_M": round(M, 1), f"w_{n}_Om": round(Om, 1), f"w_{n}_w": round(w, 1)})
        return out
    raise RuntimeError("could not draw a natural-looking day 1")


def _with_phase(planet, phase):
    from dataclasses import replace
    return replace(planet, mean_anomaly_deg=phase)


HEADER = """# Where everything starts at year 1, day 1 (a placeholder, made from the seed below; the Day 1 box in the sidebar rolls a new one or saves the current numbers).
# phase_deg: where each planet is on its orbit (0 = closest to Aureth). axis_azimuth_deg: which way each planet's axis leans (this sets the season on day 1;
# 0 is a winter solstice at closest approach). moons: where each moon is on its orbit (mean anomaly) and how its orbit is turned (node, periapsis).
# start_distance_km: the starting distance retuned for this start, per moon design, so the phase cycles stay at 38.5 and 9.70 days.
"""


def system_start_block(seed: int, values: dict, base: World, distances: dict | None = None) -> str:
    f = lambda v: f"{float(v):g}"
    pl = ", ".join(f"{p.name}: {f(values[f'w_{p.name}_ph'])}" for p in base.planets)
    ax = ", ".join(f"{p.name}: {f(values[f'w_{p.name}_axis'])}" for p in base.planets)
    lines = [HEADER + "system_start:", "  status: placeholder", f"  seed: {int(seed)}", f"  phase_deg: {{{pl}}}", f"  axis_azimuth_deg: {{{ax}}}", "  moons:"]
    for _host, ms in base.moons:
        for k, m in enumerate(ms):
            dist = ""
            if distances:
                dist = ", start_distance_km: {" + ", ".join(f"{d}: {distances[d][k]:.0f}" for d in distances) + "}"
            lines.append(f"    {m.name}: {{mean_anomaly_deg: {f(values[f'w_{m.name}_M'])}, asc_node_deg: {f(values[f'w_{m.name}_Om'])}, arg_peri_deg: {f(values[f'w_{m.name}_w'])}{dist}}}")
    lines.append("  inclination_deg: 0         # placeholder: all orbits flat in one plane")
    return "\n".join(lines) + "\n"


def save_start(seed: int, values: dict, base: World, distances: dict | None = None, path: Path = DATA_FILE) -> None:
    """Replace the system_start block (and the comments just above it) in the data file, leaving the rest alone."""
    text = Path(path).read_text()
    pat = re.compile(r"(?:^#[^\n]*\n)*^system_start:\n(?:^[ \t]+[^\n]*\n)*", re.M)
    if not pat.search(text):
        raise ValueError("could not find system_start in the data file")
    Path(path).write_text(pat.sub(lambda _m: system_start_block(seed, values, base, distances), text, count=1))


TARGET_SYNODIC = (38.5, 32 * 38.5 / 127)       # local days: Liwaya's phase cycle, and the Nineya cycle that makes 32 of Liwaya's = 127 of Nineya's


def retune_distances(system: System, days: int = 6000, tol: float = 3e-4, max_iter: int = 6) -> tuple[float, float]:
    """Starting distances (km) for the two moons that give the target phase cycles from this start. Takes about 10 seconds."""
    D = system.planet.local_day_days
    a = [m.a_km for m in system.moons]
    slope = [1.7e-4, 1.0e-4]                                       # change in phase cycle (local days) per km, refined by a secant after the first step
    prev: list = [None, None]
    for _ in range(max_iter):
        meas = np.array(mean_rates(System(system.planet, tuple(replace(m, a_km=x) for m, x in zip(system.moons, a)), system.neighbors), days=days)["synodic"]) / D
        err = meas - np.array(TARGET_SYNODIC)
        if np.all(np.abs(err) < tol):
            break
        for k in range(2):
            if prev[k] is not None and abs(a[k] - prev[k][0]) > 1e-6:
                slope[k] = (meas[k] - prev[k][1]) / (a[k] - prev[k][0])
            prev[k] = (a[k], meas[k])
            a[k] -= err[k] / slope[k]
    return float(a[0]), float(a[1])
