from __future__ import annotations

from .model import Moon

# Each preset: (description, Liwaya, Nineya). Canon: Liwaya is the larger moon,
# Nineya the smaller. Inclinations are relative to the planet's orbital plane.
# Orbits and sizes are hypothetical design choices, not canon.
PRESETS: dict[str, tuple[str, Moon, Moon]] = {
    "A. Brief candidates as written": (
        "240,000 / 400,000 km with Moon-like disks. Included as a cautionary case: "
        "expect a moon to be lost within decades to centuries.",
        Moon("Liwaya", 3500, 3.3, 400_000, 0.01, 3.0, 120, 200, 80),
        Moon("Nineya", 2100, 3.3, 240_000, 0.01, 1.5, 0, 40, 10),
    ),
    "B. Large outer, small inner (selected design)": (
        "Liwaya is the large outer moon and dominant tide-raiser; Nineya is a small close moon. "
        "Orbits are tuned so that 39 Liwaya months = 155 Nineya cycles = 4 Bayurindan years, and the "
        "start moment is a double full moon with the two moons overlapping. Tilts are 1.5 degrees. "
        "Listed distances are the starting osculating values (the average Liwaya distance is slightly smaller).",
        Moon("Liwaya", 3450, 3.3, 364_462, 0.01, 1.5, 0, 0, 0),
        Moon("Nineya", 1200, 3.3, 150_047, 0.01, 1.5, 0, 0, 0),
    ),
    "C. 2:1 resonant pair": (
        "Period ratio 2.00. Liwaya orbits exactly twice as slowly as Nineya, so "
        "conjunction geometry repeats. Resonance is stabilising only if it is genuinely locked.",
        Moon("Liwaya", 2200, 3.3, 254_000, 0.01, 3.0, 120, 200, 80),
        Moon("Nineya", 1100, 3.3, 160_000, 0.01, 1.5, 0, 40, 10),
    ),
    "D. 3:1 period ratio": (
        "Period ratio ~3.0, equal apparent size and equal tides. Two similar-looking moons "
        "on clearly different clocks; Liwaya is physically the larger.",
        Moon("Liwaya", 2700, 3.3, 312_000, 0.01, 3.0, 120, 200, 80),
        Moon("Nineya", 1300, 3.3, 150_000, 0.01, 1.5, 0, 40, 10),
    ),
    "E. Mid-range, smaller moons": (
        "190,000 / 330,000 km, both ~0.4 degrees. Weaker tides (~0.45 of Earth's lunar tide each), "
        "comfortable stability margin.",
        Moon("Liwaya", 2300, 3.3, 330_000, 0.01, 3.0, 120, 200, 80),
        Moon("Nineya", 1330, 3.3, 190_000, 0.01, 1.5, 0, 40, 10),
    ),
}

DEFAULT_PRESET = "B. Large outer, small inner (selected design)"
