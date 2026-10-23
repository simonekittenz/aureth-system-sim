from __future__ import annotations

from .model import Moon

# Each preset: (description, Liwaya, Nineya). Inclinations are relative to the
# planet's orbital plane. All are hypothetical design choices, not canon.
PRESETS: dict[str, tuple[str, Moon, Moon]] = {
    "A. Brief candidates as written": (
        "240,000 / 400,000 km with Moon-like disks. Included as a cautionary case: "
        "expect Liwaya or Nineya to be lost within decades to centuries.",
        Moon("Liwaya", 2100, 3.3, 240_000, 0.01, 1.5, 0, 40, 10),
        Moon("Nineya", 3500, 3.3, 400_000, 0.01, 3.0, 120, 200, 80),
    ),
    "B. Small inner, big outer (selected design)": (
        "Period ratio ~3.7. Liwaya is a small close moon, Nineya is the dominant "
        "tide-raiser. Most robust against Solar perturbation.",
        Moon("Liwaya", 1200, 3.3, 150_000, 0.01, 1.5, 0, 40, 10),
        Moon("Nineya", 3450, 3.3, 360_000, 0.01, 3.0, 120, 200, 80),
    ),
    "C. 2:1 resonant pair": (
        "Period ratio 2.00. Nineya orbits exactly twice as slowly as Liwaya, so "
        "conjunction geometry repeats. Resonance is stabilising only if it is genuinely locked.",
        Moon("Liwaya", 1100, 3.3, 160_000, 0.01, 1.5, 0, 40, 10),
        Moon("Nineya", 2200, 3.3, 254_000, 0.01, 3.0, 120, 200, 80),
    ),
    "D. 3:1 period ratio": (
        "Period ratio ~3.0, equal apparent size and equal tides. Two similar moons "
        "on clearly different clocks.",
        Moon("Liwaya", 1300, 3.3, 150_000, 0.01, 1.5, 0, 40, 10),
        Moon("Nineya", 2700, 3.3, 312_000, 0.01, 3.0, 120, 200, 80),
    ),
    "E. Mid-range, smaller moons": (
        "190,000 / 330,000 km, both ~0.4 degrees. Weaker tides (~0.45 of Earth's lunar tide each), "
        "comfortable stability margin.",
        Moon("Liwaya", 1330, 3.3, 190_000, 0.01, 1.5, 0, 40, 10),
        Moon("Nineya", 2300, 3.3, 330_000, 0.01, 3.0, 120, 200, 80),
    ),
}

DEFAULT_PRESET = "B. Small inner, big outer (selected design)"
