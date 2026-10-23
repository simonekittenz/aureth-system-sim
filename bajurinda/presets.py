from __future__ import annotations

from .model import Moon

# Design B is the selected moon design (hypothetical, not yet canon). Canon: Liwaya is the larger moon.
# Both orbits are tilted 1.5 degrees to the ecliptic and the run starts at a double full moon with the
# moons overlapping. Orbits are tuned so a Liwaya phase cycle is 38.5 Bajurindan days and a Nineya cycle 9.70,
# so 32 Liwaya cycles = 127 Nineya cycles = 1,232 days (3.29 years, not tied to the year).
# Distances are the starting osculating values; the simulated averages are ~360,000 and ~149,900 km.
# Earlier alternatives (brief candidates, 2:1, 3:1, mid-range) are in the git history.
PRESETS: dict[str, tuple[str, Moon, Moon]] = {
    "Design B": (
        "Liwaya is the large outer moon and dominant tide-raiser; Nineya is a small close moon. "
        "Liwaya phase cycle 38.5 Bajurindan days, Nineya 9.70; 32 Liwaya cycles = 127 Nineya cycles. Starts at a double full moon.",
        Moon("Liwaya", 3450, 3.3, 367_434, 0.01, 1.5, 0, 0, 0),
        Moon("Nineya", 1200, 3.3, 150_860, 0.01, 1.5, 0, 0, 0),
    ),
}

DEFAULT_PRESET = "Design B"
