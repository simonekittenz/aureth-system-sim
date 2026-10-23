from __future__ import annotations

from .model import Moon

# Design B is the selected moon design (hypothetical, not yet canon). Canon: Liwaya is the larger moon.
# Both orbits are tilted 1.5 degrees to the ecliptic and the run starts at a double full moon with the
# moons overlapping. Orbits are tuned so 39 Liwaya cycles = 155 Nineya cycles = 4 Bajurindan years.
# Distances are the starting osculating values; the simulated averages are ~359,000 and ~149,400 km.
# Earlier alternatives (brief candidates, 2:1, 3:1, mid-range) are in the git history.
PRESETS: dict[str, tuple[str, Moon, Moon]] = {
    "Design B": (
        "Liwaya is the large outer moon and dominant tide-raiser; Nineya is a small close moon. "
        "39 Liwaya cycles = 155 Nineya cycles = 4 Bajurindan years. Starts at a double full moon.",
        Moon("Liwaya", 3450, 3.3, 364_462, 0.01, 1.5, 0, 0, 0),
        Moon("Nineya", 1200, 3.3, 150_047, 0.01, 1.5, 0, 0, 0),
    ),
}

DEFAULT_PRESET = "Design B"
