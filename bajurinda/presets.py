from __future__ import annotations

from .model import Moon

# Design B is the selected moon design (hypothetical, not yet canon). Canon: Liwaya is the larger moon.
# Both orbits are tilted 1.5 degrees to the ecliptic. Where the moons start (day 1) comes from data/aureth.yaml. Orbits are tuned so a Liwaya phase cycle is 38.5 Bajurindan days and a Nineya cycle 9.70,
# so 32 Liwaya cycles = 127 Nineya cycles = 1,232 days (3.29 years, not tied to the year).
# Distances are starting osculating values for the old aligned start; the simulated averages are ~362,900 and ~150,250 km. Each saved day 1 in
# data/aureth.yaml carries its own retuned starting distances (start_distance_km), because where a moon starts changes how its starting distance
# relates to its average distance (about 1%).
# The starting distances were nudged by 38 km and 4 km (from 367,434 and 150,860) when the planet and its moons were placed so that their
# shared centre of mass follows the 440-day orbit; that restored the 38.5 and 9.70 day phase cycles exactly.
# Earlier alternatives (brief candidates, 2:1, 3:1, mid-range) are in the git history.
PRESETS: dict[str, tuple[str, Moon, Moon]] = {
    "Design B": (
        "Liwaya is the large outer moon and dominant tide-raiser; Nineya is a small close moon. "
        "Liwaya phase cycle 38.5 Bajurindan days, Nineya 9.70; 32 Liwaya cycles = 127 Nineya cycles.",
        Moon("Liwaya", 3450, 3.3, 367_396, 0.01, 1.5, 0, 0, 0),
        Moon("Nineya", 1200, 3.3, 150_856, 0.01, 1.5, 0, 0, 0),
    ),
    "Design C": (
        "Experiment: Liwaya as a captured body. Its orbit is eccentric (0.08) and tilted 2 degrees, so it swells and shrinks by about 6% "
        "either way as it moves between nearest and farthest. Liwaya is denser and a little smaller (3.6 g/cm3, 3,351 km), Nineya lighter and a "
        "little bigger (3.0 g/cm3, 1,239 km), each with the same mass as in Design B. The starting distances are retuned so the phase cycles "
        "stay at 38.5 and 9.70 days. Nineya's orbit is pumped to about the same eccentricity by Liwaya. Stable for 10,000 years in the tests "
        "(10 of 10 random-phase starts for 1,000 years); eccentricity 0.10 starts to fail some random starts and 0.15 breaks Nineya's orbit.",
        Moon("Liwaya", 3350.9, 3.6, 366_505, 0.08, 2.0, 0, 0, 0),
        Moon("Nineya", 1238.7, 3.0, 151_111, 0.01, 1.5, 0, 0, 0),
    ),
}

DEFAULT_PRESET = "Design B"
