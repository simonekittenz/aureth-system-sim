"""The whole Aureth system as one object: the star, every planet with a known orbit, and the moons that belong to each planet.

Both pages build their picture from this. A planet is a `Planet` (spin, size, mass and orbit); `System` picks one planet as the
point of view and carries its moons and the other planets, so one REBOUND simulation serves every view."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from .aureth import Body, Star, body_from_planet
from .constants import G, G_EARTH, M_EARTH
from .model import Moon, Planet, System, canon_planet, mass_from_gravity
from .presets import DEFAULT_PRESET, PRESETS

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "aureth.yaml"


@dataclass(frozen=True)
class World:
    star: Star
    planets: tuple[Planet, ...]
    moons: tuple[tuple[str, tuple[Moon, ...]], ...] = ()      # (planet name, its moons)
    start_seed: int = 0                                         # the seed the saved day 1 was made from
    start_distances: tuple[tuple[str, str, float], ...] = ()    # (design, moon, retuned starting distance in km) for the saved day 1

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.planets]

    def planet(self, name: str) -> Planet:
        return next(p for p in self.planets if p.name == name)

    def moons_of(self, name: str) -> tuple[Moon, ...]:
        return next((m for n, m in self.moons if n == name), ())

    def system(self, name: str) -> System:
        """The world as seen from one planet: its moons are the moons, every other planet is a neighbour."""
        return System(self.planet(name), self.moons_of(name), tuple(p for p in self.planets if p.name != name))

    def bodies(self) -> list[Body]:
        return [body_from_planet(p) for p in self.planets]


def load_world(path: Path = DATA_FILE) -> World:
    """Read data/aureth.yaml. Bayurinda comes from `canon_planet()` (its distance follows from the 440-day year); the other
    planets come from the file. Moons are the Design B preset."""
    d = yaml.safe_load(Path(path).read_text())
    base = canon_planet()
    star = Star(d["star"]["name"], base.star_mass_msun, d["star"]["luminosity_lsun"], d["star"]["radius_rsun"])
    start = d.get("system_start", {})
    phases = start.get("phase_deg", {})
    axes = start.get("axis_azimuth_deg", {})
    planets = []
    for p in d["planets"]:
        if p.get("distance_au") is None:
            continue
        name = p["name"]
        if name == "Bayurinda":
            pl = base
        else:
            radius = p["diameter_km"] / 2
            if p.get("mass_earth") is not None:
                mass_kg = float(p["mass_earth"]) * M_EARTH
            else:
                mass_kg = mass_from_gravity(float(p["gravity_g"]), radius)
            pl = replace(base, name=name, radius_km=float(radius), mass_kg=mass_kg, rotation_hours=float(p.get("day_hours", 24.0)),
                         rotation_is_sidereal=False, a_au=float(p["distance_au"]), ecc=float(p.get("eccentricity", 0.0)),
                         obliquity_deg=float(p.get("axial_tilt_deg", 0.0)))
        holes = p.get("placeholders", [])
        words = {"axial_tilt_deg": "axial tilt", "day_hours": "day length", "eccentricity": "eccentricity"}
        names = [words.get(h, str(h).replace("_", " ")) for h in holes]
        note = (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else (names[0] if names else "")) + (" are placeholders" if len(names) > 1 else " is a placeholder" if names else "")
        planets.append(replace(pl, mean_anomaly_deg=float(phases.get(name, 0.0)), axis_azimuth_deg=float(axes.get(name, 0.0)), placeholder_note=note))
    _, liw, nin = PRESETS[DEFAULT_PRESET]
    starts = start.get("moons", {})
    dists = tuple((str(design), name, float(km)) for name, mv in starts.items() for design, km in mv.get("start_distance_km", {}).items())
    mine = {name: km for design, name, km in dists if design == DEFAULT_PRESET}
    moons = tuple(replace(m, mean_anomaly_deg=float(starts.get(m.name, {}).get("mean_anomaly_deg", m.mean_anomaly_deg)),
                          asc_node_deg=float(starts.get(m.name, {}).get("asc_node_deg", m.asc_node_deg)),
                          arg_peri_deg=float(starts.get(m.name, {}).get("arg_peri_deg", m.arg_peri_deg)),
                          a_km=mine.get(m.name, m.a_km)) for m in (liw, nin))
    return World(star, tuple(planets), (("Bayurinda", moons),), int(start.get("seed", 0)), dists)
