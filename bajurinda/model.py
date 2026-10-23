from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .constants import (
    AU_KM, D_LUNA_KM, DAY_S, G, G_EARTH, M_EARTH, M_LUNA, M_SUN,
    R_EARTH_KM, R_LUNA_KM, R_SUN_KM, YEAR_DAYS,
)

KM = 1e3


def kepler_period_days(a_km: float, total_mass_kg: float) -> float:
    a = a_km * KM
    return 2 * math.pi * math.sqrt(a**3 / (G * total_mass_kg)) / DAY_S


def mass_from_gravity(gravity_g: float, radius_km: float) -> float:
    return gravity_g * G_EARTH * (radius_km * KM) ** 2 / G


@dataclass(frozen=True)
class Planet:
    name: str = "Bajurinda"
    radius_km: float = 6000.0
    mass_kg: float = field(default_factory=lambda: mass_from_gravity(0.9, 6000.0))
    rotation_hours: float = 19.0
    rotation_is_sidereal: bool = False
    a_au: float = 0.87
    ecc: float = 0.04
    obliquity_deg: float = 16.0
    star_mass_kg: float = 0.0
    star_luminosity_lsun: float = 1.0
    star_radius_rsun: float = 1.0

    @property
    def mass_earth(self) -> float:
        return self.mass_kg / M_EARTH

    @property
    def gravity_g(self) -> float:
        return G * self.mass_kg / (self.radius_km * KM) ** 2 / G_EARTH

    @property
    def density_gcc(self) -> float:
        vol = 4 / 3 * math.pi * (self.radius_km * KM) ** 3
        return self.mass_kg / vol / 1000.0

    @property
    def escape_velocity_kms(self) -> float:
        return math.sqrt(2 * G * self.mass_kg / (self.radius_km * KM)) / 1e3

    @property
    def a_km(self) -> float:
        return self.a_au * AU_KM

    @property
    def year_days(self) -> float:
        return kepler_period_days(self.a_km, self.star_mass_kg + self.mass_kg)

    @property
    def sidereal_rotation_hours(self) -> float:
        if self.rotation_is_sidereal:
            return self.rotation_hours
        # supplied value is a mean solar day: 1/solar = 1/sidereal - 1/year (prograde)
        year_h = self.year_days * 24
        return 1 / (1 / self.rotation_hours + 1 / year_h)

    @property
    def solar_day_hours(self) -> float:
        year_h = self.year_days * 24
        return 1 / (1 / self.sidereal_rotation_hours - 1 / year_h)

    @property
    def sidereal_rotations_per_year(self) -> float:
        return self.year_days * 24 / self.sidereal_rotation_hours

    @property
    def solar_days_per_year(self) -> float:
        return self.year_days * 24 / self.solar_day_hours

    @property
    def irradiance_earth(self) -> float:
        return self.star_luminosity_lsun / self.a_au**2

    @property
    def star_mass_msun(self) -> float:
        return self.star_mass_kg / M_SUN

    @property
    def local_day_days(self) -> float:
        return self.solar_day_hours / 24

    @property
    def hill_radius_km(self) -> float:
        return self.a_km * (self.mass_kg / (3 * self.star_mass_kg)) ** (1 / 3)

    @property
    def synchronous_orbit_km(self) -> float:
        T = self.sidereal_rotation_hours * 3600
        return (G * self.mass_kg * T**2 / (4 * math.pi**2)) ** (1 / 3) / KM

    @property
    def j2_estimate(self) -> float:
        # J2 ~ 0.31 q (Earth-like fluid response), q = w^2 R^3 / GM
        w = 2 * math.pi / (self.sidereal_rotation_hours * 3600)
        q = w**2 * (self.radius_km * KM) ** 3 / (G * self.mass_kg)
        return 0.314 * q

    @property
    def laplace_radius_km(self) -> float:
        r5 = (
            2 * self.j2_estimate * (self.radius_km) ** 2 * self.a_km**3
            * (1 - self.ecc**2) ** 1.5 * self.mass_kg / self.star_mass_kg
        )
        return r5 ** 0.2

    @property
    def star_radius_km(self) -> float:
        return self.star_radius_rsun * R_SUN_KM

    def sun_angular_diameter_deg(self) -> float:
        return angular_diameter_deg(self.star_radius_km, self.a_km)


def canon_planet(year_local_days: float = 374.0, day_hours: float = 19.0) -> Planet:
    """Planet with the canon values. Star mass is derived from the canon year."""
    p = Planet(rotation_hours=day_hours, rotation_is_sidereal=False)
    year_s = year_local_days * day_hours * 3600
    total = 4 * math.pi**2 * (p.a_km * KM) ** 3 / (G * year_s**2)
    return replace(p, star_mass_kg=total - p.mass_kg)


@dataclass(frozen=True)
class Moon:
    name: str
    diameter_km: float
    density_gcc: float
    a_km: float
    ecc: float = 0.0
    inc_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    asc_node_deg: float = 0.0
    arg_peri_deg: float = 0.0

    @property
    def radius_km(self) -> float:
        return self.diameter_km / 2

    @property
    def mass_kg(self) -> float:
        vol = 4 / 3 * math.pi * (self.radius_km * KM) ** 3
        return self.density_gcc * 1000.0 * vol

    @property
    def mass_luna(self) -> float:
        return self.mass_kg / M_LUNA

    def with_mass_luna(self, mass_luna: float) -> "Moon":
        vol = (mass_luna * M_LUNA) / (self.density_gcc * 1000.0)
        radius_km = (vol * 3 / (4 * math.pi)) ** (1 / 3) / KM
        return replace(self, diameter_km=2 * radius_km)


@dataclass(frozen=True)
class System:
    planet: Planet
    moons: tuple[Moon, Moon]


def moon_period_days(planet: Planet, moon: Moon) -> float:
    return kepler_period_days(moon.a_km, planet.mass_kg + moon.mass_kg)


def synodic_period_days(sidereal_days: float, year_days: float) -> float:
    """Phase cycle: time between successive identical phases as seen from the planet."""
    rate = abs(1 / sidereal_days - 1 / year_days)
    return math.inf if rate == 0 else 1 / rate


def angular_diameter_deg(radius_km: float, distance_km: float) -> float:
    return math.degrees(2 * math.atan(radius_km / distance_km))


def tide_amplitude_m(planet: Planet, body_mass_kg: float, distance_km: float) -> float:
    """Peak equilibrium-tide height of the degree-2 bulge: (m/M)(R/d)^3 R."""
    R = planet.radius_km * KM
    return body_mass_kg / planet.mass_kg * (R / (distance_km * KM)) ** 3 * R


def earth_moon_tide_amplitude_m() -> float:
    R = R_EARTH_KM * KM
    return M_LUNA / M_EARTH * (R / (D_LUNA_KM * KM)) ** 3 * R


def tide_vs_luna(planet: Planet, body_mass_kg: float, distance_km: float) -> float:
    return tide_amplitude_m(planet, body_mass_kg, distance_km) / earth_moon_tide_amplitude_m()


def stable_limit_km(planet: Planet, moon_ecc: float) -> float:
    """Prograde single-satellite stability limit (Domingos et al. 2006)."""
    return 0.4895 * planet.hill_radius_km * (1 - 1.0305 * planet.ecc - 0.2738 * moon_ecc)


def roche_fluid_km(planet: Planet, moon: Moon) -> float:
    return 2.456 * planet.radius_km * (planet.density_gcc / moon.density_gcc) ** (1 / 3)


def roche_rigid_km(planet: Planet, moon: Moon) -> float:
    return 1.26 * planet.radius_km * (planet.density_gcc / moon.density_gcc) ** (1 / 3)


def mutual_hill_separation(planet: Planet, inner: Moon, outer: Moon) -> float:
    """Spacing in mutual Hill radii, using peri/apo distances. Coplanar two-body
    stability needs roughly > 2*sqrt(3) = 3.46 for circular orbits."""
    rh = ((inner.mass_kg + outer.mass_kg) / (3 * planet.mass_kg)) ** (1 / 3) * (
        (inner.a_km + outer.a_km) / 2
    )
    gap = outer.a_km * (1 - outer.ecc) - inner.a_km * (1 + inner.ecc)
    return gap / rh


def near_resonances(period_ratio: float, max_order: int = 4, tol: float = 0.05):
    """Mean-motion resonances p:q (outer:inner = q:p ... returned as inner:outer) within tol."""
    out = []
    for q in range(1, 9):
        for p in range(q + 1, q + max_order + 1):
            if math.gcd(p, q) != 1:
                continue
            target = p / q
            off = (period_ratio - target) / target
            if abs(off) <= tol:
                out.append({"resonance": f"{p}:{q}", "order": p - q, "offset_pct": 100 * off})
    return sorted(out, key=lambda r: abs(r["offset_pct"]))


def moon_report(system: System, moon: Moon) -> dict:
    pl = system.planet
    P = moon_period_days(pl, moon)
    syn = synodic_period_days(P, pl.year_days)
    peri = moon.a_km * (1 - moon.ecc)
    apo = moon.a_km * (1 + moon.ecc)
    limit = stable_limit_km(pl, moon.ecc)
    return {
        "name": moon.name,
        "mass_luna": moon.mass_luna,
        "mass_kg": moon.mass_kg,
        "a_km": moon.a_km,
        "a_planet_radii": moon.a_km / pl.radius_km,
        "a_hill_frac": moon.a_km / pl.hill_radius_km,
        "stability_fraction": apo / limit,
        "sidereal_days": P,
        "sidereal_local_days": P / pl.local_day_days,
        "synodic_days": syn,
        "synodic_local_days": syn / pl.local_day_days,
        "ang_diam_mean_deg": angular_diameter_deg(moon.radius_km, moon.a_km),
        "ang_diam_peri_deg": angular_diameter_deg(moon.radius_km, peri - pl.radius_km),
        "ang_diam_apo_deg": angular_diameter_deg(moon.radius_km, apo),
        "tide_amp_m": tide_amplitude_m(pl, moon.mass_kg, moon.a_km),
        "tide_vs_luna": tide_vs_luna(pl, moon.mass_kg, moon.a_km),
        "tide_vs_luna_peri": tide_vs_luna(pl, moon.mass_kg, peri),
        "tide_vs_luna_apo": tide_vs_luna(pl, moon.mass_kg, apo),
        "roche_fluid_km": roche_fluid_km(pl, moon),
        "roche_rigid_km": roche_rigid_km(pl, moon),
        "outside_roche": peri > roche_fluid_km(pl, moon),
        "stable_limit_km": limit,
        "beyond_synchronous": moon.a_km > pl.synchronous_orbit_km,
        "beyond_laplace": moon.a_km > pl.laplace_radius_km,
        "mass_ratio": moon.mass_kg / pl.mass_kg,
    }


def system_report(system: System) -> dict:
    pl = system.planet
    inner, outer = sorted(system.moons, key=lambda m: m.a_km)
    r_in, r_out = moon_report(system, inner), moon_report(system, outer)
    ratio = r_out["sidereal_days"] / r_in["sidereal_days"]
    sun_tide = tide_amplitude_m(pl, pl.star_mass_kg, pl.a_km)
    return {
        "inner": inner.name,
        "outer": outer.name,
        "period_ratio": ratio,
        "resonances": near_resonances(ratio),
        "mutual_hill_sep": mutual_hill_separation(pl, inner, outer),
        "sun_tide_amp_m": sun_tide,
        "sun_tide_vs_luna": sun_tide / earth_moon_tide_amplitude_m(),
        "combined_max_tide_m": r_in["tide_amp_m"] + r_out["tide_amp_m"] + sun_tide,
        "moons": {inner.name: r_in, outer.name: r_out},
    }


def planet_warnings(system: System) -> list[tuple[str, str]]:
    """(severity, message) pairs. severity: error | warning | info."""
    pl = system.planet
    msgs: list[tuple[str, str]] = []
    rep = system_report(system)
    for name, r in rep["moons"].items():
        if not r["outside_roche"]:
            msgs.append(("error", f"{name}: periapsis is inside the fluid Roche limit ({r['roche_fluid_km']:,.0f} km)."))
        if r["stability_fraction"] > 1:
            msgs.append(("error", f"{name}: apoapsis is beyond the ~{r['stable_limit_km']:,.0f} km Hill-stability limit."))
        elif r["stability_fraction"] > 0.75:
            msgs.append(("warning", f"{name}: at {r['stability_fraction']:.0%} of the single-moon stability limit; run the N-body check."))
        if not r["beyond_synchronous"]:
            msgs.append(("warning", f"{name}: inside synchronous orbit ({pl.synchronous_orbit_km:,.0f} km); tides would drag it inward."))
    if rep["mutual_hill_sep"] < 3.46:
        msgs.append(("error", f"Moons are only {rep['mutual_hill_sep']:.1f} mutual Hill radii apart (< 3.46); likely unstable."))
    for res in rep["resonances"][:1]:
        if abs(res["offset_pct"]) < 1.5:
            msgs.append(("info", f"Moons are near a {res['resonance']} period resonance ({res['offset_pct']:+.2f}%)."))
    return msgs
