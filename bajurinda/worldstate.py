"""The settings of the whole world, shared by both pages.

Every number you can change in a sidebar (a planet's size, orbit and spin, the star, Bayurinda's moons) lives in the session under
a key starting with `w_`, so a change made on one page shows up on the other. The data files give the starting values; a value you
have not touched follows the file when the file changes, and a value you have changed stays as you left it.
"""
from __future__ import annotations

from dataclasses import replace

import streamlit as st

from .aureth import Star
from .constants import M_EARTH, M_SUN
from .model import Moon
from .presets import DEFAULT_PRESET, PRESETS
from .world import World

PLANET_KEYS = ("radius", "mass", "rot", "sid", "a", "ecc", "obl", "ph")
MOON_FIELDS = ("dia", "rho", "a", "e", "inc", "M", "Om", "w")


def _planet_defaults(base: World) -> dict:
    out = {}
    for p in base.planets:
        n = p.name
        out.update({f"w_{n}_radius": float(p.radius_km), f"w_{n}_mass": float(p.mass_earth), f"w_{n}_rot": float(p.rotation_hours),
                    f"w_{n}_sid": "sidereal" if p.rotation_is_sidereal else "solar", f"w_{n}_a": float(p.a_au), f"w_{n}_ecc": float(p.ecc),
                    f"w_{n}_obl": float(p.obliquity_deg), f"w_{n}_ph": float(p.mean_anomaly_deg)})
    return out


def _moon_defaults(base: World) -> dict:
    out = {}
    for _host, moons in base.moons:
        for m in moons:
            out.update({f"w_{m.name}_dia": float(m.diameter_km), f"w_{m.name}_rho": float(m.density_gcc), f"w_{m.name}_a": float(m.a_km),
                        f"w_{m.name}_e": float(m.ecc), f"w_{m.name}_inc": float(m.inc_deg), f"w_{m.name}_M": float(m.mean_anomaly_deg),
                        f"w_{m.name}_Om": float(m.asc_node_deg), f"w_{m.name}_w": float(m.arg_peri_deg)})
    return out


def defaults(base: World) -> dict:
    p0 = base.planets[0]
    return {**_planet_defaults(base), **_moon_defaults(base), "w_design": DEFAULT_PRESET, "w_star_mass": float(p0.star_mass_msun), "w_star_lum": float(p0.star_luminosity_lsun),
            "w_star_rad": float(p0.star_radius_rsun)}


def ensure_state(base: World) -> None:
    """Put the settings in place before any widget is drawn. Call this at the top of each page."""
    ss = st.session_state
    for k, v in ss.get("_world_saved", {}).items():       # Streamlit drops widgets' values when another page is shown
        if k not in ss:
            ss[k] = v
    new = defaults(base)
    old = ss.get("_world_defaults", {})
    for k, v in new.items():
        if k not in ss or (k in old and ss[k] == old[k] and old[k] != v):   # missing, or untouched and the file has changed
            ss[k] = v
    ss["_world_defaults"] = new
    for k in [k for k in ss.keys() if k.startswith("w_")]:      # re-assigning keeps a value alive when it is first drawn on another page
        ss[k] = ss[k]


def remember() -> None:
    """Keep a copy of the settings for when the other page has been shown. Call this after the sidebar is drawn."""
    ss = st.session_state
    ss["_world_saved"] = {k: v for k, v in ss.items() if k.startswith("w_")}


def world_from_state(base: World) -> World:
    ss = st.session_state
    star = Star(base.star.name, ss["w_star_mass"], ss["w_star_lum"], ss["w_star_rad"])
    planets = []
    for p in base.planets:
        n = p.name
        planets.append(replace(
            p, radius_km=ss[f"w_{n}_radius"], mass_kg=ss[f"w_{n}_mass"] * M_EARTH, rotation_hours=ss[f"w_{n}_rot"],
            rotation_is_sidereal=ss[f"w_{n}_sid"] == "sidereal", a_au=ss[f"w_{n}_a"], ecc=ss[f"w_{n}_ecc"], obliquity_deg=ss[f"w_{n}_obl"],
            mean_anomaly_deg=ss[f"w_{n}_ph"], star_mass_kg=ss["w_star_mass"] * M_SUN, star_luminosity_lsun=ss["w_star_lum"],
            star_radius_rsun=ss["w_star_rad"]))
    moons = tuple(
        (host, tuple(Moon(m.name, ss[f"w_{m.name}_dia"], ss[f"w_{m.name}_rho"], float(ss[f"w_{m.name}_a"]), ss[f"w_{m.name}_e"],
                          ss[f"w_{m.name}_inc"], ss[f"w_{m.name}_M"], ss[f"w_{m.name}_Om"], ss[f"w_{m.name}_w"]) for m in ms))
        for host, ms in base.moons)
    return World(star, tuple(planets), moons)


# ------------------------------------------------------------------ widgets

def _reset(keys: list[str], base: World) -> None:
    d = defaults(base)
    st.session_state.update({k: d[k] for k in keys if k in d})


def reset_star(base: World) -> None:
    _reset(["w_star_mass", "w_star_lum", "w_star_rad"], base)


def reset_planet(name: str, base: World) -> None:
    _reset([f"w_{name}_{k}" for k in PLANET_KEYS], base)


def apply_design(host: str, name: str | None = None) -> None:
    """Set the moons' settings to one of the designs in presets.py."""
    name = name or st.session_state["w_design"]
    _, liw, nin = PRESETS[name]
    for m in (liw, nin):
        st.session_state.update({f"w_{m.name}_dia": float(m.diameter_km), f"w_{m.name}_rho": float(m.density_gcc), f"w_{m.name}_a": float(m.a_km),
                                 f"w_{m.name}_e": float(m.ecc), f"w_{m.name}_inc": float(m.inc_deg), f"w_{m.name}_M": float(m.mean_anomaly_deg),
                                 f"w_{m.name}_Om": float(m.asc_node_deg), f"w_{m.name}_w": float(m.arg_peri_deg)})


def star_editor(base: World) -> None:
    st.button("Reset the star to the data file", on_click=reset_star, args=(base,), key="reset_star")
    st.number_input("Stellar mass (solar masses)", 0.2, 3.0, step=0.001, format="%.4f", key="w_star_mass",
                    help="Aureth is G-type. The default is the mass the original canon implied (a 374-day year at 0.87 AU), kept fixed.")
    st.number_input("Stellar luminosity (solar)", 0.01, 5.0, step=0.01, format="%.3f", key="w_star_lum",
                    help="A young 1-solar-mass star is about 0.68 to 0.76 (rough).")
    st.number_input("Stellar radius (solar)", 0.1, 3.0, step=0.01, format="%.3f", key="w_star_rad",
                    help="A young 1-solar-mass star is about 0.87 to 0.9. Liwaya can only fully cover the Sun if this is under ~0.9.")


# Each setting is drawn by exactly one function, so both pages create the identical widget. (If two pages drew the same key
# with different labels or limits, Streamlit would treat them as different widgets and forget the value on the way across.)
def _radius(n): st.number_input("Radius (km)", 1000.0, 20000.0, step=50.0, key=f"w_{n}_radius")
def _mass(n): st.number_input("Mass (Earth masses)", 0.01, 30.0, step=0.01, format="%.4f", key=f"w_{n}_mass")
def _rot(n): st.number_input("Rotation period (hours)", 2.0, 200.0, step=0.5, key=f"w_{n}_rot")
def _sid(n):
    st.radio("The rotation period is a", ["solar", "sidereal"], key=f"w_{n}_sid", horizontal=True,
             help="Sidereal: relative to the stars. Solar: noon to noon.")
def _dist(n): st.number_input("Distance from the star (AU)", 0.1, 5.0, step=0.01, format="%.4f", key=f"w_{n}_a")
def _ecc(n): st.slider("Orbital eccentricity", 0.0, 0.5, step=0.005, key=f"w_{n}_ecc")
def _tilt(n): st.number_input("Axial tilt (deg)", 0.0, 90.0, step=1.0, key=f"w_{n}_obl")
def _phase(n):
    st.slider("Position at the founding moment (deg of orbit)", 0.0, 360.0, step=1.0, key=f"w_{n}_ph",
              help="0 puts the planet at its closest point to Aureth on the shared starting line.")


def planet_editor(p, base: World) -> None:
    n = p.name
    st.button(f"Reset {n} to the data file", on_click=reset_planet, args=(n, base), key=f"reset_{n}")
    if p.placeholder_note:
        st.caption(f"Not settled: {p.placeholder_note}.")
    _radius(n); _mass(n); _rot(n); _sid(n); _dist(n); _ecc(n); _tilt(n); _phase(n)


def orbit_editor(p, base: World) -> None:
    """Just the orbit and mass of a planet (what the Aureth system page needs)."""
    n = p.name
    st.button(f"Reset {n} to the data file", on_click=reset_planet, args=(n, base), key=f"sysreset_{n}")
    if p.placeholder_note:
        st.caption(f"Not settled: {p.placeholder_note}.")
    _dist(n); _ecc(n); _mass(n); _phase(n)


def moon_editors(host: str, base: World) -> None:
    st.selectbox("Moon design", list(PRESETS), key="w_design", on_change=apply_design, args=(host,),
                 help="Design B is the settled design. Design C is an experiment: Liwaya on an eccentric orbit, so it swells and shrinks.")
    st.caption(PRESETS[st.session_state["w_design"]][0])
    st.button("Reset the moons to this design", on_click=apply_design, args=(host,), key=f"reset_moons_{host}")
    for h, ms in base.moons:
        if h != host:
            continue
        for m in ms:
            k = m.name
            with st.expander(k, expanded=False):
                st.number_input("Diameter (km)", 200.0, 8000.0, step=50.0, key=f"w_{k}_dia")
                st.number_input("Density (g/cm³)", 1.0, 6.0, step=0.1, key=f"w_{k}_rho")
                st.number_input("Starting semi-major axis (km)", 30_000.0, 700_000.0, step=100.0, key=f"w_{k}_a",
                                help="Osculating value at the start. The simulated average distance differs by about 1%; see the Overview.")
                st.slider("Eccentricity", 0.0, 0.3, step=0.005, key=f"w_{k}_e")
                st.slider("Inclination to the ecliptic (deg)", 0.0, 30.0, step=0.1, key=f"w_{k}_inc")
                st.slider("Starting orbital phase (deg)", 0.0, 360.0, step=1.0, key=f"w_{k}_M")
                st.slider("Longitude of ascending node (deg)", 0.0, 360.0, step=1.0, key=f"w_{k}_Om")
                st.slider("Argument of periapsis (deg)", 0.0, 360.0, step=1.0, key=f"w_{k}_w")
