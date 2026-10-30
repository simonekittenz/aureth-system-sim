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

PLANET_KEYS = ("radius", "mass", "rot", "sid", "a", "ecc", "obl", "ph", "axis")
MOON_FIELDS = ("dia", "rho", "a", "e", "inc", "M", "Om", "w")


def _planet_defaults(base: World) -> dict:
    out = {}
    for p in base.planets:
        n = p.name
        out.update({f"w_{n}_radius": float(p.radius_km), f"w_{n}_mass": float(p.mass_earth), f"w_{n}_rot": float(p.rotation_hours),
                    f"w_{n}_sid": "sidereal" if p.rotation_is_sidereal else "solar", f"w_{n}_a": float(p.a_au), f"w_{n}_ecc": float(p.ecc),
                    f"w_{n}_obl": float(p.obliquity_deg), f"w_{n}_ph": float(p.mean_anomaly_deg),
                    f"w_{n}_axis": float(p.axis_azimuth_deg)})
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
    return {**_planet_defaults(base), **_moon_defaults(base), "w_design": DEFAULT_PRESET, "w_start_seed": int(base.start_seed), "w_star_mass": float(p0.star_mass_msun), "w_star_lum": float(p0.star_luminosity_lsun),
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
            mean_anomaly_deg=ss[f"w_{n}_ph"], axis_azimuth_deg=ss[f"w_{n}_axis"], star_mass_kg=ss["w_star_mass"] * M_SUN, star_luminosity_lsun=ss["w_star_lum"],
            star_radius_rsun=ss["w_star_rad"]))
    moons = tuple(
        (host, tuple(Moon(m.name, ss[f"w_{m.name}_dia"], ss[f"w_{m.name}_rho"], float(ss[f"w_{m.name}_a"]), ss[f"w_{m.name}_e"],
                          ss[f"w_{m.name}_inc"], ss[f"w_{m.name}_M"], ss[f"w_{m.name}_Om"], ss[f"w_{m.name}_w"]) for m in ms))
        for host, ms in base.moons)
    return World(star, tuple(planets), moons, base.start_seed, base.start_distances)


# ------------------------------------------------------------------ widgets

def _reset(keys: list[str], base: World) -> None:
    d = defaults(base)
    st.session_state.update({k: d[k] for k in keys if k in d})


def reset_star(base: World) -> None:
    _reset(["w_star_mass", "w_star_lum", "w_star_rad"], base)


def reset_planet(name: str, base: World) -> None:
    _reset([f"w_{name}_{k}" for k in PLANET_KEYS], base)


def start_matches_saved(base: World) -> bool:
    ss = st.session_state
    d = defaults(base)
    return all(abs(float(ss.get(k, d[k])) - float(d[k])) < 1e-9 for k in _start_keys(base))


def apply_design(host: str, base: World, name: str | None = None) -> None:
    """Set the moons' physical settings to one of the designs in presets.py. Where they start (day 1) comes from the data file, not the design;
    the starting distances are the retuned ones for that start (or a retune is asked for)."""
    ss = st.session_state
    name = name or ss["w_design"]
    _, liw, nin = PRESETS[name]
    start = {m.name: m for h, ms in base.moons if h == host for m in ms}
    saved = {(d, mn): km for d, mn, km in base.start_distances}
    use_saved = start_matches_saved(base) and all((name, m.name) in saved for m in (liw, nin))
    for m in (liw, nin):
        s0 = start.get(m.name, m)
        ss.update({f"w_{m.name}_dia": float(m.diameter_km), f"w_{m.name}_rho": float(m.density_gcc),
                   f"w_{m.name}_a": float(saved[(name, m.name)]) if use_saved else float(m.a_km),
                   f"w_{m.name}_e": float(m.ecc), f"w_{m.name}_inc": float(m.inc_deg), f"w_{m.name}_M": float(s0.mean_anomaly_deg),
                   f"w_{m.name}_Om": float(s0.asc_node_deg), f"w_{m.name}_w": float(s0.arg_peri_deg)})
    if not use_saved:
        ss["_needs_retune"] = host


def ask_retune(host: str) -> None:
    st.session_state["_needs_retune"] = host


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
    st.slider("Position on day 1 (deg of orbit)", 0.0, 360.0, step=1.0, key=f"w_{n}_ph",
              help="0 puts the planet at its closest point to Aureth on the shared starting line.")


def _axis(n):
    st.slider("Which way the axis leans on day 1 (deg)", 0.0, 360.0, step=1.0, key=f"w_{n}_axis",
              help="This sets the season on day 1. 0 is a winter solstice at the planet's closest approach to Aureth; 90 and 270 are equinoxes; 180 is a summer solstice at closest approach.")


def planet_editor(p, base: World) -> None:
    n = p.name
    st.button(f"Reset {n} to the data file", on_click=reset_planet, args=(n, base), key=f"reset_{n}")
    if p.placeholder_note:
        st.caption(f"Not settled: {p.placeholder_note}.")
    _radius(n); _mass(n); _rot(n); _sid(n); _dist(n); _ecc(n); _tilt(n); _phase(n); _axis(n)


def orbit_editor(p, base: World) -> None:
    """Just the orbit and mass of a planet (what the Aureth system page needs)."""
    n = p.name
    st.button(f"Reset {n} to the data file", on_click=reset_planet, args=(n, base), key=f"sysreset_{n}")
    if p.placeholder_note:
        st.caption(f"Not settled: {p.placeholder_note}.")
    _dist(n); _ecc(n); _mass(n); _phase(n)


def moon_editors(host: str, base: World) -> None:
    st.selectbox("Moon design", list(PRESETS), key="w_design", on_change=apply_design, args=(host, base),
                 help="Design B is the settled design. Design C is an experiment: Liwaya on an eccentric orbit, so it swells and shrinks.")
    st.caption(PRESETS[st.session_state["w_design"]][0])
    st.button("Reset the moons to this design", on_click=apply_design, args=(host, base), key=f"reset_moons_{host}")
    st.button("Retune the moons' distances", on_click=ask_retune, args=(host,), key=f"retune_{host}",
              help="Moves each moon's starting distance a little so its phase cycle comes back to 38.5 and 9.70 days from where it starts on day 1. About 15 seconds.")
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


# ------------------------------------------------------------------ year 1, day 1
def _start_keys(base: World) -> list[str]:
    keys = [f"w_{p.name}_{k}" for p in base.planets for k in ("ph", "axis")]
    keys += [f"w_{m.name}_{k}" for _h, ms in base.moons for m in ms for k in ("M", "Om", "w")]
    return keys


def roll_start(base: World, seed: int | None = None) -> None:
    """Draw a natural-looking day 1 from a seed (a new random one if none is given) and put it into the sidebar settings. The moons then get retuned."""
    import random

    from .start import random_start
    ss = st.session_state
    if seed is None:
        seed = random.randint(0, 99999)
    ss["w_start_seed"] = int(seed)
    ss.update(random_start(int(seed), base))
    ss["_needs_retune"] = "Bayurinda"


def saved_start(base: World) -> None:
    ss = st.session_state
    d = defaults(base)
    ss.update({k: d[k] for k in _start_keys(base)})
    ss["w_start_seed"] = int(base.start_seed)
    apply_design("Bayurinda", base)


def save_start_cb() -> None:
    st.session_state["_start_action"] = "save"


@st.cache_data(show_spinner=False)
def _retuned(system):
    from .start import retune_distances
    return retune_distances(system)


def process_start_actions(base: World) -> None:
    """Do the slow things a click asked for (retuning, saving) before the sidebar is drawn. Call this right after ensure_state."""
    ss = st.session_state
    host = ss.pop("_needs_retune", None)
    if host and base.moons_of(host):
        with st.spinner("Retuning the moons' starting distances so their phase cycles stay at 38.5 and 9.70 days (about 15 seconds)..."):
            sysm = world_from_state(base).system(host)
            for m, km in zip(sysm.moons, _retuned(sysm)):
                ss[f"w_{m.name}_a"] = round(km)
    if ss.pop("_start_action", None) == "save":
        from .start import save_start
        with st.spinner("Retuning the moons for each design and saving (about 30 seconds)..."):
            w_now = world_from_state(base)
            dists = {}
            for design, (_blurb, liw, nin) in PRESETS.items():
                ms = tuple(replace(m, mean_anomaly_deg=ss[f"w_{m.name}_M"], asc_node_deg=ss[f"w_{m.name}_Om"], arg_peri_deg=ss[f"w_{m.name}_w"]) for m in (liw, nin))
                s_d = w_now.system("Bayurinda")
                s_d = type(s_d)(s_d.planet, ms, s_d.neighbors)
                dists[design] = _retuned(s_d)
            save_start(int(ss["w_start_seed"]), {k: ss[k] for k in _start_keys(base)}, base, dists)
        ss["_start_saved"] = True


def start_editor(base: World) -> None:
    """The Day 1 box: roll a new natural-looking start from a seed, go back to the saved one, or save the current numbers as the default."""
    ss = st.session_state
    st.caption("Where every planet and moon is on year 1, day 1, and which way each axis leans (the season). A seed always gives the same start. "
               "Rolling avoids contrived ones: planets lined up, moons on top of each other, both moons full or both new. A roll retunes the moons' distances "
               "(about 15 seconds). You can also set each number by hand in the planet and moon settings.")
    st.number_input("Day 1 seed", 0, 999_999, step=1, key="w_start_seed", on_change=lambda: roll_start(base, int(st.session_state["w_start_seed"])),
                    help="Type a number and press Enter to use that start.")
    c1, c2 = st.columns(2)
    c1.button("New random day 1", on_click=roll_start, args=(base,), key="start_roll")
    c2.button("Back to saved", on_click=saved_start, args=(base,), key="start_back", help="Return to the day 1 written in data/aureth.yaml.")
    st.button("Save this day 1 as the default", on_click=save_start_cb, key="start_save",
              help="Writes the current day 1 into data/aureth.yaml (the system_start block), with the moons' distances retuned for each design. About 30 seconds.")
    if ss.pop("_start_saved", False):
        st.success("Saved to data/aureth.yaml.")
