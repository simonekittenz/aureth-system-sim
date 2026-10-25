from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from bajurinda import plots
from bajurinda.constants import YEAR_DAYS
from bajurinda.eclipses import SITE_ANNULAR, SITE_PARTIAL, SITE_TOTAL, eclipse_events, site_eclipse_types, site_rates, summarise
from bajurinda.cycles import axial_precession, moon_cycles
from bajurinda.sky import spin_axis
from bajurinda.starfield import ecl_vec
from bajurinda.landmarks import (
    moon_declination_envelope, rise_offset_deg, return_cycles, star_alignment_life, sun_markers,
)
from bajurinda.nightsky import compute_sky, day_crossings, find_sight, readout, sky_figure
from bajurinda.starfield import load_faint_settings, load_sky, save_faint_seed
from bajurinda.constellations import (clone, figure_stars, load_constellations, missing_stars, new_set, save_constellations)
from bajurinda.eras import era_bands, era_summary, eras_data
from bajurinda.lunar_calendar import build_calendar
from bajurinda.dynamics import integrate, mean_rates, phase_ensemble, stability_run
from bajurinda.model import System, planet_warnings, system_report
from bajurinda.presets import DEFAULT_PRESET, PRESETS
from bajurinda.world import load_world
from bajurinda.worldstate import ensure_state, moon_editors, planet_editor, remember, star_editor, world_from_state
from bajurinda.sky import (
    alignment_events, altitude_deg, crossings, tide_series,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = load_world()
ss = st.session_state

# Streamlit forgets the settings of widgets that are not on screen, so leaving this page and coming back would reset
# every box in the sidebar to its minimum. Keep a copy of the settings and put them back before the widgets are drawn.
for _k, _v in ss.get("_moons_saved", {}).items():
    if _k not in ss:
        ss[_k] = _v
for _k, _v in {"v_pov": "Bayurinda", "v_window": 240, "v_step": 3, "v_lat": 10, "v_lon": 0, "v_unit": "Earth days", "v_sky_year": 1, "v_sky_day": 1,
               "v_sky_clock": 0.0, "v_sky_time": "local", "v_sky_view": "Panorama", "v_sky_fov": 180, "v_sky_height": 860, "v_sky_look": "Liwaya", "v_sky_scale": 12,
               "v_sky_pick": False, "v_sky_pv_on": False, "v_sky_pv_lon": 100.0, "v_sky_pv_lat": 20.0, "v_sky_pv_mag": 1.0,
               "v_sky_always": False, "v_sky_move": 10, "v_con_show": None, "v_con_labels": True, "v_con_edit": None, "v_con_fig": None,
               "v_sky_named": True, "v_sky_mw": True, "v_sky_faint": True, "v_sky_seed": load_faint_settings()["seed"], "v_sky_step": "30 minutes", "v_sky_span": "1 day", "v_sky_ms": 140}.items():
    if _k not in ss:
        ss[_k] = _v
if ss["v_unit"] == "Bajurindan days":                 # an older name for the same choice
    ss["v_unit"] = "Local days"
ensure_state(BASE)

with st.sidebar:
    st.title("Planet view")
    pov = st.selectbox("Point of view", BASE.names, key="v_pov",
                       help="Whose sky, seasons and calendar to look at. The star and the planets are the same ones as on the Aureth system page.")
    base_p = BASE.planet(pov)
    has_moons_base = bool(BASE.moons_of(pov))
    st.caption(PRESETS[DEFAULT_PRESET][0] if has_moons_base else f"{pov} has no moons in the world files yet.")

    with st.expander("Star (Aureth)", expanded=False):
        star_editor(BASE)
    with st.expander(f"{pov}: size, spin and orbit", expanded=False):
        planet_editor(base_p, BASE)
    if has_moons_base:
        moon_editors(pov, BASE)

    st.subheader("View")
    if has_moons_base:
        window = st.slider("Simulated span (Earth days)", 30, 1500, step=10, key="v_window")
        step_h = st.select_slider("Sampling step (hours)", [1, 2, 3, 4, 6, 12], key="v_step")
    else:
        window, step_h = ss["v_window"], ss["v_step"]
    lat = st.slider("Observer latitude (deg)", -90, 90, key="v_lat")
    lon = st.slider("Observer longitude (deg)", -180, 180, key="v_lon")
    unit = st.radio("Time unit for plots", ["Earth days", "Local days"], horizontal=True, key="v_unit",
                    format_func=lambda u: "Earth days" if u == "Earth days" else f"{pov} solar days")

    st.subheader("Faint stars")
    st.caption("The dim background stars are generated from a seed: the same number always gives the same sky.")

    def shuffle_seed():
        import random
        st.session_state["v_sky_seed"] = random.randint(0, 99999)

    def saved_seed():
        st.session_state["v_sky_seed"] = load_faint_settings()["seed"]

    def save_seed():
        save_faint_seed(int(st.session_state["v_sky_seed"]))
        st.session_state["_seed_saved"] = int(st.session_state["v_sky_seed"])

    st.number_input("Seed", 0, 999_999, step=1, key="v_sky_seed")
    cs1, cs2 = st.columns(2)
    cs1.button("New random seed", on_click=shuffle_seed, key="seed_shuffle")
    cs2.button("Back to saved", on_click=saved_seed, key="seed_back", help="Return to the seed written in data/aureth.yaml.")
    st.button("Save this seed as the default", on_click=save_seed, key="seed_save", help="Writes the seed into data/aureth.yaml.")
    saved_now = load_faint_settings()["seed"]
    st.caption(f"Saved seed: {saved_now}" + ("  (just saved)" if st.session_state.get("_seed_saved") == saved_now else "")
               + ("" if int(st.session_state["v_sky_seed"]) == saved_now else "  - the view is using a different one"))

ss["_moons_saved"] = {k: v for k, v in ss.items() if k.startswith("v_")}
remember()

world = world_from_state(BASE)
system = world.system(pov)
planet = system.planet
has_moons = bool(system.moons)
unit_days = planet.local_day_days if unit == "Local days" else 1.0
unit_label = f"{pov} solar days" if unit == "Local days" else "Earth days"
day_h = planet.solar_day_hours


@st.cache_data(show_spinner="Integrating orbits with REBOUND...")
def get_track(system: System, window_days: int, step_hours: int):
    times = np.arange(step_hours / 24, window_days + 1e-9, step_hours / 24)
    times = np.insert(times, 0, 0.0)
    return integrate(system, times)


@st.cache_data(show_spinner="Measuring mean orbital periods...")
def get_rates(system: System):
    return mean_rates(system)


if has_moons:
    track = get_track(system, window, step_h)
    rates = get_rates(system)
    report_system = System(planet, tuple(replace(m, a_km=d) for m, d in zip(system.moons, rates["distance"])))
    report = system_report(report_system)

TAB_NAMES = (["Overview", "Orbits", "Phases & sky", "Tides", "Alignments", "Calendar", "Eras", "Eclipses", "Landmarks", "Night sky", "Stability", "Method & caveats"]
             if has_moons else ["Overview", "Landmarks", "Night sky", "Method & caveats"])
T = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))

# ------------------------------------------------------------------ Overview
with T["Overview"]:
    st.header(f"{planet.name}: two-moon system" if has_moons else f"{planet.name}")
    if has_moons:
        for sev, msg in planet_warnings(report_system):
            {"error": st.error, "warning": st.warning, "info": st.info}[sev](msg)
    elif planet.placeholder_note:
        st.warning(f"Not settled for {planet.name}: {planet.placeholder_note}. They are stand-in values so the sky can be worked out; "
                   "change them in the sidebar or in data/aureth.yaml.")
    canon_bay = planet.name == "Bayurinda"
    origin = lambda canon_text, other="from the data file or the sidebar": canon_text if canon_bay else other

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Planet and calendar")
        rows = [
            ("Radius", f"{planet.radius_km:,.0f} km", origin("canon (diameter 12,000 km)")),
            ("Surface gravity", f"{planet.gravity_g:.3f} g", origin("canon 0.9 g", "derived")),
            ("Mass", f"{planet.mass_earth:.4f} Earth", origin("derived from g and radius")),
            ("Mean density", f"{planet.density_gcc:.2f} g/cm³", "derived"),
            ("Escape velocity", f"{planet.escape_velocity_kms:.2f} km/s", "derived"),
            ("Rotation (sidereal)", f"{planet.sidereal_rotation_hours:.4f} h", origin("derived from the 19 h solar day", "derived from the solar day")),
            ("Solar day", f"{planet.solar_day_hours:.4f} h", origin("canon 19 h")),
            ("Year", f"{planet.year_days:.3f} Earth days", "derived (Kepler)"),
            ("Year in sidereal rotations", f"{planet.sidereal_rotations_per_year:.2f}", "derived"),
            ("Year in solar days", f"{planet.solar_days_per_year:.2f}", "derived"),
            ("Axial tilt", f"{planet.obliquity_deg:g}°", origin("canon")),
            ("Implied stellar mass", f"{planet.star_mass_msun:.4f} Msun", "derived"),
            ("Stellar flux", f"{planet.irradiance_earth:.2f} x Earth", "derived from the star's luminosity"),
            ("Sun's apparent size", f"{planet.sun_angular_diameter_deg():.3f}°", "derived"),
            ("Star surface temperature", f"{planet.star_teff_k:,.0f} K", "derived from luminosity and radius"),
            ("Hill sphere radius", f"{planet.hill_radius_km:,.0f} km", "derived"),
            ("Synchronous orbit (this spin)", f"{planet.synchronous_orbit_km:,.0f} km", "derived"),
            ("Laplace-plane radius (approx.)", f"{planet.laplace_radius_km:,.0f} km", "derived, rough J2"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Property", "Value", "Origin"]), hide_index=True, width="stretch")
    with c2:
        if has_moons:
            st.subheader("Moons")
            a, b = (dict(report["moons"][m.name]) for m in system.moons)
            ld = planet.local_day_days
            for r, k in ((a, 0), (b, 1)):
                r.update(sidereal_days=rates["sidereal"][k], sidereal_local_days=rates["sidereal"][k] / ld,
                         synodic_days=rates["synodic"][k], synodic_local_days=rates["synodic"][k] / ld)
            spec = [
                ("Mass (Earth's Moon = 1)", "mass_luna", "{:.3f}"),
                ("Mean distance, simulated (km)", "a_km", "{:,.0f}"),
                ("Distance (planet radii)", "a_planet_radii", "{:.1f}"),
                ("Distance / planet Hill radius", "a_hill_frac", "{:.3f}"),
                ("Use of single-moon stability limit", "stability_fraction", "{:.0%}"),
                ("Sidereal period, simulated mean (Earth days)", "sidereal_days", "{:.3f}"),
                ("Sidereal period, simulated mean (local days)", "sidereal_local_days", "{:.2f}"),
                ("Synodic / phase period, simulated mean (Earth days)", "synodic_days", "{:.3f}"),
                ("Synodic / phase period, simulated mean (local days)", "synodic_local_days", "{:.2f}"),
                ("Apparent diameter, mean (deg)", "ang_diam_mean_deg", "{:.3f}"),
                ("Apparent diameter, closest (deg)", "ang_diam_peri_deg", "{:.3f}"),
                ("Apparent diameter, farthest (deg)", "ang_diam_apo_deg", "{:.3f}"),
                ("Peak tide height (m, equilibrium)", "tide_amp_m", "{:.3f}"),
                ("Tide strength vs Earth's Moon", "tide_vs_luna", "{:.2f}"),
                ("  at periapsis", "tide_vs_luna_peri", "{:.2f}"),
                ("  at apoapsis", "tide_vs_luna_apo", "{:.2f}"),
                ("Fluid Roche limit (km)", "roche_fluid_km", "{:,.0f}"),
                ("Moon / planet mass", "mass_ratio", "{:.2%}"),
            ]
            st.dataframe(pd.DataFrame(
                [(label, fmt.format(a[k]), fmt.format(b[k])) for label, k, fmt in spec],
                columns=["", system.moons[0].name, system.moons[1].name]), hide_index=True, width="stretch", height=670)
        else:
            st.subheader("Neighbours")
            from bajurinda.aureth import period_days, star_of, synodic_days, body_from_planet
            star_ = star_of(planet)
            me = body_from_planet(planet)
            rows_ = []
            for nb in system.neighbors:
                nb_b = body_from_planet(nb)
                near, far = abs(nb.a_au - planet.a_au), nb.a_au + planet.a_au
                syn_ = synodic_days(period_days(star_, me), period_days(star_, nb_b))
                rows_.append({"Planet": nb.name, "Distance from the star (AU)": round(nb.a_au, 3), "Year (Earth days)": round(period_days(star_, nb_b), 1),
                              "Closest to here (AU)": round(near, 2), "Farthest (AU)": round(far, 2),
                              "Same line-up again every (Earth years)": round(syn_ / 365.25, 2)})
            st.dataframe(pd.DataFrame(rows_), hide_index=True, width="stretch")
            st.caption("Closest and farthest are rough (circular orbits). There are no moons for this planet in the world files, so the "
                       "moon tabs are hidden. Pick Bayurinda as the point of view to see them.")

    if has_moons:
        st.subheader("Recurring patterns")
        S, P = rates["synodic"], rates["sidereal"]
        names = [m.name for m in system.moons]
        lineup = 1 / abs(1 / P[0] - 1 / P[1])
        pr = [
            (f"{names[0]} phase cycle (synodic)", S[0]),
            (f"{names[1]} phase cycle (synodic)", S[1]),
            (f"{names[0]} and {names[1]} line up again (also the phase-beat)", lineup),
            (f"{names[0]}-Sun spring tide recurrence", S[0] / 2),
            (f"{names[1]}-Sun spring tide recurrence", S[1] / 2),
            ("Moon-moon tidal reinforcement recurrence", lineup / 2),
        ]
        longi = int(np.argmax(S)); shorti = 1 - longi
        repeat = None
        q = round(S[longi] / S[shorti]) if np.all(np.isfinite(S)) and S[shorti] > 0 else 0
        if q >= 2 and abs(S[longi] / S[shorti] - q) < 0.25:
            repeat = 1 / abs(1 / S[shorti] - q / S[longi])
            pr.append(("Both phase patterns repeat together", repeat))
        st.dataframe(pd.DataFrame(
            [(n, f"{v:.2f}", f"{v / planet.local_day_days:.2f}") for n, v in pr],
            columns=["Cycle", "Earth days", "Bajurindan days"]), hide_index=True, width="stretch")
        if repeat:
            st.caption(f"Phase repeat: {repeat / S[longi]:.1f} {names[longi]} cycles = {repeat / S[shorti]:.1f} {names[shorti]} cycles "
                       f"= {repeat / planet.year_days:.2f} Bayurindan years. All periods here are measured from the simulation "
                       "(they include the Sun's perturbation), so they differ slightly from a plain Kepler calculation.")
        c3, c4, c5 = st.columns(3)
        c3.metric(f"Period ratio ({report['outer']} / {report['inner']})", f"{report['period_ratio']:.3f}")
        c4.metric("Spacing (mutual Hill radii)", f"{report['mutual_hill_sep']:.1f}", help="Below ~3.5 is unstable for circular coplanar orbits.")
        c5.metric("Solar tide vs Earth's Moon", f"{report['sun_tide_vs_luna']:.2f}")
        if report["resonances"]:
            st.caption("Nearest mean-motion resonances: " + ", ".join(
                f"{r['resonance']} ({r['offset_pct']:+.2f}%)" for r in report["resonances"][:3]))


# ------------------------------------------------------------------ Orbits
if has_moons:
    with T["Orbits"]:
        st.subheader("Top-down view, planet-centred")
        st.caption("Viewed from above the ecliptic. The dotted circle is the approximate single-moon Hill stability limit. "
                   "Press Play, or drag the slider.")
        st.plotly_chart(plots.orbit_figure(system, track), width="stretch")

# ------------------------------------------------------------------ Phases & sky
if has_moons:
    with T["Phases & sky"]:
        st.subheader("Sky at a chosen moment")
        sel_day = st.slider("Time within the simulated span (Earth days)", 0.0, float(window), 0.0, step=step_h / 24, key="sel_day")
        i_sel = int(np.argmin(np.abs(track.t - sel_day)))
        st.plotly_chart(plots.sky_discs_figure(system, track, i_sel), width="stretch")
        st.caption("Disc widths are true apparent sizes. Lit side is drawn as it would look from the northern hemisphere "
                   "(waxing = lit on the right).")

        alt_vecs = {"Sun": track.star_pos, "Liwaya": track.moon_pos[:, 0], "Nineya": track.moon_pos[:, 1]}
        alts = {k: altitude_deg(planet, track, v, lat, lon) for k, v in alt_vecs.items()}
        hi = min(len(track.t), i_sel + int(round(3 * 24 / step_h)))
        st.subheader(f"Altitude over the next 3 Earth days at {lat}°, {lon}°")
        st.plotly_chart(plots.altitude_figure(track, alts, unit_days, unit_label, i_sel, hi), width="stretch")

        rs = []
        for name in ("Liwaya", "Nineya"):
            up, down = crossings(track.t, alts[name])
            rs.append((name, up[(up >= sel_day) & (up <= sel_day + 3)], down[(down >= sel_day) & (down <= sel_day + 3)]))
        cols = st.columns(2)
        for c, (name, up, down) in zip(cols, rs):
            c.markdown(f"**{name}**  \nrises (day): {', '.join(f'{x:.2f}' for x in up) or 'none'}  \n"
                       f"sets (day): {', '.join(f'{x:.2f}' for x in down) or 'none'}")

        st.subheader("Phase timeline")
        st.plotly_chart(plots.phase_figure(system, track, unit_days, unit_label), width="stretch")

# ------------------------------------------------------------------ Tides
if has_moons:
    with T["Tides"]:
        st.subheader("Equilibrium tide")
        st.caption("Gravitational forcing only: the height of the water surface if the ocean could respond instantly "
                   "and had no continents. Real coastal tides depend on basin shape, depth and resonance, and can be "
                   "many times larger or smaller.")
        ts = tide_series(system, track, lat, lon)
        st.plotly_chart(plots.tide_figure(ts, track, unit_days, unit_label), width="stretch")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strongest tide in span (m)", f"{ts.global_peak.max():.2f}")
        m2.metric("Weakest 'maximum' tide in span (m)", f"{ts.global_peak.min():.2f}")
        m3.metric("Largest daily range at site (m)", f"{ts.site_range.max():.2f}")
        m4.metric("Smallest daily range at site (m)", f"{ts.site_range.min():.2f}")
        st.caption("Reference: Earth's Moon alone raises roughly 0.36 m of equilibrium bulge on Earth; Moon plus Sun at spring tide, ~0.5 m.")

# ------------------------------------------------------------------ Alignments
if has_moons:
    with T["Alignments"]:
        st.subheader("Calendar of notable alignments")
        window_deg = st.slider("'Double full / new' tolerance (deg)", 5, 45, 15)
        ev = alignment_events(track, window_deg)
        if ev:
            df = pd.DataFrame(ev)
            df["Bajurindan day"] = (df["day"] / planet.local_day_days).round(2)
            df["day"] = df["day"].round(2)
            df = df.rename(columns={"day": "Earth day", "event": "Event"})[["Earth day", "Bajurindan day", "Event"]]
            kinds = sorted({e["event"] for e in ev})
            pick = st.multiselect("Show", kinds, default=[k for k in kinds if "Double" in k or "conjunction" in k or "opposition" in k])
            st.dataframe(df[df["Event"].isin(pick)], hide_index=True, width="stretch")
        st.caption("Extend the simulated span in the sidebar (up to 1,500 days) to see rarer alignments.")

# ------------------------------------------------------------------ Calendar
if has_moons:
    with T["Calendar"]:
        st.subheader("Calendar across the repeating cycle")
        st.write("Months follow the larger moon's phase cycle and weeks follow the smaller moon's, each counted in whole "
                 "Bajurindan solar days from the founding moment (day 1). A *double* is a Liwaya phase with a Nineya phase "
                 "of the same kind falling within the tolerance. Names and structure are placeholders.")
        names = tuple(m.name for m in system.moons)
        c1, c2, c3 = st.columns(3)
        start_kind = c1.radio("Months and weeks begin at", ["full", "new"], format_func=lambda k: f"{k} moon", horizontal=True)
        tol = c2.slider("Tolerance for a 'double' (Bajurindan days)", 0.1, 2.0, 0.5, step=0.1)
        n_cycles = c3.select_slider("Cycles to show", [1, 2, 3], value=1)

        cycle_days = repeat if repeat else 4 * planet.year_days
        span = cycle_days * n_cycles

        @st.cache_data(show_spinner="Simulating the calendar span...")
        def get_cycle_track(system: System, span_days: float):
            return integrate(system, np.insert(np.arange(0.25, span_days + 120, 0.25), 0, 0.0))

        cal = build_calendar(system, get_cycle_track(system, span), start_kind, tol, span_days=span)
        ml = [m["length_days"] for m in cal.months]
        wl = [w["length_days"] for w in cal.weeks]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Cycle length", f"{cycle_days / planet.local_day_days:,.0f} days", f"{cycle_days / planet.year_days:.2f} years")
        k2.metric("Months", f"{len(ml)}", f"{ml.count(min(ml))} x {min(ml)} days, {ml.count(max(ml))} x {max(ml)} days" if ml else "")
        k3.metric("Weeks", f"{len(wl)}", f"{wl.count(min(wl))} x {min(wl)} days, {wl.count(max(wl))} x {max(wl)} days" if wl else "")
        wraps = (n_cycles == 1 and len(cal.runs) > 1 and cal.runs[0]["kind"] == cal.runs[-1]["kind"]
                 and cal.runs[0]["first_month"] == 1 and cal.runs[-1]["last_month"] == len(cal.months))
        k4.metric("Double runs", f"{len(cal.runs) - (1 if wraps else 0)}", "first and last runs join across the cycle boundary" if wraps else None, delta_color="off")
        st.plotly_chart(plots.calendar_figure(cal, names), width="stretch")
        st.caption("Gold bands: runs of consecutive double-full months. Purple bands: double-new. A run that straddles the end "
                   "of one cycle continues at the start of the next; show 2 cycles to see it whole.")

        if cal.runs:
            st.markdown("**Double runs**")
            st.dataframe(pd.DataFrame([{
                "Kind": f"double {r['kind']}", "Months": f"{r['first_month']}-{r['last_month']}", "Length (months)": r["count"],
                "Days": f"{r['start_day']}-{r['end_day']}", "Best month": r["best_month"], "Best day": r["best_day"],
                "Best offset (days)": round(r["best_offset"], 2)} for r in cal.runs]), hide_index=True, width="stretch")

        st.markdown(f"**Months** (follow {names[0]})")
        def mark(flag, off):
            return f"yes ({off:+.2f})" if flag else (f"{off:+.2f}" if off is not None else "")
        st.dataframe(pd.DataFrame([{
            "Month": m["month"], "Year": m["year"], "Day of year": m["day_of_year"], "Starts (day #)": m["start_day"],
            "Length (days)": m["length_days"], "Exact length": round(m["exact_days"], 2),
            "Full moon (day #)": m["full_day"], "New moon (day #)": m["new_day"],
            f"{names[1]} offset at full": mark(m["double_full"], m["full_offset"]),
            f"{names[1]} offset at new": mark(m["double_new"], m["new_offset"]),
        } for m in cal.months]), hide_index=True, width="stretch", height=420)

        st.markdown(f"**Weeks** (follow {names[1]})")
        st.dataframe(pd.DataFrame([{
            "Week": w["week"], "Year": w["year"], "Day of year": w["day_of_year"], "Starts (day #)": w["start_day"],
            "Length (days)": w["length_days"], "Exact length": round(w["exact_days"], 2),
            "In month": w["month"], "Day of month": w["day_of_month"]} for w in cal.weeks]),
            hide_index=True, width="stretch", height=420)
        st.caption("Real lunar months wander by a few tenths of a day around these averages, so a calendar that counts days "
                   "will drift slightly from the moon. Nineya's cycle is steadier than Liwaya's.")

# ------------------------------------------------------------------ Eras
if has_moons:
    with T["Eras"]:
        st.subheader("Eras of overlap")
        st.write("Liwaya pulls Nineya's orbit around, so the two orbital planes drift apart and come back together on a long cycle. "
                 "When the planes line up, the moons overlap often; when they are tilted away from each other, rarely. "
                 "Gold bands mark the stretches when the planes are within the chosen angle of each other.")
        e1, e2 = st.columns(2)
        e_years = e1.select_slider("Years to simulate", [70, 140, 200, 300], value=140)
        e_thr = e2.slider("Count the planes as aligned within (degrees)", 0.5, 5.0, 2.0, step=0.5)

        @st.cache_data(show_spinner="Simulating the eras...")
        def get_eras(system: System, years: int):
            return eras_data(system, years)

        ed = get_eras(system, e_years)
        bands = era_bands(ed, e_thr)
        es = era_summary(ed, bands)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overlaps per year, average", f"{es['per_year']:.1f}", f"{es['overlaps']} in {e_years} years", delta_color="off")
        m2.metric("Time between eras", f"{es['spacing']:.1f} years" if es["spacing"] else "n/a", f"{es['n_eras']} eras shown", delta_color="off")
        m3.metric("Overlaps inside eras", f"{es['share_in_era']:.0%}", f"eras cover {es['era_share_of_time']:.0%} of the time", delta_color="off")
        m4.metric("Double runs starting inside an era", f"{es['runs_in_era']} of {es['runs']}", "unrelated rhythms if close to the time share", delta_color="off")
        st.plotly_chart(plots.eras_figure(ed, bands, e_thr), width="stretch")

        if bands:
            mut = np.nan_to_num(ed.mutual_deg, nan=99.0)
            rows = []
            for i, (a0, b0) in enumerate(bands, 1):
                sel = (ed.plane_years >= a0) & (ed.plane_years <= b0)
                n_in = int(((ed.overlap_years >= a0) & (ed.overlap_years <= b0)).sum())
                rows.append({"Era": i, "From year": round(a0, 1), "To year": round(b0, 1), "Length (years)": round(b0 - a0, 1),
                             "Closest alignment (year)": round(float(ed.plane_years[sel][np.argmin(mut[sel])]), 1),
                             "Smallest plane angle (deg)": round(float(mut[sel].min()), 2), "Overlaps": n_in,
                             "Overlaps per year": round(n_in / max(b0 - a0, 0.01), 1)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Overlap = the two moons' discs touch or cover each other as seen from the planet's centre; seen from the "
                   "surface, Nineya shifts by up to 2.3 degrees depending on where you stand. Double-full and double-new use the "
                   "Calendar tab's half-day tolerance. The first era is cut off at year 0 because the founding moment starts aligned.")

# ------------------------------------------------------------------ Eclipses
if has_moons:
    with T["Eclipses"]:
        st.subheader("Eclipses")
        st.write("Counts are for the **whole planet**: an eclipse is counted if it is visible from anywhere. Any one spot sees far fewer. "
                 "A *total* solar eclipse needs the moon to look bigger than the Sun; if it looks smaller, the Sun shows as a ring "
                 "(*annular*); *hybrid* is in between. The Sun's size comes from the star radius in the sidebar.")
        names = tuple(m.name for m in system.moons)
        ey = st.select_slider("Years to simulate", [10, 20, 40, 100], value=40, key="ecl_years")

        @st.cache_data(show_spinner="Simulating eclipses...")
        def get_ecl_track(system: System, years: int):
            return integrate(system, np.arange(0, years * system.planet.year_solar_days * system.planet.local_day_days + 60, 0.25), elements=False)

        etr = get_ecl_track(system, ey)
        ev = eclipse_events(system, etr)
        rows, mets = [], []
        for nm in names:
            s_ = summarise([e for e in ev if e["year"] <= ey], nm, "solar", ey)
            c = s_["counts"]
            tot = (c["total"] + c["hybrid"]) / ey
            rows.append({"Moon": nm, "Total or hybrid per year": round(tot, 2), "Annular per year": round(c["annular"] / ey, 2),
                         "Partial only per year": round(c["partial"] / ey, 2),
                         "Share of new moons with an eclipse somewhere": f"{s_['share_any']:.0%}",
                         "Share of central eclipses that are total": f"{s_['share_total_of_central']:.0%}"})
        st.markdown("**Solar eclipses**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        lrows = []
        for nm in names:
            s_ = summarise([e for e in ev if e["year"] <= ey], nm, "lunar", ey)
            c = s_["counts"]
            lrows.append({"Moon": nm, "Total per year": round(c["total"] / ey, 2), "Partial per year": round(c["partial"] / ey, 2),
                          "Share of full moons eclipsed": f"{s_['share_any']:.0%}"})
        st.markdown("**Lunar eclipses** (moon inside the planet's shadow)")
        st.dataframe(pd.DataFrame(lrows), hide_index=True, width="stretch")
        st.plotly_chart(plots.eclipse_figure(ev, names, min(ey, 20)), width="stretch")
        st.caption("Timeline shows the first 20 years. Circles are solar eclipses, diamonds are lunar.")

        st.markdown("**What if the star were a different size?** (orbits, tilts and the Sun's distance unchanged)")
        wi = []
        for R in (1.0, 0.95, 0.92, 0.90, 0.88, 0.869, 0.80, 0.75):
            ev_r = eclipse_events(system, etr, R)
            row = {"Star radius (solar)": R, "Sun looks (deg)": round(planet.sun_angular_diameter_deg() * R / planet.star_radius_rsun, 3),
                   "Sunlight at Bayurinda": f"{planet.star_luminosity_lsun / planet.a_au**2:.2f}x (at current luminosity)"}
            for nm in names:
                s_ = summarise([e for e in ev_r if e["year"] <= ey], nm, "solar", ey)
                row[f"{nm}: central eclipses that are total"] = f"{s_['share_total_of_central']:.0%}"
            wi.append(row)
        st.dataframe(pd.DataFrame(wi), hide_index=True, width="stretch")
        st.caption("Penumbral lunar eclipses are not counted. The star-size table changes only the Sun's apparent size.")

        st.markdown("**What one place on the planet sees (solar eclipses)**")
        st.caption("The Sun has to be above the horizon, and the moon's position as seen from the surface is used, which matters most for Nineya.")

        @st.cache_data(show_spinner="Working out what each latitude sees...")
        def get_site_grid(system: System, years: int):
            tr = get_ecl_track(system, years)
            lats = np.arange(-80, 81, 20)
            lons = np.arange(0, 360, 45)
            la, lo = np.meshgrid(lats, lons, indexing="ij")
            ev_, ty_ = site_eclipse_types(system, tr, la.ravel(), lo.ravel())
            return lats, lons, ev_, ty_

        g_lats, g_lons, g_ev, g_ty = get_site_grid(system, ey)
        r0, r1 = site_rates(g_ev, g_ty, 0, ey), site_rates(g_ev, g_ty, 1, ey)
        prof = []
        for i, la in enumerate(g_lats):
            sl = slice(i * len(g_lons), (i + 1) * len(g_lons))
            tot = r0["total"][sl].mean() + r0["annular"][sl].mean()
            prof.append({"Latitude": int(la),
                         f"{names[0]}: eclipses seen per year": round(float(r0["any"][sl].mean()), 2),
                         f"{names[0]}: a total or ring eclipse about every (years)": (f"{1 / tot:,.0f}" if tot > 0.0005 else "rarely"),
                         f"{names[1]}: eclipses seen per year": round(float(r1["any"][sl].mean()), 2)})
        st.dataframe(pd.DataFrame(prof), hide_index=True, width="stretch")
        st.caption("Averaged over longitude. Totality paths are narrow (about 100 km), so a total eclipse at any one spot is a once-in-decades event, "
                   "while partial eclipses are common. The counts are rough for the rarest cases.")

        s_ev, s_ty = site_eclipse_types(system, etr, [lat], [lon])
        TYPE = {SITE_PARTIAL: "partial", SITE_ANNULAR: "ring (annular)", SITE_TOTAL: "total"}
        seen = [(e, int(s_ty[i, 0])) for i, e in enumerate(s_ev) if s_ty[i, 0] > 0]
        seen.sort(key=lambda x: x[0][1])
        st.markdown(f"**At latitude {lat}, longitude {lon}** (sidebar): {len(seen)} solar eclipses in {ey} years")
        if seen:
            st.dataframe(pd.DataFrame([{"Year": round(e[1] / planet.local_day_days / planet.year_solar_days, 2), "Moon": names[e[0]], "Type": TYPE[t_]}
                                       for e, t_ in seen[:40]]), hide_index=True, width="stretch", height=300)

# ------------------------------------------------------------------ Landmarks
with T["Landmarks"]:
    st.subheader("Sky landmarks")
    st.write("A monument can mark where something rises on the horizon, and the bodies come back to the same marks on a schedule. "
             f"Directions below are for a flat horizon at **latitude {lat}°** (change it in the sidebar), in degrees north (+) or south (-) of due east. "
             "Refraction and the size of the discs are ignored; Nineya is close enough that where you stand on the planet shifts it by up to 2.3°.")

    st.markdown("**The Sun**")
    sm = sun_markers(system)
    off = rise_offset_deg(sm["max_dec"], lat)
    rows = [("Northern winter solstice", sm["north_winter_day"], sm["min_dec"]), ("Spring equinox", sm["spring_equinox_day"], 0.0),
            ("Northern summer solstice", sm["north_summer_day"], sm["max_dec"]), ("Autumn equinox", sm["autumn_equinox_day"], 0.0)]
    st.dataframe(pd.DataFrame([{"Event": n, f"Day of the {sm['year_days_solar']:.0f}-day year": round(d), "Sun's declination": f"{dec:+.1f}°",
                                "Sunrise (° from east)": (f"{rise_offset_deg(dec, lat):+.1f}" if rise_offset_deg(dec, lat) is not None else "never rises / sets")}
                               for n, d, dec in rows]), hide_index=True, width="stretch")
    st.caption("The seasons are not equal because the orbit is slightly oval. Which way the axis leans relative to the closest point to Aureth is not settled; "
               "here the founding moment is a northern winter solstice and the closest approach together.")

    if has_moons:
        st.markdown("**The moons: standstills**")
        st.write("Each month a moon swings between a highest and a lowest point in the sky. Over the years that swing grows and shrinks; "
                 "the extremes are the *standstills*, when the moon rises at its farthest point along the horizon (major) or its nearest (minor).")

        @st.cache_data(show_spinner="Following the moons for 130 years...")
        def get_envelope(system: System):
            return moon_declination_envelope(system)

        env = get_envelope(system)
        st.plotly_chart(plots.standstill_figure(env, planet.year_days), width="stretch")
        srows = []
        for nm, v in env.items():
            for kind, key in (("major", "major"), ("minor", "minor")):
                for y_, amp in v[key][:4]:
                    ro = rise_offset_deg(amp, lat)
                    srows.append({"Moon": nm, "Standstill": kind, "Year": round(y_, 1), "Highest it climbs (declination)": f"{amp:.1f}°",
                                  "Rises (° from east, at its extreme)": f"{ro:+.1f}" if ro is not None else "-"})
        srows.sort(key=lambda r: (r["Moon"], r["Year"]))
        st.dataframe(pd.DataFrame(srows), hide_index=True, width="stretch", height=300)
        lw = env["Liwaya"]
        st.caption(f"Liwaya stays within {lw['range'][0]:.1f}° to {lw['range'][1]:.1f}° and its major standstills come about every 19 years, so a gate aimed at its "
                   f"most northerly moonrise is lit again each time. Nineya swings between {env['Nineya']['range'][0]:.0f}° and {env['Nineya']['range'][1]:.0f}°, "
                   "and how far it goes follows the 52-year cycle of the moons' orbital planes.")

        st.markdown("**When the whole sky comes back**")
        st.write("A moon is in the same place at the same hour on the same date again only when several cycles line up at once.")
        if st.button("Measure the return cycles (about 10 seconds)"):
            with st.spinner("Following the moons for 250 years..."):
                st.session_state["moon_returns"] = (system, moon_cycles(system))
        saved = st.session_state.get("moon_returns")
        if saved:
            if saved[0] != system:
                st.warning("The settings have changed since this was measured. Press the button again.")
            mcs = saved[1]
            label = {"calendar": "Same phase, same hour, same date in the year",
                     "eclipse": "Same phase, same height in the sky (node), same distance",
                     "sky": "All of these together, within about a day"}
            rr = []
            for nm in ("Liwaya", "Nineya"):
                res = return_cycles(mcs, planet.local_day_days, planet.year_days, nm)
                for prof, lst in res.items():
                    for x in lst[:3]:
                        rr.append({"Moon": nm, "What repeats": label[prof], "Months": x["months"], "Bayurindan years": round(x["years"], 2),
                                   "Node off (days)": round(x["node"], 2), "Distance off (days)": round(x["dist"], 2),
                                   "Hour off (h)": round(x["hour"], 1), "Date off (days)": round(x["date"], 1)})
            st.dataframe(pd.DataFrame(rr), hide_index=True, width="stretch", height=380)
            st.caption("'Off' columns show how far each ingredient misses after that many whole months (small is good). The node and distance misses are "
                       "uncertain by a few tenths of a day, and real months wander by a few tenths too.")

    st.markdown("**Fixed stars**")
    ap = axial_precession(system)
    sl = star_alignment_life(ap["years"], planet.obliquity_deg)
    stars_, _mw = load_sky()
    ax_ = spin_axis(planet)
    srows = []
    for st_ in stars_:
        v_ = ecl_vec(np.array(st_.lon), np.array(st_.lat))
        dec_ = math.degrees(math.asin(max(-1.0, min(1.0, float(v_ @ ax_)))))
        ro_ = rise_offset_deg(dec_, lat)
        if ro_ is not None:
            where = f"{ro_:+.1f}"
        elif (dec_ > 0) == (lat >= 0) and abs(dec_) > 90 - abs(lat):
            where = "never sets here"
        else:
            where = "never rises here"
        srows.append({"Star": st_.name, "Brightness (magnitude)": st_.mag, "Declination": f"{dec_:+.1f}°", "Rises (° from east)": where,
                      "Passes": ("north" if dec_ > lat else "south") + " of the zenith", "Note": st_.note})
    st.dataframe(pd.DataFrame(srows).sort_values("Brightness (magnitude)"), hide_index=True, width="stretch")
    st.write(f"The spin axis wobbles round once in about {ap['years']:,.0f} Bayurindan years, so a star's rising point slowly creeps. "
             f"At worst it moves {sl['per_century']:.2f}° per century (stars crossing the sky at 0 or 12 hours of right ascension), "
             "and not at all for stars at 6 or 18 hours.")
    st.dataframe(pd.DataFrame([{"A marker is good to": f"{tol}°" + (" (half a moon's width)" if tol == 0.25 else ""),
                                "Stays aligned for (worst case)": f"{yrs:.0f} years"} for tol, yrs in sl["years_to_drift"].items()]),
                 hide_index=True, width="stretch")
    st.caption("The stars here are placeholders (names, positions and brightness are in data/aureth.yaml). "
               + ("Alignments to the Sun's solstices and to Liwaya's standstills do not creep this way, because the moons hold the planet's tilt steady."
                  if has_moons else "Without moons to steady it, the axis wobbles only under the Sun's pull, which is slow but not nothing; "
                  "alignments to the Sun's solstices creep the same way."))


# ------------------------------------------------------------------ Night sky
with T["Night sky"]:
    st.subheader(f"The sky from {planet.name}")
    same_instant = ss["v_sky_time"] == "world"
    st.write(f"What you would see standing at **latitude {lat}°, longitude {lon}°** (set them in the sidebar), on a flat horizon. "
             + ("Time is the **world clock**, the clock at longitude 0, and every longitude sees the same instant, so moving east or west turns the planet under the sky. "
                if same_instant else
                "Time is **local solar time**: the clock moves with you, so the Sun is in the same place at the same reading wherever you stand. ")
             + "Midnight at longitude 0 is the founding moment, when the star and the three planets are in a line"
             + (" and both moons are full overhead." if has_moons else "."))

    def jump_founding():
        st.session_state.update({"v_sky_year": 1, "v_sky_day": 1, "v_sky_clock": 0.0})

    sky_box = st.container()          # the picture goes here, above the settings, and is filled in once the settings are known
    st.markdown("**Settings**")
    year_len = planet.year_solar_days
    STEPS = {"5 minutes": 5 / 60, "15 minutes": 0.25, "30 minutes": 0.5, "1 hour": 1.0, "3 hours": 3.0, "1 day": day_h,
             "3 days": 3 * day_h, "10 days": 10 * day_h}
    SPANS = {"1 hour": 1.0, "6 hours": 6.0, "1 day": day_h, "3 days": 3 * day_h, "10 days": 10 * day_h}
    if has_moons:
        moon_month = float(max(rates["synodic"])) / planet.local_day_days
        SPANS[f"one moon-month ({moon_month:.1f} days)"] = moon_month * day_h
    SPANS.update({"1 year": year_len * day_h, "4 years": 4 * year_len * day_h})
    if ss["v_sky_span"] not in SPANS:                     # the moon-month label changes with the settings
        ss["v_sky_span"] = "1 day"

    st.radio("Time is", ["local", "world"], key="v_sky_time", horizontal=True,
             format_func=lambda v: "Local time at the observer" if v == "local" else "Same instant everywhere (world clock)",
             help="Local time: the clock follows you, so the stars barely change with longitude. Same instant: the clock is the one at "
                  "longitude 0 and every longitude sees that moment, so longitude turns the sky.")
    k1, k2, k3, k4 = st.columns([1, 1, 2, 1.2])
    k1.number_input("Start: year", 1, 5000, step=1, key="v_sky_year")
    k2.number_input("Day of the year", 1, int(round(year_len)), step=1, key="v_sky_day")
    k3.slider(f"Start at time (hours after midnight; a day is {day_h:g} hours)", 0.0, float(int(2 * day_h - 1) / 2), step=0.25, key="v_sky_clock",
              help="Read as local time or as the world clock, depending on the 'Time is' choice above.")
    k4.button("Jump to the founding moment", on_click=jump_founding)
    p1, p2, p3 = st.columns(3)
    step_label = p1.select_slider("Time between frames", list(STEPS), key="v_sky_step")
    span_label = p2.select_slider("Play for", list(SPANS), key="v_sky_span")
    frame_ms = p3.slider("Pause between frames (milliseconds)", 20, 1000, step=10, key="v_sky_ms", help="Smaller is faster.")
    v1, v2, v3, v4, v5 = st.columns([1.2, 1.2, 1.3, 1.4, 1.4])
    view = v1.radio("View", ["Panorama", "All-sky dome"], key="v_sky_view", horizontal=True,
                    help="Panorama: the horizon laid out flat. Dome: looking straight up, north at the top, east on the left.")
    fov = v2.select_slider("Panorama width (degrees)", [360, 180, 90, 45], key="v_sky_fov",
                           help="360 shows the whole horizon in one wide, short strip. 180 or less gives a taller, bigger picture.")
    look_choices = ([m_.name for m_ in system.moons] + ["Aureth"] + [n_.name for n_ in system.neighbors] + [st_.name for st_ in load_sky()[0]]
                    + ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    if ss["v_sky_look"] not in look_choices:
        ss["v_sky_look"] = look_choices[0]
    look = v3.selectbox("Look toward", look_choices, key="v_sky_look")
    scale = v4.slider("Draw the Sun and moons larger by", 1, 40, key="v_sky_scale", help="1 is true size: the discs are about half a degree across.")
    sky_h = v5.slider("Picture height (pixels)", 450, 1500, step=50, key="v_sky_height",
                      help="Bigger is easier to see. The full-screen button at the top right of the picture goes even larger.")
    t1, t2, t3, t4, t5 = st.columns([1, 1, 1, 1.4, 1.8])
    named_on = t1.checkbox("Named stars", key="v_sky_named", help="The bright stars from the sky section of data/aureth.yaml. Names and positions are placeholders.")
    mw_on = t2.checkbox("Milky Way", key="v_sky_mw", help="Built from the galaxy's pole, centre and width in data/aureth.yaml.")
    faint_on = t3.checkbox("Faint stars", key="v_sky_faint", help="Decorative: crowded along the Milky Way, thin elsewhere.")
    pick_on = t4.checkbox("Pointer coordinates", key="v_sky_pick",
                          help="Hover over the sky to read its coordinates. Shows the start moment as a still picture, because the sky turns while it plays.")
    always_on = t5.checkbox("Stars in daytime too", key="v_sky_always",
                            help="Star-chart mode: the stars, Milky Way and constellation lines stay visible whatever the Sun is doing. The sky background goes dark "
                                 "so they can be seen. Switch it off for the realistic sky, where daylight hides them.")
    preview = None
    if pick_on:
        with st.expander("Preview a star at coordinates you choose", expanded=True):
            st.caption("Read a spot off the picture, type it here, and see where a star would sit before you edit data/aureth.yaml.")
            pc1, pc2, pc3, pc4 = st.columns(4)
            pv_on = pc1.checkbox("Show the preview", key="v_sky_pv_on")
            pv_lon = pc2.number_input("Ecliptic longitude (°)", 0.0, 360.0, step=0.5, key="v_sky_pv_lon")
            pv_lat = pc3.number_input("Ecliptic latitude (°)", -90.0, 90.0, step=0.5, key="v_sky_pv_lat")
            pv_mag = pc4.number_input("Brightness (magnitude)", -2.0, 6.0, step=0.1, key="v_sky_pv_mag", help="Smaller is brighter.")
            st.code(f'- {{name: "New star", lon: {pv_lon:.1f}, lat: {pv_lat:.1f}, mag: {pv_mag:.1f}, color: "#ffffff", label: true}}', language="yaml")
            st.caption("Paste that line into the `stars` list under `sky` in data/aureth.yaml, then change the name and colour.")
            if pv_on:
                preview = {"lon": float(pv_lon), "lat": float(pv_lat), "mag": float(pv_mag), "name": "preview"}

    # ---- constellation sets: the working copy lives in the session; the Save button writes it to data/constellations.yaml
    if "_con_work" not in ss:
        ss["_con_work"] = clone(load_constellations())
        ss["_con_disk"] = clone(ss["_con_work"])
    work = ss["_con_work"]
    star_list = list(load_sky()[0])
    star_names = [s_.name for s_ in sorted(star_list, key=lambda s_: (s_.mag, s_.name))]
    known = set(star_names)
    set_names = [s_["name"] for s_ in work]
    if ss.get("v_con_show") is None or any(n_ not in set_names for n_ in ss["v_con_show"]):
        ss["v_con_show"] = [n_ for n_ in (ss.get("v_con_show") or set_names[:1]) if n_ in set_names]
    if ss.get("v_con_edit") not in set_names:
        ss["v_con_edit"] = set_names[0] if set_names else None

    with st.expander("Constellation sets: join the same stars in different ways", expanded=False):
        st.caption("A **set** is one way of joining the stars (one culture's pictures, say). A set holds **figures**, and a figure is a handful of lines, each joining two "
                   "named stars. Sets are independent, so as many as you like can sit on the same stars, and you can show several at once in different colours. "
                   "Everything prefilled is a placeholder. Changes show in the picture straight away; **Save** writes them to data/constellations.yaml.")
        def sel_set():
            return next(s_ for s_ in work if s_["name"] == ss["v_con_edit"])

        def do_new():
            nm = (ss.get("_con_newname") or "").strip() or f"New set {len(work) + 1}"
            if nm in set_names:
                ss["_con_msg"] = f"A set called '{nm}' already exists."
                return
            work.append(new_set(nm, ["#8ecae6", "#f4a261", "#b8e986", "#d4a5ff", "#ff8fab", "#ffe066"][len(work) % 6]))
            ss["v_con_edit"] = nm; ss["v_con_show"] = ss["v_con_show"] + [nm]; ss["_con_newname"] = ""

        def do_dup():
            c0 = sel_set(); nm = f"{c0['name']} (copy)"
            while nm in set_names:
                nm += " (copy)"
            c1 = clone([c0])[0]; c1["name"] = nm
            work.append(c1); ss["v_con_edit"] = nm; ss["v_con_show"] = ss["v_con_show"] + [nm]

        def do_del():
            nm = ss["v_con_edit"]
            work[:] = [s_ for s_ in work if s_["name"] != nm]
            ss["v_con_show"] = [x for x in ss["v_con_show"] if x != nm]
            ss["v_con_edit"] = work[0]["name"] if work else None; ss["v_con_fig"] = None

        cA, cB = st.columns([3, 1])
        cA.multiselect("Show these sets", set_names, key="v_con_show", help="Each shown set is drawn in its own colour. The picture above updates as you change this.")
        cB.checkbox("Figure names", key="v_con_labels")
        if set_names:
            e1, e2, e3 = st.columns([2.2, 1, 1])
            e1.selectbox("Edit this set", set_names, key="v_con_edit")
            cur = sel_set()
            if ss["v_con_edit"] not in ss["v_con_show"]:
                e1.caption("This set isn't shown. Add it to 'Show these sets' to see your edits.")

            e2.button("Duplicate set", on_click=do_dup, key="con_dup", help="Copies this set, so you can rework the same figures into another reading.")
            e3.button("Delete set", on_click=do_del, key="con_del")
        n1, n2 = st.columns([3, 1])
        n1.text_input("Name for a new set", key="_con_newname", placeholder="e.g. The Salt Road reading")
        n2.button("Create new set", on_click=do_new, key="con_new")
        if ss.get("_con_msg"):
            st.warning(ss.pop("_con_msg"))

        if set_names and ss.get("v_con_edit") in set_names:
            cur = sel_set()
            st.markdown("---")
            g1, g2 = st.columns([3, 1])

            def set_name_cb():
                new_ = (ss["_con_setname"] or "").strip()
                if new_ and new_ not in [s_["name"] for s_ in work if s_ is not cur]:
                    was = cur["name"]; cur["name"] = new_
                    ss["v_con_show"] = [new_ if x == was else x for x in ss["v_con_show"]]; ss["v_con_edit"] = new_

            ss["_con_setname"] = cur["name"]
            g1.text_input("Set name", key="_con_setname", on_change=set_name_cb)
            cur["color"] = g2.color_picker("Line colour", cur["color"], key=f"_con_col_{id(cur)}")
            cur["note"] = st.text_input("Note (who tells these, what they mean...)", cur["note"], key=f"_con_note_{id(cur)}")
            bad = missing_stars(cur, known)
            if bad:
                st.warning("These stars are used in this set but no longer exist in the star list, so their lines are skipped: " + ", ".join(bad)
                           + ". Rename them back in data/aureth.yaml, or remove those lines below.")

            fig_names = [f_["name"] for f_ in cur["figures"]]
            if ss.get("v_con_fig") not in fig_names:
                ss["v_con_fig"] = fig_names[0] if fig_names else None
            f1, f2, f3 = st.columns([2.2, 2, 1])
            f1.selectbox("Figure to edit", fig_names or ["(no figures yet)"], key="v_con_fig", disabled=not fig_names)

            def add_fig():
                nm = (ss.get("_con_newfig") or "").strip() or f"Figure {len(cur['figures']) + 1}"
                if nm in [f_["name"] for f_ in cur["figures"]]:
                    ss["_con_msg"] = f"This set already has a figure called '{nm}'."
                    return
                cur["figures"].append({"name": nm, "lines": []}); ss["v_con_fig"] = nm; ss["_con_newfig"] = ""

            f2.text_input("Name for a new figure", key="_con_newfig", placeholder="e.g. The Heron")
            f3.button("Add figure", on_click=add_fig, key="con_addfig")
            if cur["figures"] and ss.get("v_con_fig") in fig_names:
                fg = next(f_ for f_ in cur["figures"] if f_["name"] == ss["v_con_fig"])

                def rename_fig():
                    new_ = (ss["_con_figname"] or "").strip()
                    if new_ and new_ not in [f_["name"] for f_ in cur["figures"] if f_ is not fg]:
                        fg["name"] = new_; ss["v_con_fig"] = new_

                def del_fig():
                    cur["figures"][:] = [f_ for f_ in cur["figures"] if f_ is not fg]; ss["v_con_fig"] = None

                r1, r2 = st.columns([3, 1])
                ss["_con_figname"] = fg["name"]
                r1.text_input("Figure name", key="_con_figname", on_change=rename_fig)
                r2.button("Delete this figure", on_click=del_fig, key="con_delfig")

                st.markdown("**Join stars with a chain.** Pick stars in the order you want them joined: the first to the second, the second to the third, and so on.")
                chain = st.multiselect("Stars in order", star_names, key="_con_chain", help="Brightest stars are listed first. Type to search.")

                def add_chain():
                    ch = ss["_con_chain"]
                    for a_, b_ in zip(ch, ch[1:]):
                        if a_ != b_ and [a_, b_] not in fg["lines"] and [b_, a_] not in fg["lines"]:
                            fg["lines"].append([a_, b_])
                    ss["_con_chain"] = []

                st.button("Add chain to this figure", on_click=add_chain, key="con_addchain", disabled=len(chain) < 2)
                lines_txt = [f"{a_}  -  {b_}" for a_, b_ in fg["lines"]]
                st.multiselect(f"Lines in '{fg['name']}' ({len(lines_txt)})", lines_txt, key="_con_rm",
                               placeholder="Pick lines here to remove them" if lines_txt else "No lines yet",
                               help="Select the lines you want to take out, then press 'Remove selected lines'.")

                def rm_lines():
                    gone = set(ss["_con_rm"])
                    fg["lines"][:] = [ln for ln in fg["lines"] if f"{ln[0]}  -  {ln[1]}" not in gone]; ss["_con_rm"] = []

                st.button("Remove selected lines", on_click=rm_lines, key="con_rm", disabled=not ss.get("_con_rm"))
            elif not cur["figures"]:
                st.caption("This set has no figures yet. Name one above and press Add figure.")

        dirty = work != ss["_con_disk"]

        def save_cons():
            save_constellations(work); ss["_con_disk"] = clone(work)

        def revert_cons():
            work[:] = clone(load_constellations()); ss["_con_disk"] = clone(work); ss["v_con_edit"] = None; ss["v_con_fig"] = None

        s1, s2, s3 = st.columns([1, 1, 2.2])
        s1.button("Save to data/constellations.yaml", on_click=save_cons, key="con_save", type="primary" if dirty else "secondary", disabled=not dirty)
        s2.button("Undo unsaved changes", on_click=revert_cons, key="con_revert", disabled=not dirty)
        s3.caption("**Unsaved changes.** They show in the picture and survive switching pages, but are lost if you close the app." if dirty
                   else "Everything is saved. Commit data/constellations.yaml if you want it in the repo.")
        with st.expander("Which stars can I use?"):
            st.dataframe(pd.DataFrame([{"star": s_.name, "brightness (mag)": s_.mag, "ecliptic longitude": s_.lon, "ecliptic latitude": s_.lat,
                                        "used in sets": ", ".join(c_["name"] for c_ in work if s_.name in sum((figure_stars(f_) for f_ in c_["figures"]), []))}
                                       for s_ in sorted(star_list, key=lambda s_: s_.mag)]), hide_index=True, width="stretch")

    shown_sets = [s_ for s_ in work if s_["name"] in ss["v_con_show"]]

    span_h = SPANS[span_label]
    step_h = min(STEPS[step_label], span_h)
    MAX_FRAMES = 240
    if span_h / step_h + 1 > MAX_FRAMES:
        step_h = span_h / (MAX_FRAMES - 1)
        st.info(f"That would be more than {MAX_FRAMES} frames, so the time between frames is widened to "
                f"{step_h:.1f} hours ({step_h / day_h:.2f} days).")
    day_total = (ss["v_sky_year"] - 1) * year_len + (ss["v_sky_day"] - 1)
    start_h = float(ss["v_sky_clock"])

    @st.cache_data(show_spinner="Working out the sky...")
    def get_sky(system: System, lat_: float, lon_: float, day_: float, start_: float, span_: float, step_: float, same_: bool):
        return compute_sky(system, lat_, lon_, day_, start_, span_, step_, same_)

    sky = get_sky(system, float(lat), float(lon), float(day_total), start_h, float(span_h), float(step_h), same_instant)
    ref = sky if (abs(span_h - day_h) < 1e-9 and abs(step_h - 0.5) < 1e-9) else get_sky(system, float(lat), float(lon), float(day_total), start_h, day_h, 0.5, same_instant)
    compass = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
    center = float(compass[look]) if look in compass else find_sight(sky, look, 0)
    def move_by(dlat: int, dlon: int):
        ss["v_lat"] = int(max(-90, min(90, ss["v_lat"] + dlat)))
        ss["v_lon"] = int((ss["v_lon"] + dlon + 180) % 360 - 180)

    def move_home():
        ss["v_lat"], ss["v_lon"] = 10, 0

    with sky_box:
        mv = st.columns([1, 1, 1, 1, 0.9, 2.4, 1.6])
        stp = int(ss["v_sky_move"])
        mv[0].button("West", icon=":material/arrow_back:", on_click=move_by, args=(0, -stp), key="mv_w", width="stretch", help="Move west (lower longitude).")
        mv[1].button("North", icon=":material/arrow_upward:", on_click=move_by, args=(stp, 0), key="mv_n", width="stretch", help="Move north (higher latitude).")
        mv[2].button("South", icon=":material/arrow_downward:", on_click=move_by, args=(-stp, 0), key="mv_s", width="stretch", help="Move south (lower latitude).")
        mv[3].button("East", icon=":material/arrow_forward:", on_click=move_by, args=(0, stp), key="mv_e", width="stretch", help="Move east (higher longitude).")
        mv[4].selectbox("Step", [1, 2, 5, 10, 15, 30, 45], key="v_sky_move", format_func=lambda v: f"{v}°", label_visibility="collapsed",
                        help="How far each arrow moves you, in degrees.")
        la_, lo_ = int(ss["v_lat"]), int(ss["v_lon"])
        mv[5].markdown(f"**Standing at {abs(la_)}°{'N' if la_ >= 0 else 'S'}, {abs(lo_)}°{'E' if lo_ >= 0 else 'W'}**"
                       f"  \n1° is about {math.pi * planet.radius_km / 180:.0f} km", help="The same position as the latitude and longitude sliders in the sidebar.")
        mv[6].button("Back to 10°N, 0°E", on_click=move_home, key="mv_home", width="stretch", icon=":material/home:")
        st.plotly_chart(sky_figure(sky, "dome" if view == "All-sky dome" else "panorama", center, float(fov), float(scale), named_on, 0, ref,
                                   int(frame_ms), mw_on, faint_on, int(ss["v_sky_seed"]), int(sky_h), pick_on, preview, shown_sets, bool(ss["v_con_labels"]), bool(always_on)),
                        width="stretch", config={"displaylogo": False, "toImageButtonOptions": {"width": 1920, "height": 1080, "scale": 1}})
        if pick_on:
            st.caption(f"Pointer mode: a still picture of {sky.labels[0]}. Move the pointer over the sky to read its **ecliptic longitude and latitude**, "
                       "the same coordinates the stars use in data/aureth.yaml (longitude 180 is where Aureth sits at the founding moment). "
                       "These are positions on the sky, not on the planet's surface. Narrow the panorama width for finer pointing.")
        else:
            st.caption(f"{len(sky.clock)} frames from {sky.labels[0]} to {sky.labels[-1]}. Press Play or drag the slider under the picture. "
                       "Hover over the picture and use the arrows at its top right to go full screen."
                       + (" Times are the world clock at longitude 0; 'here' is the local time where you stand." if same_instant else ""))
    st.caption(f"The table and the dotted paths describe the first moment and the {day_h:g} hours after it. Faded objects are below the horizon; "
               + ("the bright side of each moon faces Aureth. Nineya is close enough that where you stand shifts it against the sky. " if has_moons else "")
               + "The stars and the Milky Way are placeholders you can edit in data/aureth.yaml.")
    st.dataframe(pd.DataFrame(readout(sky, 0, ref)), hide_index=True, width="stretch")
    cx = {k: v for k, v in day_crossings(ref).items() if v and k in ["Aureth"] + [m_.name for m_ in system.moons]}
    if len({v["rightward"] for v in cx.values()}) > 1:
        def group(r):
            names = [k for k, v in cx.items() if v["rightward"] == r]
            return " and ".join(names), len(names) == 1
        east_names, east_one = group(True)
        west_names, west_one = group(False)
        st.info(f"Everything still rises in the east and sets in the west, but {east_names} cross{'es' if east_one else ''} south of the zenith "
                f"and drift{'s' if east_one else ''} left to right on the panorama, while {west_names} cross{'es' if west_one else ''} north of it "
                f"and drift{'s' if west_one else ''} right to left. At low latitudes, with a {planet.obliquity_deg:g}° axial tilt, bodies can pass on either side of the zenith. "
                "The dotted paths and arrows show each one's route, and the all-sky dome shows them going the same way round.")
    if ss["v_sky_year"] == 1 and ss["v_sky_day"] == 1 and ss["v_sky_clock"] == 0.0:
        st.caption("The first frame is the founding moment: " + ("Liwaya and Nineya full, overlapping, almost overhead, with " if has_moons else "")
                   + "the planets in the same line and Aureth directly beneath.")


# ------------------------------------------------------------------ Stability
if has_moons:
    with T["Stability"]:
        st.subheader("Orbital stability")
        st.write("N-body integration of star + planet + both moons. A single run is not conclusive: this system "
                 "can be stable for some starting phases and not others. Use the phase ensemble before trusting a configuration.")
        c1, c2, c3 = st.columns(3)
        years = c1.select_slider("Years to simulate", [10, 30, 100, 300, 1000, 3000], value=300)
        n_draw = c2.slider("Random-phase draws", 4, 16, 6)
        megno = c3.checkbox("Compute MEGNO chaos indicator", value=False, help="About 2 means regular; much larger than 2 means chaotic.")
        b1, b2 = st.columns(2)
        sig = (system, years)
        if b1.button("Run this configuration"):
            with st.spinner("Integrating..."):
                ss["stab"] = (sig, stability_run(system, years, megno=megno))
        if b2.button("Run phase ensemble"):
            bar = st.progress(0.0)
            res = phase_ensemble(system, years, n=n_draw, progress=bar.progress)
            ss["ens"] = (sig, res)

        if "stab" in ss:
            s_sig, r = ss["stab"]
            if s_sig != sig:
                st.info("Settings changed since this run.")
            (st.success if r.survived else st.error)(f"{'Stable' if r.survived else 'Unstable'}: {r.reason}.")
            if r.track.megno is not None:
                st.metric("MEGNO <Y>", f"{r.track.megno:.2f}")
            st.plotly_chart(plots.stability_figure(system, r), width="stretch")
            d = pd.DataFrame({
                "": [m.name for m in system.moons],
                "a min (km)": r.a_min.round(0), "a max (km)": r.a_max.round(0),
                "max e": r.e_max.round(3), "max incl (deg)": r.inc_max.round(2),
                "max distance / stability limit": r.max_dist_frac_of_limit.round(2),
            })
            st.dataframe(d, hide_index=True, width="stretch")
            st.caption(f"Closest moon-moon approach: {r.min_moon_sep_km:,.0f} km.  Osculating elements include real short-period "
                       "wobbles from the Sun, so they are not constant even in a perfectly stable system.")
        if "ens" in ss:
            e_sig, res = ss["ens"]
            ok = sum(x.survived for x in res)
            (st.success if ok == len(res) else st.warning if ok else st.error)(
                f"Phase ensemble: {ok} of {len(res)} random starts survived {e_sig[1]} years.")
            lost = sorted(x.lost_year for x in res if x.lost_year is not None)
            if lost:
                st.caption("Losses at (years): " + ", ".join(f"{y:.0f}" for y in lost))

# ------------------------------------------------------------------ Method
with T["Method & caveats"]:
    st.markdown("""
**Status labels** used throughout: *canon* (your established worldbuilding), *derived* (calculated from canon),
*hypothetical* (a design choice, such as anything about the moons), and *validated* (checked by simulation, within the stated limits).

**Orbits.** REBOUND IAS15 integration of the star, every planet and the chosen planet's moons (Newtonian point masses,
no planetary oblateness, no tides). The planet and its moons are started so that their shared centre of mass follows the planet's
orbit, which keeps the year at its intended length. The same planets, started the same way, drive the Aureth system page. Moon inclinations are measured from the planet's orbital plane, because both moons lie
well beyond the planet's Laplace radius (~10 planet radii), where the Sun, not the equatorial bulge, controls the orbit plane.

**Stability.** The Hill-limit figures use Domingos et al. (2006) for a single prograde satellite. With two moons the
real limit is lower. The N-body results only cover the simulated span (up to a few thousand years); extrapolating
to millions of years is not justified from this alone.

**Tides.** Equilibrium tide, degree-2 only: ζ = (m/M)(R/d)³·R·(3cos²ψ − 1)/2 for each body, summed. Valid for comparing
forcing, not for predicting real coastal tides. The 'strongest tide on the planet' trace is the largest eigenvalue of the
combined tidal tensor, i.e. the best case at any location.

**Phases.** Computed exactly from the three-dimensional geometry (angle at the moon between the Sun and the planet).

**Not yet modelled:** tidal evolution (spin-down, outward migration, locking times), eclipses,
planetary precession, moon obliquity and libration, atmospheric refraction.
""")
