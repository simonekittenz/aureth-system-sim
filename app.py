from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from bajurinda import plots
from bajurinda.constants import M_EARTH, M_SUN, YEAR_DAYS
from bajurinda.eclipses import SITE_ANNULAR, SITE_PARTIAL, SITE_TOTAL, eclipse_events, site_eclipse_types, site_rates, summarise
from bajurinda.eras import era_bands, era_summary, eras_data
from bajurinda.lunar_calendar import build_calendar
from bajurinda.dynamics import integrate, mean_rates, phase_ensemble, stability_run
from bajurinda.model import (
    Moon, Planet, System, canon_planet, moon_report, planet_warnings, system_report,
)
from bajurinda.presets import DEFAULT_PRESET, PRESETS
from bajurinda.sky import (
    alignment_events, altitude_deg, crossings, tide_series,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
st.set_page_config(page_title="Bajurinda two-moon simulator", layout="wide")

CANON = canon_planet()
MOON_KEYS = ("L", "N")


def load_preset():
    _, liw, nin = PRESETS[st.session_state["preset"]]
    for key, m in zip(MOON_KEYS, (liw, nin)):
        st.session_state.update({
            f"{key}_dia": float(m.diameter_km), f"{key}_rho": float(m.density_gcc),
            f"{key}_a": int(m.a_km), f"{key}_e": float(m.ecc), f"{key}_inc": float(m.inc_deg),
            f"{key}_M": float(m.mean_anomaly_deg), f"{key}_Om": float(m.asc_node_deg),
            f"{key}_w": float(m.arg_peri_deg),
        })


def reset_planet():
    st.session_state.update({
        "p_lum": CANON.star_luminosity_lsun, "p_srad": CANON.star_radius_rsun,
        "p_radius": CANON.radius_km, "p_mass": CANON.mass_earth, "p_rot": CANON.rotation_hours,
        "p_sid": "solar", "p_a": CANON.a_au, "p_ecc": CANON.ecc, "p_obl": CANON.obliquity_deg,
        "p_star": CANON.star_mass_msun,
    })


if "preset" not in st.session_state:
    st.session_state["preset"] = DEFAULT_PRESET
    load_preset()
    reset_planet()

ss = st.session_state

with st.sidebar:
    st.title("Bajurinda")
    st.caption(PRESETS[DEFAULT_PRESET][0])
    st.button("Reset moons to Design B", on_click=load_preset)

    with st.expander("Planet and star (canon defaults)", expanded=False):
        st.button("Reset to canon", on_click=reset_planet)
        st.number_input("Radius (km)", 1000.0, 20000.0, step=50.0, key="p_radius")
        st.number_input("Mass (Earth masses)", 0.05, 10.0, step=0.01, format="%.4f", key="p_mass")
        st.number_input("Rotation period (Earth hours)", 2.0, 200.0, step=0.5, key="p_rot")
        st.radio("The rotation period is a", ["solar", "sidereal"], key="p_sid", horizontal=True,
                 help="Sidereal: relative to the stars. Solar: noon to noon.")
        st.number_input("Distance from star (AU)", 0.1, 5.0, step=0.01, format="%.3f", key="p_a")
        st.number_input("Orbital eccentricity", 0.0, 0.5, step=0.005, format="%.3f", key="p_ecc")
        st.number_input("Axial tilt (deg)", 0.0, 90.0, step=1.0, key="p_obl")
        st.number_input("Stellar mass (solar masses)", 0.2, 3.0, step=0.001, format="%.4f", key="p_star",
                        help="Aureth is G-type. The default is the mass the original canon implied (a 374-day year at 0.87 AU), kept fixed.")
        st.number_input("Stellar luminosity (solar)", 0.01, 5.0, step=0.01, format="%.3f", key="p_lum",
                        help="A young 1-solar-mass star is about 0.68 to 0.76 (rough).")
        st.number_input("Stellar radius (solar)", 0.1, 3.0, step=0.01, format="%.3f", key="p_srad",
                        help="A young 1-solar-mass star is about 0.87 to 0.9. Liwaya can only fully cover the Sun if this is under ~0.9.")

    for key, label in zip(MOON_KEYS, ("Liwaya", "Nineya")):
        with st.expander(label, expanded=False):
            st.number_input("Diameter (km)", 200.0, 8000.0, step=50.0, key=f"{key}_dia")
            st.number_input("Density (g/cm³)", 1.0, 6.0, step=0.1, key=f"{key}_rho")
            st.number_input("Starting semi-major axis (km)", 30_000, 700_000, step=100, key=f"{key}_a",
                            help="Osculating value at the start. The simulated average distance differs by about 1%; see the Overview.")
            st.slider("Eccentricity", 0.0, 0.3, step=0.005, key=f"{key}_e")
            st.slider("Inclination to the ecliptic (deg)", 0.0, 30.0, step=0.1, key=f"{key}_inc")
            st.slider("Starting orbital phase (deg)", 0.0, 360.0, step=1.0, key=f"{key}_M")
            st.slider("Longitude of ascending node (deg)", 0.0, 360.0, step=1.0, key=f"{key}_Om")
            st.slider("Argument of periapsis (deg)", 0.0, 360.0, step=1.0, key=f"{key}_w")

    st.subheader("View")
    window = st.slider("Simulated span (Earth days)", 30, 1500, 240, step=10)
    step_h = st.select_slider("Sampling step (hours)", [1, 2, 3, 4, 6, 12], value=3)
    lat = st.slider("Observer latitude (deg)", -90, 90, 10)
    lon = st.slider("Observer longitude (deg)", -180, 180, 0)
    unit = st.radio("Time unit for plots", ["Earth days", "Bajurindan days"], horizontal=True)


def build_system() -> System:
    planet = Planet(
        radius_km=ss["p_radius"], mass_kg=ss["p_mass"] * M_EARTH, rotation_hours=ss["p_rot"],
        rotation_is_sidereal=ss["p_sid"] == "sidereal", a_au=ss["p_a"], ecc=ss["p_ecc"],
        obliquity_deg=ss["p_obl"], star_mass_kg=ss["p_star"] * M_SUN,
        star_luminosity_lsun=ss["p_lum"], star_radius_rsun=ss["p_srad"],
    )
    moons = tuple(
        Moon(name, ss[f"{k}_dia"], ss[f"{k}_rho"], float(ss[f"{k}_a"]), ss[f"{k}_e"], ss[f"{k}_inc"],
             ss[f"{k}_M"], ss[f"{k}_Om"], ss[f"{k}_w"])
        for k, name in zip(MOON_KEYS, ("Liwaya", "Nineya"))
    )
    return System(planet, moons)


system = build_system()
planet = system.planet
unit_days = planet.local_day_days if unit == "Bajurindan days" else 1.0
unit_label = "Bajurindan (solar) days" if unit == "Bajurindan days" else "Earth days"


@st.cache_data(show_spinner="Integrating orbits with REBOUND...")
def get_track(system: System, window_days: int, step_hours: int):
    times = np.arange(step_hours / 24, window_days + 1e-9, step_hours / 24)
    times = np.insert(times, 0, 0.0)
    return integrate(system, times)


track = get_track(system, window, step_h)


@st.cache_data(show_spinner="Measuring mean orbital periods...")
def get_rates(system: System):
    return mean_rates(system)


rates = get_rates(system)
report_system = System(planet, tuple(replace(m, a_km=d) for m, d in zip(system.moons, rates["distance"])))
report = system_report(report_system)

tabs = st.tabs(["Overview", "Orbits", "Phases & sky", "Tides", "Alignments", "Calendar", "Eras", "Eclipses", "Stability", "Method & caveats"])

# ------------------------------------------------------------------ Overview
with tabs[0]:
    st.header("Bajurinda: two-moon system")
    for sev, msg in planet_warnings(report_system):
        {"error": st.error, "warning": st.warning, "info": st.info}[sev](msg)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Planet and calendar")
        rows = [
            ("Radius", f"{planet.radius_km:,.0f} km", "canon (diameter 12,000 km)"),
            ("Surface gravity", f"{planet.gravity_g:.3f} g", "canon 0.9 g"),
            ("Mass", f"{planet.mass_earth:.4f} Earth", "derived from g and radius"),
            ("Mean density", f"{planet.density_gcc:.2f} g/cm³", "derived"),
            ("Escape velocity", f"{planet.escape_velocity_kms:.2f} km/s", "derived"),
            ("Rotation (sidereal)", f"{planet.sidereal_rotation_hours:.4f} h", "derived from the 19 h solar day"),
            ("Solar day", f"{planet.solar_day_hours:.4f} h", "canon 19 h"),
            ("Year", f"{planet.year_days:.3f} Earth days", "derived (Kepler)"),
            ("Year in sidereal rotations", f"{planet.sidereal_rotations_per_year:.2f}", "derived"),
            ("Year in solar days", f"{planet.solar_days_per_year:.2f}", "derived"),
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
    q = round(S[longi] / S[shorti])
    repeat = None
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
with tabs[1]:
    st.subheader("Top-down view, planet-centred")
    st.caption("Viewed from above the ecliptic. The dotted circle is the approximate single-moon Hill stability limit. "
               "Press Play, or drag the slider.")
    st.plotly_chart(plots.orbit_figure(system, track), width="stretch")

# ------------------------------------------------------------------ Phases & sky
with tabs[2]:
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
with tabs[3]:
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
with tabs[4]:
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
with tabs[5]:
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
with tabs[6]:
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
with tabs[7]:
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

# ------------------------------------------------------------------ Stability
with tabs[8]:
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
with tabs[9]:
    st.markdown("""
**Status labels** used throughout: *canon* (your established worldbuilding), *derived* (calculated from canon),
*hypothetical* (a design choice, such as anything about the moons), and *validated* (checked by simulation, within the stated limits).

**Orbits.** REBOUND IAS15 integration of the star, planet and both moons (Newtonian point masses, no planetary
oblateness, no tides). Moon inclinations are measured from the planet's orbital plane, because both moons lie
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
