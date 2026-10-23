from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from bajurinda import system_plots as sp
from bajurinda.aureth import (
    AU_KM, COLORS, K2, Body, Star, all_positions, conjunction_minima, crossing_times, grand_alignments, load_system,
    nbody_track, pair_table, period_days, planet_table, sky_from, synodic_days,
)
from bajurinda.transfers import find_windows, hohmann, launch_speed, porkchop

warnings.filterwarnings("ignore", category=RuntimeWarning)

STAR0, BODIES0 = load_system()
BAY = next(b for b in BODIES0 if b.name == "Bayurinda")
YEAR = period_days(STAR0, BAY)              # Bayurindan year, Earth days (440 solar days)
DAY = 19 / 24                               # Bayurindan solar day, Earth days
NAMES = [b.name for b in BODIES0]


def when(t_days: float) -> str:
    y = int(math.floor(t_days / YEAR)) + 1
    d = int((t_days % YEAR) / DAY) + 1
    return f"year {y}, day {d}"


def reset_system():
    for b in BODIES0:
        st.session_state.update({f"sys_{b.name}_a": float(b.a_au), f"sys_{b.name}_e": float(b.ecc),
                                 f"sys_{b.name}_m": float(b.mass_earth), f"sys_{b.name}_ph": float(b.phase_deg)})


if "sys_Bayurinda_a" not in st.session_state:
    reset_system()
ss = st.session_state

with st.sidebar:
    st.subheader("Aureth system")
    st.caption("Star: mass 1.0021 solar, luminosity 0.87, radius 0.985 (set on the moons page / data file).")
    st.button("Reset planets", on_click=reset_system, key="sys_reset")
    for b in BODIES0:
        with st.expander(b.name + ("  (placeholders)" if b.placeholder_note else "")):
            st.number_input("Distance (AU)", 0.3, 5.0, step=0.01, format="%.4f", key=f"sys_{b.name}_a")
            st.slider("Eccentricity", 0.0, 0.4, step=0.005, key=f"sys_{b.name}_e")
            st.number_input("Mass (Earth masses)", 0.01, 30.0, step=0.01, key=f"sys_{b.name}_m")
            st.slider("Position at the founding moment (deg of orbit)", 0.0, 360.0, step=1.0, key=f"sys_{b.name}_ph",
                      help="0 puts the planet at its closest point to Aureth on the shared starting line.")
            if b.placeholder_note:
                st.caption(f"Not settled: {b.placeholder_note}.")

STAR = STAR0
BODIES = [replace(b, a_au=ss[f"sys_{b.name}_a"], ecc=ss[f"sys_{b.name}_e"], mass_earth=ss[f"sys_{b.name}_m"],
                  phase_deg=ss[f"sys_{b.name}_ph"]) for b in BODIES0]
BY = {b.name: b for b in BODIES}

st.title("Aureth system")
st.caption("Time is counted from the founding moment (the double full moon), when the star, Bayurinda, Ashara and Tarnic "
           "start in a line. Dates read 'year, day' in Bayurindan years of 440 solar days. Orbits are flat and coplanar "
           "(a placeholder), and Ashara's eccentricity and mass are placeholders.")

tabs = st.tabs(["Orbits", "Cycles", "Sky from Bayurinda", "Transfers", "Stability"])


# ------------------------------------------------------------------ Orbits
with tabs[0]:
    years = st.select_slider("Years to animate (Earth years)", [2, 5, 10, 20], value=5)

    @st.cache_data(show_spinner="Integrating the planets...")
    def get_orbits(star: Star, bodies: tuple[Body, ...], span_days: float):
        t = np.arange(0, span_days, 5.0)
        return nbody_track(star, list(bodies), t)

    track = get_orbits(STAR, tuple(BODIES), years * 365.25)
    st.plotly_chart(sp.orbit_figure(STAR, BODIES, track["t"], track["pos"]), width="stretch")
    st.caption("N-body integration (REBOUND). Dotted lines are the orbits; coloured tails show recent motion.")


# ------------------------------------------------------------------ Cycles
with tabs[1]:
    st.subheader("Planets")
    rows = []
    for r in planet_table(STAR, BODIES):
        rows.append({"Planet": r["name"], "Distance (AU)": round(r["a_au"], 4), "Eccentricity": r["ecc"],
                     "Year (Earth days)": round(r["period_days"], 2), "Year (Bayurindan days)": round(r["period_bayurinda_days"], 1),
                     "Year (its own days)": round(r["period_local_days"], 1) if r["period_local_days"] else None,
                     "Sunlight (Earth = 1)": round(r["flux"], 2),
                     "Sunlight over the orbit": f"{r['flux_range'][0]:.2f} to {r['flux_range'][1]:.2f}",
                     "Sun looks (deg)": round(r["sun_deg"], 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Mars gets 0.43 and Venus 1.9 times Earth's sunlight, for reference.")

    st.subheader("Pairs of planets")
    prow = []
    for r in pair_table(STAR, BODIES):
        res = ", ".join(f"{x['resonance']} ({x['offset_pct']:+.2f}%)" for x in r["resonances"]) or "none close"
        prow.append({"Pair": r["pair"], "Period ratio": round(r["ratio"], 4), "Time between like alignments (Earth years)": round(r["synodic_days"] / 365.25, 2),
                     "(Bayurindan days)": round(r["synodic_days"] / DAY), "Near a resonance": res})
    st.dataframe(pd.DataFrame(prow), hide_index=True, width="stretch")

    st.subheader("Grand alignments")
    g1, g2 = st.columns(2)
    g_years = g1.select_slider("Look ahead (Earth years)", [100, 300, 1000, 3000], value=300)
    g_tol = g2.slider("All planets within (degrees of each other, seen from the star)", 3, 40, 12)
    al = grand_alignments(STAR, BODIES, g_years, g_tol)
    if al:
        st.dataframe(pd.DataFrame([{"When": when(a["t"]), "Earth years": round(a["t"] / 365.25, 1),
                                    "Tightest spread (deg)": round(a["spread_deg"], 1), "Lasts (days)": round(a["length_days"])} for a in al]),
                     hide_index=True, width="stretch", height=300)
        gaps = np.diff([a["t"] for a in al]) / 365.25
        if len(gaps):
            st.caption(f"{len(al)} alignments in {g_years} years; the gap between them averages {gaps.mean():.1f} Earth years "
                       f"(shortest {gaps.min():.1f}, longest {gaps.max():.1f}). The first is the founding moment itself.")
    else:
        st.info("No alignment that tight in this span.")


# ------------------------------------------------------------------ Sky
with tabs[2]:
    st.subheader("What Bayurinda's sky shows")
    s_years = st.select_slider("Years to look at (Earth years)", [5, 10, 20, 40, 80], value=20, key="sky_years")
    t = np.arange(0, s_years * 365.25, 1.0)
    pos = all_positions(STAR, BODIES, t)
    sky = sky_from(pos, NAMES, "Bayurinda")
    opp, rows = {}, []
    for n, d in sky.items():
        o = crossing_times(t, d["lon_offset"], math.pi)
        c = crossing_times(t, d["lon_offset"], 0.0)
        opp[n] = o
        for tt in o:
            rows.append({"Planet": n, "Event": "opposition (rises at sunset, closest)", "When": when(tt), "t": tt,
                         "Distance (AU)": round(float(np.interp(tt, t, d["distance_au"])), 3)})
        for tt in c:
            rows.append({"Planet": n, "Event": "behind the Sun (hidden)", "When": when(tt), "t": tt,
                         "Distance (AU)": round(float(np.interp(tt, t, d["distance_au"])), 3)})
    rows.sort(key=lambda r: r["t"])
    st.plotly_chart(sp.sky_figure(t, sky, YEAR, opp), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Oppositions and hidden periods**")
        st.dataframe(pd.DataFrame(rows).drop(columns="t"), hide_index=True, width="stretch", height=320)
    with c2:
        st.markdown("**Ashara and Tarnic close together in the sky**")
        sep = np.degrees(np.arccos(np.clip(
            np.einsum("ij,ij->i", sky["Ashara"]["vec"], sky["Tarnic"]["vec"])
            / (np.linalg.norm(sky["Ashara"]["vec"], axis=1) * np.linalg.norm(sky["Tarnic"]["vec"], axis=1)), -1, 1)))
        lim = st.slider("Closer than (degrees)", 1, 20, 5)
        close = conjunction_minima(t, sep, lim)
        if close:
            st.dataframe(pd.DataFrame([{"When": when(tt), "Gap (deg)": round(g, 2),
                                        "Angle from the Sun (deg)": round(float(np.interp(tt, t, sky['Tarnic']['elongation_deg'])))}
                                       for tt, g in close]), hide_index=True, width="stretch", height=260)
        else:
            st.info("Never that close in this span.")
    st.caption("Angles are seen from Bayurinda's centre. Planets close to the Sun in the sky (small angle) are lost in its glare.")


# ------------------------------------------------------------------ Transfers
with tabs[3]:
    st.subheader("Launch windows")
    st.write("A transfer is cheapest when the two planets are in the right places. The cost shown is the speed the traveller "
             "gains leaving one planet plus the speed they must shed at the other, both measured against the planet (km/s). "
             "How the mechs actually pay for that, with their wings and nearing, is up to you; use the budget slider to see "
             "which departures fit a limit you choose.")
    t1, t2, t3, t4 = st.columns(4)
    origin = t1.selectbox("From", NAMES, index=0)
    dest = t2.selectbox("To", [n for n in NAMES if n != origin], index=1 if origin == "Bayurinda" else 0)
    span_y = t3.select_slider("Departures over (Earth years)", [5, 10, 20], value=10)
    which = t4.selectbox("Show", ["total", "vinf_dep", "vinf_arr"], format_func=lambda k: {"total": "Total", "vinf_dep": "Leaving only", "vinf_arr": "Arriving only"}[k])
    O, Dst = BY[origin], BY[dest]
    mu = K2 * STAR.mass_msun
    syn = synodic_days(period_days(STAR, O), period_days(STAR, Dst))
    h = hohmann(O.a_au, Dst.a_au, mu)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cheapest flight time (circular orbits)", f"{h['tof_days']:.0f} days", f"{h['tof_days'] / DAY:.0f} Bayurindan days", delta_color="off")
    m2.metric("Speed change, leaving + arriving", f"{h['vinf_dep'] + h['vinf_arr']:.2f} km/s", f"{h['vinf_dep']:.2f} + {h['vinf_arr']:.2f}", delta_color="off")
    m3.metric("Time between good windows", f"{syn / 365.25:.2f} years", f"{syn / DAY:.0f} Bayurindan days", delta_color="off")
    m4.metric("Destination must lead by", f"{h['lead_deg']:.0f}°", "at departure", delta_color="off")
    default_budget = round(1.3 * (h["vinf_dep"] + h["vinf_arr"]), 1)
    budget = st.slider("Budget: total speed change you will allow (km/s)", 0.5, 20.0, float(default_budget), step=0.1, key=f"budget_{origin}_{dest}")

    @st.cache_data(show_spinner="Working out the flights...")
    def get_porkchop(star: Star, o: Body, d: Body, span_days: float):
        sy = synodic_days(period_days(star, o), period_days(star, d))
        hh = hohmann(o.a_au, d.a_au, K2 * star.mass_msun)
        dep = np.arange(-0.5 * sy, span_days + 0.5 * sy, 5.0)
        tof = np.arange(0.4 * hh["tof_days"], 2.0 * hh["tof_days"], 5.0)
        return porkchop(star, o, d, dep, tof)

    span_days = span_y * 365.25
    pc = get_porkchop(STAR, O, Dst, span_days)
    wins = find_windows(pc, syn, 0, span_days, 14)
    cap = max(12.0, 2.5 * (h["vinf_dep"] + h["vinf_arr"]))
    st.plotly_chart(sp.porkchop_figure(pc, YEAR, wins, budget, cap, 0, span_days, key=which), width="stretch")
    st.caption("Darker is cheaper. The dashed line marks your budget. Crosses mark the best departure in each window. "
               "Dates count from the founding moment, when all the planets start in a line; that is close to a window for some routes and far from one for others.")
    if wins:
        out = []
        for w in wins:
            ls = launch_speed(O, w["vinf_dep"])
            out.append({"Depart": when(w["dep_day"]), "Flight (Earth days)": round(w["tof_days"]),
                        "Flight (Bayurindan days)": round(w["tof_days"] / DAY), "Arrive": when(w["dep_day"] + w["tof_days"]),
                        "Leaving (km/s)": round(w["vinf_dep"], 2), "Arriving (km/s)": round(w["vinf_arr"], 2),
                        "Total (km/s)": round(w["total"], 2), "Within budget": "yes" if w["total"] <= budget else "no",
                        "Launch speed from the surface (km/s)": round(ls, 2) if ls else "unknown"})
        st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")
        st.caption("Launch speed from the surface = what it takes to escape the planet plus the 'leaving' speed, ignoring atmosphere and spin. "
                   "Flights use exact two-body orbits for the planets and a single curved path between them.")


# ------------------------------------------------------------------ Stability
with tabs[4]:
    st.subheader("Do the planets keep their orbits?")
    s1, s2 = st.columns(2)
    s_len = s1.select_slider("Years to simulate (Earth years)", [100, 1000, 10_000, 100_000], value=1000)
    go_ = s2.button("Run the integration")

    @st.cache_data(show_spinner="Integrating...")
    def get_stability(star: Star, bodies: tuple[Body, ...], years_: float):
        t = np.linspace(0, years_ * 365.25, 500)
        return nbody_track(star, list(bodies), t)

    if go_:
        st.session_state["sys_stab"] = (s_len, get_stability(STAR, tuple(BODIES), s_len))
    if "sys_stab" in st.session_state:
        yrs, tr = st.session_state["sys_stab"]
        st.plotly_chart(sp.stability_figure(tr["t"], tr["a"], tr["e"], NAMES), width="stretch")
        rows = []
        for k, b in enumerate(BODIES):
            rows.append({"Planet": b.name, "Distance range (AU)": f"{tr['a'][:, k].min():.4f} to {tr['a'][:, k].max():.4f}",
                         "Largest eccentricity": round(float(tr["e"][:, k].max()), 3)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        pos = tr["pos"]
        close = min(np.linalg.norm(pos[:, i] - pos[:, j], axis=1).min() for i in range(len(BODIES)) for j in range(i + 1, len(BODIES)))
        sep = []
        for i in range(len(BODIES) - 1):
            b1, b2 = BODIES[i], BODIES[i + 1]
            rh = ((b1.mass_msun + b2.mass_msun) / (3 * STAR.mass_msun)) ** (1 / 3) * (b1.a_au + b2.a_au) / 2
            sep.append(f"{b1.name}-{b2.name}: {(b2.a_au - b1.a_au) / rh:.0f}")
        st.caption(f"Simulated {yrs:,} years (sampled 500 times). Closest the sampled planets came to each other: {close:.3f} AU. "
                   f"Spacing in mutual Hill radii (above about 3.5 is stable for circular orbits): {'; '.join(sep)}. "
                   "Moons are treated as part of their planet.")
    else:
        st.info("Press the button to simulate.")
