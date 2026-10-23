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
from bajurinda.flights import route_extremes, scan_route
from bajurinda.transfers import find_windows, hohmann, porkchop

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


ss = st.session_state
for _k, _v in ss.get("_sys_saved", {}).items():      # see moons.py: Streamlit drops widget values when another page is shown
    if _k not in ss:
        ss[_k] = _v
if "sys_Bayurinda_a" not in ss:
    reset_system()

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

ss["_sys_saved"] = {k: v for k, v in ss.items() if k.startswith("sys_") and k not in ("sys_stab", "sys_reset")}

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
    st.subheader("Sternreyker flights")
    st.write("A Sternreyker pilot keeps the vessel accelerating the whole way: speed up until halfway, then brake to stop at the "
             "destination. There are no launch windows. What changes from day to day is how far apart the two planets are, "
             "and whether Aureth is in the way. A Weltreyker gets the vessel airborne first; the flights below begin once the "
             "wings are up, so add the launch and landing yourself with the overhead box.")
    f1, f2, f3, f4, f5 = st.columns(5)
    origin = f1.selectbox("From", NAMES, index=0)
    dest = f2.selectbox("To", [n for n in NAMES if n != origin], index=1 if origin == "Bayurinda" else 0)
    span_y = f3.select_slider("Departures over (Earth years)", [3, 10, 20], value=10)
    keep = f4.slider("Stay away from Aureth by (AU)", 0.05, 0.6, 0.30, step=0.05,
                     help="Closer than this the heat and gravity are too much. The Sun itself is only 0.005 AU in radius.")
    over_h = f5.number_input("Launch + landing overhead (hours)", 0.0, 72.0, 0.0, step=1.0)
    O, Dst = BY[origin], BY[dest]
    span_days = span_y * 365.25
    G_NOV, G_EXP = 0.6, 1.4
    g_mid = st.slider("Your own pilot's acceleration (g)", 0.3, 2.0, 1.0, step=0.1)
    levels = {f"novice {G_NOV} g": G_NOV, f"your pilot {g_mid} g": g_mid, f"expert {G_EXP} g": G_EXP}

    @st.cache_data(show_spinner="Working out the flights...")
    def get_extremes(star: Star, bodies: tuple[Body, ...], span: float, keep_au: float):
        out = {}
        by = {b.name: b for b in bodies}
        for o, d in (("Bayurinda", "Ashara"), ("Ashara", "Tarnic"), ("Bayurinda", "Tarnic")):
            for g in (G_NOV, G_EXP):
                out[(o, d, g)] = route_extremes(star, by[o], by[d], g, span, keep_au)
        return out

    @st.cache_data(show_spinner="Scanning departure dates...")
    def get_scan(star: Star, o: Body, d: Body, g: float, span: float, keep_au: float):
        return scan_route(star, o, d, g, span, keep_au)

    ext = get_extremes(STAR, tuple(BODIES), span_days, keep)
    ov = over_h / 24
    st.markdown("**Best and worst timing, every route** (days, including your overhead)")
    rows = []
    for (o, d) in (("Bayurinda", "Ashara"), ("Ashara", "Tarnic"), ("Bayurinda", "Tarnic")):
        r1, r2 = ext[(o, d, G_NOV)], ext[(o, d, G_EXP)]
        rows.append({"Route": f"{o} - {d}",
                     f"Best, novice {G_NOV} g": round(r1["best_T"] + ov, 2), f"Best, expert {G_EXP} g": round(r2["best_T"] + ov, 2),
                     f"Worst, novice {G_NOV} g": round(r1["worst_T"] + ov, 2), f"Worst, expert {G_EXP} g": round(r2["worst_T"] + ov, 2),
                     "Planets closest (AU)": round(r1["best_dist"], 2), "Planets farthest (AU)": round(r1["worst_dist"], 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Best: the planets are on the same side of Aureth and as close as they get. Worst: on opposite sides, with Aureth between "
               "them, so the pilot has to curve around it. Flights are straight and the Sun's pull is ignored except for the detours, "
               "so allow about 2% either way. Top speeds are 1,300 to 2,300 km/s, under 1% of the speed of light.")

    scans = {lab: get_scan(STAR, O, Dst, g, span_days, keep) for lab, g in levels.items()}
    mid = scans[f"your pilot {g_mid} g"]
    marks = {"best_t0": float(mid["t0"][mid["T"].argmin()]), "best_T": float(mid["T"].min()),
             "worst_t0": float(mid["t0"][mid["T"].argmax()]), "worst_T": float(mid["T"].max())}
    for sc in scans.values():
        sc["T"] = sc["T"] + ov
    marks["best_T"] += ov
    marks["worst_T"] += ov
    st.plotly_chart(sp.flight_figure(scans, YEAR, keep, marks), width="stretch")
    blocked_share = float(mid["blocked"].mean())
    st.caption(f"Orange bands: the straight line would pass within {keep:.2f} AU of Aureth ({blocked_share:.0%} of departures), "
               "so the pilot curves around it. That adds about 2% at a 0.3 AU berth and more for a wider one (about 10% at 0.55 AU). The curves show straight-line times; "
               "the table above includes the detour in the worst case. Markers are for your pilot.")

    st.markdown(f"**Good days to leave** ({origin} to {dest}, your pilot {g_mid} g)")
    syn = synodic_days(period_days(STAR, O), period_days(STAR, Dst))
    T_ = mid["T"]
    half = max(int(0.3 * syn), 1)
    good = [i for i in range(half, len(T_) - half) if T_[i] <= T_[i - half:i + half + 1].min() + 1e-12]
    if good:
        st.dataframe(pd.DataFrame([{"Depart": when(float(mid["t0"][i])), "Flight (days)": round(float(T_[i]), 2),
                                    "Flight (hours)": round(float(T_[i]) * 24), "Planets apart (AU)": round(float(mid["dist_au"][i]), 2),
                                    "Top speed (km/s)": round(float(mid["v_peak_kms"][i]))} for i in good[:10]]),
                     hide_index=True, width="stretch")
        st.caption(f"The planets are closest again every {syn / 365.25:.2f} Earth years ({syn / DAY:.0f} Bayurindan days).")

    with st.expander("For comparison: a ship that only coasts (no wings)"):
        mu = K2 * STAR.mass_msun
        h = hohmann(O.a_au, Dst.a_au, mu)

        @st.cache_data(show_spinner="Working out the coasting flights...")
        def get_porkchop(star: Star, o: Body, d: Body, span: float):
            sy = synodic_days(period_days(star, o), period_days(star, d))
            hh = hohmann(o.a_au, d.a_au, K2 * star.mass_msun)
            dep = np.arange(-0.5 * sy, span + 0.5 * sy, 5.0)
            tof = np.arange(0.4 * hh["tof_days"], 2.0 * hh["tof_days"], 5.0)
            return porkchop(star, o, d, dep, tof)

        pc = get_porkchop(STAR, O, Dst, span_days)
        wins = find_windows(pc, syn, 0, span_days, 10)
        st.write(f"Without wings, a ship coasts on a curved path: about {h['tof_days']:.0f} days at best, "
                 f"with a window every {syn / 365.25:.1f} years, and it would need {h['vinf_dep'] + h['vinf_arr']:.1f} km/s of speed change.")
        if wins:
            st.dataframe(pd.DataFrame([{"Depart": when(w["dep_day"]), "Flight (days)": round(w["tof_days"]),
                                        "Speed change (km/s)": round(w["total"], 2)} for w in wins]), hide_index=True, width="stretch")


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
