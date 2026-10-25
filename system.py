from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from bajurinda import system_plots as sp
from bajurinda.aureth import (
    AU_KM, COLORS, K2, Body, Ephemeris, Star, conjunction_minima, crossing_times, grand_alignments,
    nbody_track, pair_table, period_days, planet_table, sky_from, synodic_days,
)
from bajurinda.cycles import axial_precession, great_cycle, moon_cycles, orbit_secular, plane_cycle_years, planet_repeat
from bajurinda.flights import route_extremes, scan_route
from bajurinda.constellations import load_constellations
from bajurinda.starfield import faint_stars, load_faint_settings, load_sky
from bajurinda.starmap import find as find_on_map, hemisphere_of, star_map_figure
from bajurinda.world import load_world
from bajurinda.worldstate import ensure_state, orbit_editor, remember, star_editor, world_from_state
from bajurinda.transfers import find_windows, hohmann, porkchop

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = load_world()
ss = st.session_state
ensure_state(BASE)
for _k, _v in ss.get("_map_saved", {}).items():       # the star map's settings survive a visit to the other page
    if _k not in ss:
        ss[_k] = _v
for _k in [k for k in ss.keys() if k.startswith("m_")]:
    ss[_k] = ss[_k]

with st.sidebar:
    st.subheader("Aureth system")
    st.caption("The same star and planets as on the Planet view page: a change made here shows there too.")
    with st.expander("Star (Aureth)"):
        star_editor(BASE)
    for p_ in BASE.planets:
        with st.expander(p_.name + ("  (some values unsettled)" if p_.placeholder_note else "")):
            orbit_editor(p_, BASE)
remember()

world = world_from_state(BASE)
STAR = world.star
BODIES = world.bodies()
BY = {b.name: b for b in BODIES}
NAMES = [b.name for b in BODIES]
BAYURINDA = world.planet("Bayurinda")
YEAR = BAYURINDA.year_days                          # Bayurindan year, Earth days (440 solar days)
DAY = BAYURINDA.solar_day_hours / 24                # Bayurindan solar day, Earth days


def when(t_days: float) -> str:
    y = int(math.floor(t_days / YEAR)) + 1
    d = int((t_days % YEAR) / DAY) + 1
    return f"year {y}, day {d}"


@st.cache_data(show_spinner="Following the planets...")
def get_eph(star: Star, bodies: tuple[Body, ...], span_days: float, step: float = 1.0):
    return Ephemeris(star, list(bodies), span_days, step)


st.title("Aureth system")
st.caption("Time is counted from the founding moment (the double full moon), when the star, Bayurinda, Ashara and Tarnic "
           f"start in a line. Dates read 'year, day' in Bayurindan years of {BAYURINDA.year_solar_days:.0f} solar days. Orbits are flat and coplanar "
           "(a placeholder), and Ashara's eccentricity is a placeholder. Planet positions come from one N-body simulation, the same one the Planet view uses.")

TAB_NAMES = ["Orbits", "Cycles", "Sky from Bayurinda", "Constellation map", "Transfers", "Long cycles", "Stability"]
T = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))


# ------------------------------------------------------------------ Orbits
with T["Orbits"]:
    years = st.select_slider("Years to animate (Earth years)", [2, 5, 10, 20], value=5)

    @st.cache_data(show_spinner="Integrating the planets...")
    def get_orbits(star: Star, bodies: tuple[Body, ...], span_days: float):
        t = np.arange(0, span_days, 5.0)
        return nbody_track(star, list(bodies), t)

    track = get_orbits(STAR, tuple(BODIES), years * 365.25)
    st.plotly_chart(sp.orbit_figure(STAR, BODIES, track["t"], track["pos"]), width="stretch")
    st.caption("N-body integration (REBOUND). Dotted lines are the orbits; coloured tails show recent motion.")


# ------------------------------------------------------------------ Cycles
with T["Cycles"]:
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
    al = grand_alignments(STAR, BODIES, g_years, g_tol, eph=get_eph(STAR, tuple(BODIES), g_years * 365.25 + 10, 2.0))
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
with T["Sky from Bayurinda"]:
    st.subheader("What Bayurinda's sky shows")
    s_years = st.select_slider("Years to look at (Earth years)", [5, 10, 20, 40, 80], value=20, key="sky_years")
    t = np.arange(0, s_years * 365.25, 1.0)
    pos = get_eph(STAR, tuple(BODIES), s_years * 365.25 + 5).positions(t)
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


# ------------------------------------------------------------------ Constellation map
with T["Constellation map"]:
    st.subheader("Constellation map")
    st.write("The whole sky as two round charts, one for each side of the plane the planets orbit in. It does not depend on which planet you stand on, "
             "so it is the finder for the constellation sets and the bright stars. Positions are ecliptic longitude (around the edge) and latitude "
             "(the rings): the northern chart is centred on the ecliptic's north pole and the bold ring is the ecliptic itself. They are drawn as you would see "
             "them looking up.")
    sets_all = ss.get("_con_work") or load_constellations()
    set_names_ = [c["name"] for c in sets_all]
    if "m_sets" not in ss:
        ss["m_sets"] = [n for n in ss.get("v_con_show", []) if n in set_names_] or set_names_[:1]
    ss["m_sets"] = [n for n in ss["m_sets"] if n in set_names_]
    for _k, _v in {"m_hemi": "north", "m_query": "", "m_figs": True, "m_stars": False, "m_faint": True, "m_glow": True, "m_pick": True,
                     "m_pv_on": False, "m_pv_lon": 100.0, "m_pv_lat": 20.0, "m_pv_mag": 1.0}.items():
        if _k not in ss:
            ss[_k] = _v

    def jump_to_match():
        hits = find_on_map(ss.get("_con_work") or load_constellations(), ss["m_query"])
        if hits:
            ss["m_hemi"] = hemisphere_of(hits[0]["lat"])

    q1, q2 = st.columns([2, 3])
    q1.text_input("Find a constellation or star", key="m_query", placeholder="Search constellation...", on_change=jump_to_match,
                  help="Type part of a name. The matching constellation is drawn thick and boxed, the rest fade, and the chart switches to the side it is on.")
    q2.multiselect("Show these constellation sets", set_names_, key="m_sets",
                   help="The sets come from the Constellation sets box on the Planet view page (unsaved edits show here too).")
    o1, o2, o3, o4, o5, o6 = st.columns([1.3, 1, 1, 1, 1, 1.3])
    o1.radio("Chart", ["north", "south"], key="m_hemi", horizontal=True, format_func=lambda h: "Northern" if h == "north" else "Southern")
    o2.checkbox("Constellation names", key="m_figs")
    o3.checkbox("Star names", key="m_stars")
    o4.checkbox("Faint stars", key="m_faint")
    o5.checkbox("Milky Way", key="m_glow")
    o6.checkbox("Pointer coordinates", key="m_pick",
                help="Hover over the chart to read the ecliptic longitude and latitude under the pointer (the numbers the stars use in data/aureth.yaml), and the nearest named star. "
                     "While it is on, hovering a star shows the readout instead of the star's own tooltip.")
    _preview = None
    if ss["m_pick"]:
        with st.expander("Preview a star at coordinates you choose", expanded=False):
            st.caption("Read a spot off the chart, type it here, and see where a star would sit before you edit data/aureth.yaml.")
            pc1, pc2, pc3, pc4 = st.columns(4)
            _pv_on = pc1.checkbox("Show the preview", key="m_pv_on")
            _pv_lon = pc2.number_input("Ecliptic longitude (°)", 0.0, 360.0, step=0.5, key="m_pv_lon")
            _pv_lat = pc3.number_input("Ecliptic latitude (°)", -90.0, 90.0, step=0.5, key="m_pv_lat")
            _pv_mag = pc4.number_input("Brightness (magnitude)", -2.0, 6.0, step=0.1, key="m_pv_mag", help="Smaller is brighter.")
            st.code(f'- {{name: "New star", lon: {_pv_lon:.1f}, lat: {_pv_lat:.1f}, mag: {_pv_mag:.1f}, color: "#ffffff", label: true}}', language="yaml")
            st.caption("Paste that line into the `stars` list under `sky` in data/aureth.yaml, change the name and colour, then press Reload data files.")
            if _pv_on:
                _preview = {"lon": float(_pv_lon), "lat": float(_pv_lat), "mag": float(_pv_mag), "name": "preview"}

    _mw = load_sky()[1]
    _fs = load_faint_settings()
    _faint = faint_stars(_mw, _fs["n_band"], _fs["n_field"], int(ss.get("v_sky_seed", _fs["seed"]))) if ss["m_faint"] else None
    _shown = [c for c in sets_all if c["name"] in ss["m_sets"]]
    _fig, _rows = star_map_figure(ss["m_hemi"], _shown, _mw, faint=_faint, star_names=ss["m_stars"], figure_names=ss["m_figs"],
                                  show_glow=ss["m_glow"], query=ss["m_query"], height=820, pick=ss["m_pick"], preview=_preview)
    st.plotly_chart(_fig, width="stretch", config={"displaylogo": False, "toImageButtonOptions": {"width": 1600, "height": 1600, "scale": 1}})
    if ss["m_query"].strip():
        other = [r for r in _rows if r["Chart"] != ("Northern" if ss["m_hemi"] == "north" else "Southern")]
        st.markdown(f"**Matches for '{ss['m_query'].strip()}'**" + (f" ({len(other)} on the other chart)" if other else ""))
        if not _rows:
            st.info("Nothing matches. Check the spelling, or that the constellation's set is ticked in 'Show these constellation sets'.")
    else:
        st.markdown("**Constellations in the shown sets**")
    if _rows:
        st.dataframe(pd.DataFrame(_rows), hide_index=True, width="stretch", height=min(380, 38 + 35 * len(_rows)))
    st.caption("Each dashed line follows the shortest path between two stars on the sky, so it curves on a flat chart. Hover over a star for its name and position. "
               "The stars, their names and the galaxy are placeholders you can edit in data/aureth.yaml.")
    ss["_map_saved"] = {k: v for k, v in ss.items() if k.startswith("m_")}


# ------------------------------------------------------------------ Transfers
with T["Transfers"]:
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

    EPH = get_eph(STAR, tuple(BODIES), span_days + 400.0)

    @st.cache_data(show_spinner="Working out the flights...")
    def get_extremes(star: Star, bodies: tuple[Body, ...], span: float, keep_au: float):
        out = {}
        by = {b.name: b for b in bodies}
        for o, d in (("Bayurinda", "Ashara"), ("Ashara", "Tarnic"), ("Bayurinda", "Tarnic")):
            for g in (G_NOV, G_EXP):
                out[(o, d, g)] = route_extremes(star, by[o], by[d], g, span, keep_au, eph=EPH)
        return out

    @st.cache_data(show_spinner="Scanning departure dates...")
    def get_scan(star: Star, o: Body, d: Body, g: float, span: float, keep_au: float):
        return scan_route(star, o, d, g, span, keep_au, eph=EPH)

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


# ------------------------------------------------------------------ Long cycles
with T["Long cycles"]:
    st.subheader("Longer cycles for Bayurinda")
    st.write("Cycles of different lengths, from the moons' months up to the slow wobble of the planet's axis. Moons are the "
             "selected Design B. Measuring takes about 20 seconds because it follows the moons for 250 years and the planets for 400,000.")
    sig = tuple(BODIES)
    if st.button("Measure the long cycles"):
        with st.spinner("Following the moons and planets..."):
            sysm = world.system("Bayurinda")
            mc = moon_cycles(sysm)
            pcy = plane_cycle_years(sysm)
            rep = planet_repeat(STAR, BODIES)
            als = grand_alignments(STAR, BODIES, 1000, 12, eph=get_eph(STAR, tuple(BODIES), 1000 * 365.25 + 10, 2.0))
            gaps = np.diff([a["t"] for a in als]) / YEAR if len(als) > 1 else np.array([np.nan])
            ax = axial_precession(sysm)
            sec = orbit_secular(STAR, BODIES)
            lw = mc["Liwaya"]
            gp = {"planets": rep["years"], "planes": pcy, "eclipses": lw["cycle_years"]} if rep and pcy else None
            gc = (great_cycle(gp) or great_cycle(gp, tol=0.01)) if gp else None      # a strict fit first, then a looser one
            st.session_state["sys_cycles"] = dict(sig=sig, mc=mc, plane=pcy, rep=rep, align=float(np.nanmean(gaps)), ax=ax, sec=sec, gc=gc)
    cy = st.session_state.get("sys_cycles")
    if cy is None:
        st.info("Press the button to measure.")
    else:
        if cy["sig"] != sig:
            st.warning("The planets have changed since this was measured. Press the button again.")
        lw, nn = cy["mc"]["Liwaya"], cy["mc"]["Nineya"]
        S_L, S_N = lw["synodic"], nn["synodic"]
        q = round(S_L / S_N)
        dbl = 1 / abs(1 / S_N - q / S_L) / YEAR
        e_by = cy["sec"]["period_earth_years"] * 365.25 / YEAR
        rows = [
            dict(name="Eclipse seasons (twice per eclipse year)", years=lw["eclipse_year"] / YEAR, group="moons",
                 what=f"The Sun passes Liwaya's orbital nodes every {lw['eclipse_year']:.0f} days", basis="measured"),
            dict(name="Double-full moons", years=dbl, group="moons",
                 what="Liwaya and Nineya are full together again", basis="measured"),
            dict(name="Liwaya eclipse cycle", years=lw["cycle_years"], group="moons",
                 what=f"{lw['cycle_months']} months = {lw['cycle_draconic']} node-to-node months = {lw['cycle_anomalistic']} "
                      f"closest-to-closest months, matching to {lw['cycle_off_days']:.2f} days: eclipses repeat the same way",
                 basis="measured"),
            dict(name="Liwaya's orbit turns around", years=lw["node_regression_years"], group="moons",
                 what="Its orbital nodes go once round the sky", basis="measured"),
            dict(name="Grand alignments", years=cy["align"], group="planets",
                 what="All three planets within 12 degrees, seen from Aureth (average gap)", basis="measured; depends on Ashara's distance"),
            dict(name="Moons' planes line up", years=cy["plane"], group="moons",
                 what="The two moons' orbital planes coincide; overlaps of the moons peak", basis="measured"),
        ]
        if cy["rep"]:
            r = cy["rep"]
            rows.append(dict(name="Whole planet system repeats", years=r["years"], group="planets",
                             what=f"{r['ba']} Bayurinda-Ashara cycles = {r['bt']} Bayurinda-Tarnic = {r['at']} Ashara-Tarnic",
                             basis="exact orbits; depends on Ashara's distance"))
        if cy["gc"]:
            g = cy["gc"]
            rows.append(dict(name="THE GREAT CYCLE", years=g["years"], group="combined",
                             what=f"{g['counts']['planets']} planet repeats = {g['counts']['planes']} moon-plane cycles = "
                                  f"{g['counts']['eclipses']} eclipse cycles (agree to {g['spread'] * 100:.2f}%)",
                             basis="combination of the three above"))
        rows.append(dict(name="Axis wobble (precession)", years=cy["ax"]["years"], group="slow",
                         what=f"The spin axis points at a new 'pole star'; the moons do {1 - cy['ax']['shares']['Aureth']:.0%} of the pulling",
                         basis="formula; moment of inertia assumed (+-15%)"))
        rows.append(dict(name="Orbit swells and shrinks", years=e_by, group="slow",
                         what=f"Bayurinda's eccentricity runs {cy['sec']['e_min']:.3f} to {cy['sec']['e_max']:.3f} (small: the other planets are light)",
                         basis="N-body; depends on Ashara's mass and orbit"))
        rows.sort(key=lambda r: r["years"])
        for r in rows:
            r["label"] = f"{r['years']:,.1f}" if r["years"] < 100 else f"{r['years']:,.0f}"
        st.plotly_chart(sp.cycle_ladder_figure(rows), width="stretch")
        st.dataframe(pd.DataFrame([{"Cycle": r["name"], "Bayurindan years": round(r["years"], 2 if r["years"] < 100 else 0),
                                    "Earth years": round(r["years"] * YEAR / 365.25, 2 if r["years"] < 100 else 0),
                                    "What happens": r["what"], "How it was found": r["basis"]} for r in rows]),
                     hide_index=True, width="stretch")
        if cy["gc"]:
            g = cy["gc"]
            st.markdown(f"**The Great Cycle, about {g['years']:.0f} Bayurindan years.** After this long the moons' planes, the planets' arrangement "
                        "and Liwaya's eclipse pattern all return together. With this many cycles to combine, near-fits turn up by chance now and then, "
                        "so treat it as a pleasing coincidence of the current numbers rather than a law. If you want it exact, "
                        "Ashara's distance is the knob to turn.")
        st.markdown("**Cycles of the star itself** are not computed here; Aureth's activity is a design choice. For scale, the Sun's magnetic "
                    "cycle is 11 years (22 for its polarity), with slower swings near 90, 200 and 2,300 years.")
        star_cycle = st.number_input("Aureth's activity cycle (Bayurindan years)", 3.0, 40.0, round(lw["cycle_years"], 2), step=0.1)
        if cy["gc"]:
            n = cy["gc"]["years"] / star_cycle
            st.caption(f"{n:.2f} of those fit in the Great Cycle" + (" (a whole number)" if abs(n - round(n)) < 0.05 else "")
                       + f"; one is {star_cycle / lw['cycle_years']:.2f} Liwaya eclipse cycles and {star_cycle / cy['plane']:.2f} moon-plane cycles. "
                         "The default sets it equal to the eclipse cycle.")


# ------------------------------------------------------------------ Stability
with T["Stability"]:
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
