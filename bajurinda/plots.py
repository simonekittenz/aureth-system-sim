from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .constants import MOON_CHART, MOON_RGB
from .dynamics import StabilityResult, Track
from .model import System, angular_diameter_deg, stable_limit_km
from .eras import ErasData
from .lunar_calendar import CalendarData
from .sky import TideSeries, elongation, illuminated_fraction, phase_disc

COL = {**MOON_CHART, "Sun": "#ffb703", "total": "#e9ecef"}


def _circle(r: float, n: int = 200):
    a = np.linspace(0, 2 * np.pi, n)
    return r * np.cos(a), r * np.sin(a)


def orbit_figure(system: System, track: Track, max_frames: int = 120) -> go.Figure:
    pl = system.planet
    names = [m.name for m in system.moons]
    pos = track.moon_pos / 1e3
    sun_dir = track.star_pos / np.linalg.norm(track.star_pos, axis=1, keepdims=True)
    lim = max(stable_limit_km(pl, m.ecc) for m in system.moons) / 1e3
    reach = max(float(np.abs(pos[:, :, :2]).max()), lim * 0.5) * 1.15

    fig = go.Figure()
    for k, n in enumerate(names):
        fig.add_trace(go.Scatter(x=pos[:, k, 0], y=pos[:, k, 1], mode="lines", name=f"{n} path",
                                 line=dict(color=COL[n], width=1), opacity=0.5))
    px_, py_ = _circle(pl.radius_km / 1e3)
    fig.add_trace(go.Scatter(x=px_, y=py_, fill="toself", mode="lines", name=pl.name,
                             line=dict(color="#2a9d8f"), fillcolor="#2a9d8f"))
    hx, hy = _circle(lim)
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines", name="Hill stability limit (single moon)",
                             line=dict(color="#868e96", dash="dot", width=1)))

    idx = np.linspace(0, len(track.t) - 1, min(max_frames, len(track.t))).astype(int)
    def dyn(i):
        return [
            go.Scatter(x=[pos[i, 0, 0]], y=[pos[i, 0, 1]], mode="markers",
                       marker=dict(size=10, color=COL[names[0]]), name=names[0], showlegend=False),
            go.Scatter(x=[pos[i, 1, 0]], y=[pos[i, 1, 1]], mode="markers",
                       marker=dict(size=14, color=COL[names[1]]), name=names[1], showlegend=False),
            go.Scatter(x=[0, sun_dir[i, 0] * reach], y=[0, sun_dir[i, 1] * reach], mode="lines+text",
                       text=["", "to Sun"], textposition="top center",
                       line=dict(color=COL["Sun"], width=2), name="Sun direction", showlegend=False),
        ]
    for tr_ in dyn(0):
        fig.add_trace(tr_)
    base = len(fig.data) - 3
    fig.frames = [
        go.Frame(data=dyn(i), traces=[base, base + 1, base + 2], name=f"{track.t[i]:.1f}")
        for i in idx
    ]
    fig.update_layout(
        height=620,
        xaxis=dict(title="x (thousand km)", range=[-reach, reach], zeroline=False),
        yaxis=dict(title="y (thousand km)", range=[-reach, reach], scaleanchor="x", zeroline=False),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.08, buttons=[
            dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True)]),
            dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
        ])],
        sliders=[dict(currentvalue=dict(prefix="day "), pad=dict(t=40),
                      steps=[dict(label=f.name, method="animate",
                                  args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for f in fig.frames])],
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def phase_figure(system: System, track: Track, unit_days: float, unit_label: str) -> go.Figure:
    x = track.t / unit_days
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Illuminated fraction (0 = new, 1 = full)", "Elongation from the Sun (deg)"))
    for k, m in enumerate(system.moons):
        frac = illuminated_fraction(track.moon_pos[:, k], track.star_pos)
        el = np.degrees(elongation(track.moon_pos[:, k], track.star_pos))
        fig.add_trace(go.Scatter(x=x, y=frac, name=m.name, line=dict(color=COL[m.name])), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=el % 360, name=m.name, mode="markers", marker=dict(size=3, color=COL[m.name]),
                                 showlegend=False), row=2, col=1)
    fig.update_yaxes(range=[0, 1.02], row=1, col=1)
    fig.update_yaxes(range=[0, 360], dtick=90, row=2, col=1)
    fig.update_xaxes(title_text=unit_label, row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.1))
    return fig


def size_figure(system: System, track: Track, unit_days: float, unit_label: str) -> go.Figure:
    """How wide each moon and the Sun look from the planet's centre over time (degrees)."""
    x = track.t / unit_days
    fig = go.Figure()
    for k, m in enumerate(system.moons):
        d = np.linalg.norm(track.moon_pos[:, k], axis=1)
        fig.add_trace(go.Scatter(x=x, y=np.degrees(2 * np.arctan(m.radius_km / d)), name=m.name, line=dict(color=COL[m.name])))
    sun = np.degrees(2 * np.arctan(system.planet.star_radius_km / np.linalg.norm(track.star_pos, axis=1)))
    fig.add_trace(go.Scatter(x=x, y=sun, name="Sun", line=dict(color=COL["Sun"], dash="dot")))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=unit_label, yaxis_title="looks this wide (degrees)",
                      legend=dict(orientation="h", y=1.1))
    return fig


def sky_discs_figure(system: System, track: Track, i: int) -> go.Figure:
    pl = system.planet
    fig = go.Figure()
    x = 0.0
    items = []
    for k, m in enumerate(system.moons):
        d = float(np.linalg.norm(track.moon_pos[i, k]))
        ang = angular_diameter_deg(m.radius_km, d)
        frac = float(illuminated_fraction(track.moon_pos[i:i + 1, k], track.star_pos[i:i + 1])[0])
        el = float(elongation(track.moon_pos[:, k], track.star_pos)[i]) % (2 * np.pi)
        items.append((m.name, ang, frac, el < np.pi, "rgb(%d,%d,%d)" % MOON_RGB.get(m.name, (230, 230, 230))))      # pale, to read on the dark background
    items.append(("Earth's Moon (reference)", angular_diameter_deg(1737.4, 384400), 1.0, True, "#868e96"))
    sun_ang = angular_diameter_deg(pl.star_radius_km, float(np.linalg.norm(track.star_pos[i])))
    items.append(("Sun", sun_ang, 1.0, True, COL["Sun"]))
    for name, ang, frac, waxing, col in items:
        r = ang / 2
        cx = x + r
        cxs, cys = _circle(r)
        fig.add_trace(go.Scatter(x=cx + cxs, y=cys, mode="lines", line=dict(color=col, width=1),
                                 fill="toself", fillcolor="rgba(30,30,30,0.9)", showlegend=False,
                                 hoverinfo="skip"))
        px_, py_ = phase_disc(frac, waxing, r, cx, 0.0)
        fig.add_trace(go.Scatter(x=px_, y=py_, mode="lines", fill="toself", fillcolor=col,
                                 line=dict(color=col, width=0), showlegend=False,
                                 hovertemplate=f"{name}<br>{ang:.2f} deg wide<br>{frac:.0%} lit<extra></extra>"))
        fig.add_annotation(x=cx, y=-r - 0.08, text=f"<b>{name}</b><br>{ang:.2f}°, {frac:.0%} lit" +
                           ("" if name.startswith(("Sun", "Earth")) else (" waxing" if waxing else " waning")),
                           showarrow=False, font=dict(size=11))
        x += ang + 0.25
    fig.update_layout(height=300, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#0b132b", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def altitude_figure(track: Track, alts: dict[str, np.ndarray], unit_days: float, unit_label: str,
                    lo: int, hi: int) -> go.Figure:
    fig = go.Figure()
    x = track.t[lo:hi] / unit_days
    for name, a in alts.items():
        fig.add_trace(go.Scatter(x=x, y=a[lo:hi], name=name, line=dict(color=COL[name])))
    fig.add_hline(y=0, line=dict(color="#868e96", dash="dot"))
    fig.update_layout(height=320, xaxis_title=unit_label, yaxis=dict(title="altitude (deg)", range=[-90, 90]),
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.12))
    return fig


def tide_figure(ts: TideSeries, track: Track, unit_days: float, unit_label: str) -> go.Figure:
    x = track.t / unit_days
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.4, 0.3, 0.3],
                        subplot_titles=("Equilibrium tide at the chosen site (m)",
                                        "Daily tidal range at the site (m)",
                                        "Strongest possible tide anywhere on the planet (m)"))
    fig.add_trace(go.Scatter(x=x, y=ts.liwaya, name="Liwaya", line=dict(color=COL["Liwaya"], width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=ts.nineya, name="Nineya", line=dict(color=COL["Nineya"], width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=ts.sun, name="Sun", line=dict(color=COL["Sun"], width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=ts.site_total, name="Combined", line=dict(color=COL["total"], width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=ts.site_range, name="Daily range", line=dict(color="#90be6d"), showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=ts.global_peak, name="Moons + Sun", line=dict(color=COL["total"]), showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=x, y=ts.global_peak_moons_only, name="Moons only", line=dict(color="#adb5bd", dash="dot"),
                             showlegend=False), 3, 1)
    fig.update_xaxes(title_text=unit_label, row=3, col=1)
    fig.update_layout(height=760, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.07))
    return fig


def stability_figure(system: System, res: StabilityResult) -> go.Figure:
    tr = res.track
    x = tr.t / 365.25
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Semi-major axis (thousand km, osculating)", "Eccentricity", "Inclination (deg)"))
    for k, m in enumerate(system.moons):
        c = COL[m.name]
        ok = np.isfinite(tr.a[:, k]) & (tr.a[:, k] > 0) & (tr.e[:, k] < 1)
        fig.add_trace(go.Scatter(x=x[ok], y=tr.a[ok, k] / 1e3, name=m.name, line=dict(color=c)), 1, 1)
        fig.add_trace(go.Scatter(x=x[ok], y=tr.e[ok, k], showlegend=False, line=dict(color=c)), 2, 1)
        fig.add_trace(go.Scatter(x=x[ok], y=tr.inc[ok, k], showlegend=False, line=dict(color=c)), 3, 1)
    fig.update_xaxes(title_text="years", row=3, col=1)
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.07))
    return fig


def calendar_figure(cal: CalendarData, moon_names: tuple[str, str]) -> go.Figure:
    D = cal.day_len
    fig = go.Figure()
    last_day = max((m["start_day"] + m["length_days"] for m in cal.months), default=cal.year_len)
    for r in cal.runs:
        fig.add_vrect(x0=r["start_day"] - 2, x1=r["end_day"] + 2, line_width=0,
                      fillcolor="rgba(255,183,3,0.22)" if r["kind"] == "full" else "rgba(131,56,236,0.22)", layer="below")
    for y in range(0, int(last_day // cal.year_len) + 2):
        fig.add_vline(x=y * cal.year_len + 1, line=dict(color="#868e96", dash="dot", width=1))
    rows = [(moon_names[1], 1, 1), (moon_names[0], 2, 0)]
    for name, y, k in rows:
        for kind, symbol, fill in (("full", "circle", COL[name]), ("new", "circle-open", COL[name])):
            if k == 1 and kind != cal.start_kind:
                continue                                            # Nineya: only the phase that starts a week
            ts = cal.events[(k, kind)]
            ts = ts[ts / D + 1 <= last_day]
            fig.add_trace(go.Scatter(
                x=ts / D + 1, y=[y] * len(ts), mode="markers", name=f"{name} {kind}",
                marker=dict(size=9 if k == 0 else 5, symbol=symbol, color=fill, line=dict(width=1.5, color=fill)),
                hovertemplate=f"{name} {kind} moon<br>day %{{x:.1f}}<extra></extra>"))
    for kind, colour in (("full", "#ffb703"), ("new", "#8338ec")):
        pts = [m for m in cal.months if m[f"double_{kind}"]]
        fig.add_trace(go.Scatter(
            x=[m[f"{kind}_t"] / D + 1 for m in pts], y=[3] * len(pts), mode="markers", name=f"double {kind}",
            marker=dict(size=13, symbol="star", color=colour),
            hovertemplate=f"double {kind}<br>day %{{x:.1f}}<extra></extra>"))
    fig.update_layout(
        height=330, margin=dict(l=70, r=10, t=10, b=10), legend=dict(orientation="h", y=1.18),
        xaxis=dict(title="Bajurindan day (dotted lines mark the start of each year)", range=[0, last_day + 5]),
        yaxis=dict(tickmode="array", tickvals=[1, 2, 3], ticktext=[moon_names[1], moon_names[0], "doubles"], range=[0.4, 3.6]))
    return fig


def eras_figure(data: ErasData, bands: list[tuple[float, float]], threshold_deg: float) -> go.Figure:
    D = data.calendar.day_len
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.42, 0.33, 0.25],
                        subplot_titles=("Overlaps of the two moons per Bayurindan year",
                                        "Angle between the two orbital planes (deg)",
                                        "Double-full and double-new months"))
    counts, edges = np.histogram(data.overlap_years, bins=np.arange(0, data.years + 1))
    fig.add_trace(go.Bar(x=edges[:-1] + 0.5, y=counts, marker_color="#4cc9f0", name="overlaps",
                         hovertemplate="year %{x:.0f}: %{y} overlaps<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=data.plane_years, y=data.mutual_deg, mode="lines", line=dict(color="#e9ecef", width=1.5),
                             name="plane angle", hovertemplate="year %{x:.1f}: %{y:.1f} deg<extra></extra>"), 2, 1)
    fig.add_hline(y=threshold_deg, line=dict(color="#ffb703", dash="dot", width=1), row=2, col=1)
    for kind, colour, y in (("full", "#ffb703", 1), ("new", "#8338ec", 0)):
        pts = [m for m in data.calendar.months if m[f"double_{kind}"]]
        fig.add_trace(go.Scatter(x=[m[f"{kind}_t"] / D / data.year_len for m in pts], y=[y] * len(pts), mode="markers",
                                 marker=dict(symbol="star", size=9, color=colour), name=f"double {kind}",
                                 hovertemplate=f"double {kind}<br>year %{{x:.2f}}<extra></extra>"), 3, 1)
    for a, b in bands:
        fig.add_vrect(x0=a, x1=b, fillcolor="rgba(255,183,3,0.16)", line_width=0, layer="below", row="all", col=1)
    fig.update_yaxes(tickvals=[0, 1], ticktext=["new", "full"], range=[-0.6, 1.6], row=3, col=1)
    fig.update_yaxes(rangemode="tozero", row=2, col=1)
    fig.update_xaxes(title_text="Bajurindan years", range=[0, data.years], row=3, col=1)
    fig.update_layout(height=700, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig


ECLIPSE_COL = {"total": "#ffb703", "hybrid": "#fb8500", "annular": "#8ecae6", "partial": "#adb5bd"}


def eclipse_figure(events: list[dict], names: tuple[str, str], years: float) -> go.Figure:
    fig = go.Figure()
    rows = [(names[0], "solar"), (names[1], "solar"), (names[0], "lunar"), (names[1], "lunar")]
    labels = [f"{m} {k}" for m, k in rows]
    for typ, colour in ECLIPSE_COL.items():
        for kind, symbol in (("solar", "circle"), ("lunar", "diamond")):
            pts = [(e["year"], labels.index(f"{e['moon']} {kind}")) for e in events
                   if e["type"] == typ and e["kind"] == kind and e["year"] <= years]
            if not pts:
                continue
            fig.add_trace(go.Scatter(x=[a for a, _ in pts], y=[b for _, b in pts], mode="markers",
                                     name=f"{kind} {typ}", marker=dict(size=7, color=colour, symbol=symbol),
                                     hovertemplate=f"{kind} {typ}<br>year %{{x:.2f}}<extra></extra>"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.2),
                      xaxis=dict(title="Bajurindan years", range=[0, years]),
                      yaxis=dict(tickmode="array", tickvals=list(range(4)), ticktext=labels, autorange="reversed"))
    return fig


def standstill_figure(env: dict, year_days: float, major_minor: bool = True) -> go.Figure:
    fig = go.Figure()
    for name, v in env.items():
        fig.add_trace(go.Scatter(x=v["t"] / year_days, y=v["amp"], name=name, mode="lines", line=dict(color=COL[name], width=2),
                                 hovertemplate=f"{name}<br>year %{{x:.1f}}: %{{y:.1f}} deg<extra></extra>"))
    fig.add_hline(y=16, line=dict(color="#ffb703", dash="dot", width=1), annotation_text="the Sun at the solstices (16 deg)",
                  annotation_position="top left")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.12),
                      xaxis_title="Bayurindan years since the founding moment", yaxis_title="Highest point reached each month (declination, deg)")
    return fig
