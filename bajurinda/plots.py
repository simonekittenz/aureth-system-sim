from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .constants import R_SUN_KM
from .dynamics import StabilityResult, Track
from .model import System, angular_diameter_deg, stable_limit_km
from .sky import TideSeries, elongation, illuminated_fraction, phase_disc

COL = {"Liwaya": "#4cc9f0", "Nineya": "#f72585", "Sun": "#ffb703", "total": "#e9ecef"}


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
        items.append((m.name, ang, frac, el < np.pi, COL[m.name]))
    items.append(("Earth's Moon (reference)", angular_diameter_deg(1737.4, 384400), 1.0, True, "#868e96"))
    sun_ang = angular_diameter_deg(R_SUN_KM, float(np.linalg.norm(track.star_pos[i])))
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
