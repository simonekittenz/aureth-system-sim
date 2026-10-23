from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .aureth import COLORS, Body, Star, state


def _ellipse(star: Star, b: Body, n: int = 240):
    t = np.linspace(0, 2 * math.pi, n)
    a, e = b.a_au, b.ecc
    x, y = a * (np.cos(t) - e), a * math.sqrt(1 - e * e) * np.sin(t)
    return x, y


def orbit_figure(star: Star, bodies: list[Body], t: np.ndarray, pos: np.ndarray, trail: int = 25, max_frames: int = 140) -> go.Figure:
    names = [b.name for b in bodies]
    reach = max(b.a_au * (1 + b.ecc) for b in bodies) * 1.12
    fig = go.Figure()
    for b in bodies:
        x, y = _ellipse(star, b)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=COLORS.get(b.name, "#adb5bd"), width=1, dash="dot"),
                                 name=f"{b.name} orbit", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=18, color="#ffd166", line=dict(width=2, color="#f4a261")),
                             name="Aureth", hovertemplate="Aureth<extra></extra>"))
    idx = np.linspace(0, len(t) - 1, min(max_frames, len(t))).astype(int)

    def dyn(i):
        out = []
        for k, b in enumerate(bodies):
            lo = max(0, i - trail)
            out.append(go.Scatter(x=pos[lo:i + 1, k, 0], y=pos[lo:i + 1, k, 1], mode="lines",
                                  line=dict(color=COLORS.get(b.name, "#adb5bd"), width=3), showlegend=False, hoverinfo="skip"))
        for k, b in enumerate(bodies):
            out.append(go.Scatter(x=[pos[i, k, 0]], y=[pos[i, k, 1]], mode="markers+text", text=[b.name],
                                  textposition=("bottom center", "top center", "middle right")[k % 3],
                                  marker=dict(size=12, color=COLORS.get(b.name, "#adb5bd")), showlegend=False,
                                  hovertemplate=f"{b.name}<br>%{{x:.3f}}, %{{y:.3f}} AU<extra></extra>"))
        return out

    for tr in dyn(0):
        fig.add_trace(tr)
    base = len(fig.data) - 2 * len(bodies)
    nt = 2 * len(bodies)
    fig.frames = [go.Frame(data=dyn(i), traces=list(range(base, base + nt)), name=f"{t[i] / 365.25:.2f}") for i in idx]
    fig.update_layout(
        height=640, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-reach, reach], title="AU", zeroline=False),
        yaxis=dict(range=[-reach, reach], scaleanchor="x", title="AU", zeroline=False),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.08, buttons=[
            dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
            dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(currentvalue=dict(prefix="Earth year "), pad=dict(t=40),
                      steps=[dict(label=f.name, method="animate", args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for f in fig.frames])])
    return fig


def sky_figure(t: np.ndarray, sky: dict, year_days: float, opp: dict[str, np.ndarray]) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.6, 0.4],
                        subplot_titles=("Angle from the Sun in Bayurinda's sky (180 = opposition, 0 = behind the Sun)",
                                        "Distance from Bayurinda (AU)"))
    x = t / year_days
    for n, d in sky.items():
        c = COLORS.get(n, "#adb5bd")
        fig.add_trace(go.Scatter(x=x, y=d["elongation_deg"], name=n, line=dict(color=c)), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=d["distance_au"], name=n, line=dict(color=c), showlegend=False), 2, 1)
        if n in opp:
            fig.add_trace(go.Scatter(x=opp[n] / year_days, y=[180] * len(opp[n]), mode="markers", showlegend=False,
                                     marker=dict(symbol="star", size=11, color=c, line=dict(width=1, color="white")),
                                     hovertemplate=f"{n} opposition<br>year %{{x:.2f}}<extra></extra>"), 1, 1)
    fig.update_yaxes(range=[0, 190], dtick=45, row=1, col=1)
    fig.update_xaxes(title_text="Bayurindan years since the founding moment", row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.1))
    return fig


def porkchop_figure(pc: dict, year_days: float, windows: list[dict], budget: float | None, cap: float,
                    from_day: float, to_day: float, key: str = "total") -> go.Figure:
    z = np.where(np.isnan(pc[key]), cap, np.minimum(pc[key], cap))      # no usable transfer = effectively unaffordable
    keep = (pc["dep"] >= from_day) & (pc["dep"] <= to_day)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=pc["dep"][keep] / year_days, y=pc["tof"], z=z[keep].T, zmin=0, zmax=cap,
                             colorscale="Viridis_r", contours=dict(coloring="heatmap"), colorbar=dict(title="km/s"),
                             hovertemplate="year %{x:.2f}<br>flight %{y:.0f} d<br>%{z:.2f} km/s<extra></extra>", line=dict(width=0)))
    if budget:
        fig.add_trace(go.Contour(x=pc["dep"][keep] / year_days, y=pc["tof"], z=z[keep].T, showscale=False,
                                 contours=dict(coloring="lines", start=budget, end=budget, size=1),
                                 line=dict(color="white", width=2, dash="dash"), hoverinfo="skip", name="budget"))
    fig.add_trace(go.Scatter(x=[w["dep_day"] / year_days for w in windows], y=[w["tof_days"] for w in windows], mode="markers",
                             marker=dict(symbol="x", size=11, color="#ff006e", line=dict(width=2)), name="best of each window",
                             hovertemplate="best: year %{x:.2f}, %{y:.0f} d<extra></extra>"))
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Departure (Bayurindan years since the founding moment)", yaxis_title="Flight time (Earth days)",
                      legend=dict(orientation="h", y=1.08))
    return fig


def stability_figure(t: np.ndarray, a: np.ndarray, e: np.ndarray, names: list[str]) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Semi-major axis (AU)", "Eccentricity"))
    x = t / 365.25
    for k, n in enumerate(names):
        c = COLORS.get(n, "#adb5bd")
        fig.add_trace(go.Scatter(x=x, y=a[:, k], name=n, line=dict(color=c)), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=e[:, k], name=n, line=dict(color=c), showlegend=False), 2, 1)
    fig.update_xaxes(title_text="Earth years", row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.1))
    return fig
