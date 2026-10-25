"""A polar star chart of the whole sky, one hemisphere at a time, with constellation sets drawn on it.

It uses the sky's own coordinates (ecliptic longitude and latitude, the same as data/aureth.yaml), so it does not depend on which planet you
stand on. The centre of the northern chart is the ecliptic's north pole, the outer ring is the ecliptic itself (latitude 0), and the chart runs
a little past it so figures that cross the ring are not cut off. Charts are drawn as seen from inside the sky, looking up.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go

from .starfield import MilkyWay, ecl_vec, load_sky, milky_way_glow

EDGE = 108.0          # how far from the pole the chart reaches, in degrees (90 is the ecliptic)
LABEL_R = 116.0
HEMI = {"north": 1, "south": -1}


def project(v: np.ndarray, hemi: str):
    """Unit vectors (..., 3) to chart coordinates (x, y) and the distance r from the pole, in degrees."""
    lat = np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))
    lon = np.radians(np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360)
    r = 90 - lat if hemi == "north" else 90 + lat
    sgn = 1 if hemi == "north" else -1        # longitude runs clockwise in the north chart and anti-clockwise in the south one
    return sgn * r * np.sin(lon), r * np.cos(lon), r


def hemisphere_of(lat_deg: float) -> str:
    return "north" if lat_deg >= 0 else "south"


def figure_centre(fig: dict, vec: dict) -> np.ndarray | None:
    members = [n for ln in fig["lines"] for n in ln if n in vec]
    if not members:
        return None
    c = sum(vec[m] for m in set(members))
    nc = np.linalg.norm(c)
    return c / nc if nc > 1e-6 else None


def _arc(va: np.ndarray, vb: np.ndarray, trim_deg: float = 1.1, n: int = 14):
    om = math.acos(float(np.clip(va @ vb, -1, 1)))
    if om < 1e-4 or om > 3.1:
        return None
    trim = min(0.4, math.radians(trim_deg) / om)
    tt = (trim + np.linspace(0, 1, n) * (1 - 2 * trim))[:, None]
    return (np.sin((1 - tt) * om) * va + np.sin(tt * om) * vb) / math.sin(om)


def _runs(x, y, keep):
    """Split a polyline wherever points fall outside the chart."""
    ox, oy, run = [], [], []
    for q in range(len(x)):
        if keep[q]:
            run.append(q)
        else:
            if len(run) > 1:
                ox += [float(x[i]) for i in run] + [None]; oy += [float(y[i]) for i in run] + [None]
            run = []
    if len(run) > 1:
        ox += [float(x[i]) for i in run] + [None]; oy += [float(y[i]) for i in run] + [None]
    return ox, oy


@lru_cache(maxsize=8)
def glow_grid(mw: MilkyWay, hemi: str, step: float = 1.0):
    xs = np.arange(-EDGE, EDGE + 1e-9, step)
    X, Y = np.meshgrid(xs, xs)
    r = np.hypot(X, Y)
    sgn = 1 if hemi == "north" else -1
    lon = np.arctan2(sgn * X, Y)
    lat = (90 - r) if hemi == "north" else (r - 90)
    v = np.stack([np.cos(np.radians(lat)) * np.cos(lon), np.cos(np.radians(lat)) * np.sin(lon), np.sin(np.radians(lat))], axis=-1)
    z = milky_way_glow(v, mw)
    z = z * np.clip((EDGE - r) / 8.0, 0, 1)          # fade out toward the rim so the edge is soft
    return xs, z


def star_map_figure(hemi: str, sets: list[dict], mw: MilkyWay, *, faint: tuple[np.ndarray, np.ndarray] | None = None, star_names: bool = True,
                    figure_names: bool = True, show_glow: bool = True, query: str = "", height: int = 820) -> tuple[go.Figure, list[dict]]:
    """Returns the figure and a list describing every figure (constellation) that matches `query` (all of them if it is empty)."""
    stars, _ = load_sky()
    vec = {s.name: ecl_vec(np.array(s.lon), np.array(s.lat)) for s in stars}
    q = query.strip().lower()
    fig = go.Figure()

    if show_glow:
        xs, z = glow_grid(mw, hemi)
        fig.add_trace(go.Heatmap(x=xs, y=xs, z=z, zsmooth="best", showscale=False, hoverinfo="skip",
                                 colorscale=[[0, "rgba(120,140,200,0)"], [1, "rgba(190,205,255,0.50)"]]))

    # rings and spokes
    a = np.linspace(0, 2 * np.pi, 240)
    for r, w, col in ((30, 1, "rgba(255,255,255,0.14)"), (60, 1, "rgba(255,255,255,0.14)"), (90, 2, "rgba(255,255,255,0.45)"), (EDGE, 1, "rgba(255,255,255,0.10)")):
        fig.add_trace(go.Scatter(x=r * np.cos(a), y=r * np.sin(a), mode="lines", line=dict(color=col, width=w), hoverinfo="skip", showlegend=False))
    sx, sy = [], []
    for lon in range(0, 360, 30):
        t = math.radians(lon); sgn = HEMI[hemi]
        sx += [0, sgn * EDGE * math.sin(t), None]; sy += [0, EDGE * math.cos(t), None]
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color="rgba(255,255,255,0.10)", width=1), hoverinfo="skip", showlegend=False))
    lab = [(HEMI[hemi] * LABEL_R * math.sin(math.radians(lon)), LABEL_R * math.cos(math.radians(lon)), f"{lon}°") for lon in range(0, 360, 30)]
    sign = "N" if hemi == "north" else "S"
    lab += [(3, r + 2, f"{90 - r}° {sign}" if r < 90 else "ecliptic") for r in (30, 60, 90)]
    fig.add_trace(go.Scatter(x=[l[0] for l in lab], y=[l[1] for l in lab], mode="text", text=[l[2] for l in lab], hoverinfo="skip", showlegend=False,
                             textfont=dict(color="rgba(255,255,255,0.45)", size=12)))

    # faint stars
    if faint is not None:
        fv, fm = faint
        x, y, r = project(fv, hemi)
        k = r <= EDGE
        fig.add_trace(go.Scatter(x=x[k], y=y[k], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(size=np.round(1.3 + (5.6 - fm[k]) * 0.35, 2), color="white", opacity=0.45)))

    # constellation lines and names
    matches, ann = [], []
    for si, cs in enumerate(sets):
        col = cs["color"]
        for fg in cs["figures"]:
            hit = bool(q) and q in fg["name"].lower()
            dim = bool(q) and not hit
            xs_, ys_ = [], []
            for na, nb in fg["lines"]:
                if na not in vec or nb not in vec:
                    continue
                pts = _arc(vec[na], vec[nb])
                if pts is None:
                    continue
                x, y, r = project(pts, hemi)
                ox, oy = _runs(x, y, r <= EDGE)
                xs_ += ox; ys_ += oy
            fig.add_trace(go.Scatter(x=xs_, y=ys_, mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(color=col, width=3.2 if hit else 1.6, dash="dash"), opacity=0.25 if dim else (1.0 if hit else 0.85)))
            c = figure_centre(fg, vec)
            if c is None:
                continue
            lat_c = math.degrees(math.asin(float(c[2])))
            lon_c = math.degrees(math.atan2(float(c[1]), float(c[0]))) % 360
            if hit or not q:
                bright = min((s for s in stars if s.name in {n for ln in fg["lines"] for n in ln}), key=lambda s: s.mag, default=None)
                matches.append({"Constellation": fg["name"], "Set": cs["name"], "Centre longitude": round(lon_c), "Centre latitude": round(lat_c),
                                "Chart": "Northern" if lat_c >= 0 else "Southern", "Stars": len({n for ln in fg["lines"] for n in ln}),
                                "Brightest star": bright.name if bright else ""})
            x, y, r = project(c, hemi)
            if figure_names and float(r) <= EDGE:
                ann.append(dict(x=float(x), y=float(y), text=f"<b>{fg['name']}</b>" if hit else fg["name"], showarrow=False,
                                font=dict(size=16 if hit else 12, color=col if hit else ("rgba(255,255,255,0.35)" if dim else "rgba(255,255,255,0.9)"))))

    # the bright stars
    x, y, r = project(np.array([ecl_vec(np.array(s.lon), np.array(s.lat)) for s in stars]), hemi)
    k = r <= EDGE
    ks = [s for s, ok in zip(stars, k) if ok]
    star_hit = [bool(q) and q in s.name.lower() for s in ks]
    fig.add_trace(go.Scatter(x=x[k], y=y[k], mode="markers", showlegend=False,
                             marker=dict(size=[round(float(np.clip(8.5 - 2.0 * s.mag, 4, 15)), 1) for s in ks], color=[s.color for s in ks], line=dict(width=0)),
                             hovertemplate="%{customdata[0]}<br>brightness %{customdata[1]}<br>longitude %{customdata[2]:.1f}°, latitude %{customdata[3]:+.1f}°<extra></extra>",
                             customdata=[[s.name, s.mag, s.lon, s.lat] for s in ks]))
    if star_names:
        fig.add_trace(go.Scatter(x=x[k], y=y[k], mode="text", showlegend=False, hoverinfo="skip", textposition="bottom right",
                                 text=[s.name if s.label else "" for s in ks], textfont=dict(color="rgba(255,255,255,0.7)", size=10)))
    if any(star_hit):
        fig.add_trace(go.Scatter(x=[v for v, h in zip(x[k], star_hit) if h], y=[v for v, h in zip(y[k], star_hit) if h], mode="markers+text", showlegend=False,
                                 hoverinfo="skip", marker=dict(symbol="circle-open", size=26, color="#ff4d8d", line=dict(width=3)),
                                 text=[s.name for s, h in zip(ks, star_hit) if h], textposition="top center", textfont=dict(color="#ff9ec4", size=14)))
        for s, h in zip(ks, star_hit):
            if h:
                matches.append({"Constellation": f"(star) {s.name}", "Set": "", "Centre longitude": round(s.lon), "Centre latitude": round(s.lat),
                                "Chart": "Northern" if s.lat >= 0 else "Southern", "Stars": 1, "Brightest star": s.name})
    pad = 4
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="rgb(10,13,28)", paper_bgcolor="rgba(0,0,0,0)", annotations=ann,
                      xaxis=dict(range=[-LABEL_R - pad, LABEL_R + pad], visible=False, fixedrange=False),
                      yaxis=dict(range=[-LABEL_R - pad, LABEL_R + pad], visible=False, scaleanchor="x", fixedrange=False), hoverlabel=dict(font_size=13))
    return fig, matches


def find(sets: list[dict], query: str) -> list[dict]:
    """Constellations (figures) and bright stars whose names contain `query`, with where they sit on the sky."""
    q = query.strip().lower()
    if not q:
        return []
    stars, _ = load_sky()
    vec = {s.name: ecl_vec(np.array(s.lon), np.array(s.lat)) for s in stars}
    out = []
    for cs in sets:
        for fg in cs["figures"]:
            if q in fg["name"].lower():
                c = figure_centre(fg, vec)
                if c is not None:
                    out.append({"kind": "constellation", "name": fg["name"], "lat": math.degrees(math.asin(float(c[2])))})
    for s in stars:
        if q in s.name.lower():
            out.append({"kind": "star", "name": s.name, "lat": s.lat})
    return out
