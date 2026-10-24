from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from .aureth import COLORS, load_system, state
from .dynamics import build_simulation
from .model import System, angular_diameter_deg
from .sky import illuminated_fraction, phase_disc, spin_axis, surface_normal

SKY_STOPS = [(10, (122, 184, 255)), (3, (240, 170, 120)), (0, (224, 140, 100)), (-6, (61, 79, 143)),
             (-12, (29, 39, 80)), (-18, (7, 11, 26))]
MOON_TINT = {"Liwaya": (232, 241, 255), "Nineya": (255, 214, 230)}
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def sky_rgb(sun_alt: float) -> tuple[int, int, int]:
    if sun_alt >= SKY_STOPS[0][0]:
        return SKY_STOPS[0][1]
    if sun_alt <= SKY_STOPS[-1][0]:
        return SKY_STOPS[-1][1]
    for (a0, c0), (a1, c1) in zip(SKY_STOPS, SKY_STOPS[1:]):
        if a1 <= sun_alt <= a0:
            f = (a0 - sun_alt) / (a0 - a1)
            return tuple(int(c0[i] + f * (c1[i] - c0[i])) for i in range(3))
    return SKY_STOPS[-1][1]


def twilight_name(sun_alt: float) -> str:
    if sun_alt > 0:
        return "day"
    if sun_alt > -6:
        return "civil twilight"
    if sun_alt > -12:
        return "nautical twilight"
    if sun_alt > -18:
        return "astronomical twilight"
    return "night"


def star_field(n: int = 380, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Decorative background stars (not canon): random directions and sizes, fixed in inertial space."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    size = 1.2 + 3.2 * rng.random(n) ** 7
    return v, size


def moons_at(system: System, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Moon positions relative to the planet (n, 2, 3) and the star relative to the planet (n, 3), in km, at any times
    (before or after the founding moment)."""
    times = np.asarray(times, float)
    mp = np.zeros((len(times), 2, 3))
    sp = np.zeros((len(times), 3))
    for sign in (1, -1):
        idx = np.where(times >= 0)[0] if sign == 1 else np.where(times < 0)[0]
        if not len(idx):
            continue
        order = idx[np.argsort(sign * times[idx])]
        sim = build_simulation(system)
        for i in order:
            sim.integrate(float(times[i]))
            ps = sim.particles
            pl = np.array([ps[1].x, ps[1].y, ps[1].z])
            sp[i] = np.array([ps[0].x, ps[0].y, ps[0].z]) - pl
            for k in range(2):
                mp[i, k] = np.array([ps[2 + k].x, ps[2 + k].y, ps[2 + k].z]) - pl
    return mp, sp


@dataclass
class SkyData:
    t: np.ndarray                   # Earth days since the founding moment
    clock: np.ndarray               # local mean solar time of day, hours
    labels: list                    # "year 3, day 120, 04:30" for each frame
    sun: dict                       # alt, az (deg), rad (deg), probe (alt, az) for nothing
    moons: dict                     # name -> dict(alt, az, rad, k, limb_alt, limb_az, dist_km)
    planets: dict                   # name -> dict(alt, az)
    stars_alt: np.ndarray           # (frames, nstars)
    stars_az: np.ndarray
    star_size: np.ndarray
    lat: float
    lon: float


def _altaz(v, U, E, N):
    alt = np.degrees(np.arcsin(np.clip(np.einsum("ij,ij->i", v, U), -1, 1)))
    az = np.degrees(np.arctan2(np.einsum("ij,ij->i", v, E), np.einsum("ij,ij->i", v, N))) % 360
    return alt, az


def compute_sky(system: System, lat: float, lon: float, day_total: float, start_hour: float = 0.0,
                span_hours: float = 19.0, step_hours: float = 0.5) -> SkyData:
    """The sky from (lat, lon) at a run of moments, in local mean solar time (a day is 19 hours).

    `day_total` counts days since the founding moment, which is local midnight on day 0 at longitude 0. The run starts
    `start_hour` hours into that day and lasts `span_hours`, with one frame every `step_hours`."""
    pl = system.planet
    D = pl.local_day_days
    lat = max(-89.5, min(89.5, lat))
    n = int(round(span_hours / step_hours)) + 1
    hours = start_hour + step_hours * np.arange(n)
    days = day_total + hours / 19.0
    t = (days - lon / 360.0) * D
    year_len = pl.year_solar_days
    days_r = np.round(days, 7)                              # avoid 0.99999999 showing as the end of the previous day
    clock = (days_r - np.floor(days_r)) * 19.0
    labels = [f"year {int(np.floor(dd / year_len)) + 1}, day {int(np.floor(dd % year_len)) + 1}, "
              f"{int(c):02d}:{int(round((c % 1) * 60)) % 60:02d}" for dd, c in zip(days_r, clock)]
    mp, sp = moons_at(system, t)
    s = spin_axis(pl)
    U = surface_normal(pl, lat, lon, t)
    E = np.cross(s, U)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    N = np.cross(U, E)
    obs = pl.radius_km * U                                   # the observer stands on the surface, not at the centre

    sdir = sp / np.linalg.norm(sp, axis=1, keepdims=True)
    s_alt, s_az = _altaz(sdir, U, E, N)
    sun = {"alt": s_alt, "az": s_az, "rad": np.degrees(np.arctan(pl.star_radius_km / np.linalg.norm(sp, axis=1)))}

    moons = {}
    for k, m in enumerate(system.moons):
        v = mp[:, k] - obs
        d = np.linalg.norm(v, axis=1)
        u = v / d[:, None]
        alt, az = _altaz(u, U, E, N)
        w = sdir - np.einsum("ij,ij->i", sdir, u)[:, None] * u            # direction on the sky toward the Sun
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        probe = u + 0.01 * w
        probe /= np.linalg.norm(probe, axis=1, keepdims=True)
        l_alt, l_az = _altaz(probe, U, E, N)
        moons[m.name] = {"alt": alt, "az": az, "rad": np.degrees(np.arctan(m.radius_km / d)),
                         "k": illuminated_fraction(mp[:, k], sp), "limb_alt": l_alt, "limb_az": l_az, "dist_km": d}

    star, bodies = load_system()
    pos = {b.name: state(star, b, t)[0] for b in bodies}
    planets = {}
    for b in bodies:
        if b.name == "Bayurinda":
            continue
        v = pos[b.name] - pos["Bayurinda"]
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        a_, z_ = _altaz(v, U, E, N)
        planets[b.name] = {"alt": a_, "az": z_}

    sv, ssize = star_field()
    st_alt = np.zeros((len(t), len(sv)))
    st_az = np.zeros_like(st_alt)
    for i in range(len(t)):
        a_ = np.degrees(np.arcsin(np.clip(sv @ U[i], -1, 1)))
        z_ = np.degrees(np.arctan2(sv @ E[i], sv @ N[i])) % 360
        st_alt[i], st_az[i] = a_, z_
    return SkyData(t, clock, labels, sun, moons, planets, st_alt, st_az, ssize, lat, lon)


# ------------------------------------------------------------------ drawing

class Projection:
    def __init__(self, view: str, center_az: float):
        self.view, self.c = view, center_az

    def xy(self, alt, az):
        alt, az = np.asarray(alt, float), np.asarray(az, float)
        if self.view == "dome":
            r = 90.0 - alt
            return -r * np.sin(np.radians(az)), r * np.cos(np.radians(az))
        rel = (az - self.c + 180) % 360 - 180
        return rel, alt


def _circle(cx, cy, r, n=28):
    a = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(a), cy + r * np.sin(a)


def _copies(xs, ys, view):
    if view == "dome":
        return list(xs), list(ys)
    ox, oy = [], []
    for off in (-360, 0, 360):
        ox += [x + off for x in xs] + [None]
        oy += list(ys) + [None]
    return ox, oy


def _rgba(rgb, a):
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{a})"


def _frame_traces(d: SkyData, i: int, proj: Projection, scale: float, show_stars: bool):
    view = proj.view
    sun_alt = float(d.sun["alt"][i])
    sky = _rgba(sky_rgb(sun_alt), 1.0)
    traces = {}
    if view == "dome":
        a = np.linspace(0, 2 * np.pi, 120)
        traces[0] = dict(x=list(90 * np.cos(a)), y=list(90 * np.sin(a)), fillcolor=sky)
    else:
        traces[0] = dict(x=[-540, 540, 540, -540, -540], y=[0, 0, 120, 120, 0], fillcolor=sky)

    vis = max(0.0, min(1.0, (-sun_alt - 3) / 12))
    if show_stars and vis > 0:
        up = d.stars_alt[i] > 0
        x, y = proj.xy(d.stars_alt[i][up], d.stars_az[i][up])
        traces[1] = dict(x=list(x), y=list(y), marker=dict(size=list(d.star_size[up]), color="white", opacity=vis * 0.85))
    else:
        traces[1] = dict(x=[], y=[], marker=dict(size=[], color="white", opacity=0))

    def place(alt, az, rad_deg, dx=0.0):
        x, y = proj.xy(alt, az)
        return float(x), float(y), rad_deg * scale

    cx, cy, r = place(d.sun["alt"][i], d.sun["az"][i], d.sun["rad"][i])
    op = 1.0 if sun_alt > 0 else 0.35
    for key, mult, alpha in ((4, 6.0, 0.10), (5, 2.5, 0.22)):
        gx, gy = _circle(cx, cy, r * mult)
        x, y = _copies(gx, gy, view)
        traces[key] = dict(x=x, y=y, fillcolor=f"rgba(255,209,102,{alpha})", opacity=op)
    gx, gy = _circle(cx, cy, r)
    x, y = _copies(gx, gy, view)
    traces[6] = dict(x=x, y=y, fillcolor="rgb(255,222,130)", opacity=op)
    lx, ly = proj.xy(d.sun["alt"][i], d.sun["az"][i])
    traces[7] = dict(x=[float(lx)], y=[float(ly) + r * 3.2 + 1.5], text=["Aureth" + ("" if sun_alt > 0 else " (below the horizon)")], opacity=op)

    base = 8
    for name, mo in d.moons.items():
        alt, az = float(mo["alt"][i]), float(mo["az"][i])
        cx, cy, r = place(alt, az, mo["rad"][i])
        op = 1.0 if alt > 0 else 0.35
        gx, gy = _circle(cx, cy, r)
        x, y = _copies(gx, gy, view)
        traces[base] = dict(x=x, y=y, fillcolor="rgb(38,42,58)", opacity=op, line=dict(color=_rgba(MOON_TINT[name], 0.6), width=1))
        # lit part, turned so that the bright side faces the Sun
        px, py = proj.xy(mo["limb_alt"][i], mo["limb_az"][i])
        phi = math.atan2(float(py) - cy, float(px) - cx) if view != "panorama" else math.atan2(float(py) - cy, ((float(px) - cx + 180) % 360) - 180)
        lx_, ly_ = phase_disc(float(mo["k"][i]), True, r, cx, cy)
        rx = cx + (lx_ - cx) * math.cos(phi) - (ly_ - cy) * math.sin(phi)
        ry = cy + (lx_ - cx) * math.sin(phi) + (ly_ - cy) * math.cos(phi)
        x, y = _copies(rx, ry, view)
        traces[base + 1] = dict(x=x, y=y, fillcolor=_rgba(MOON_TINT[name], 1.0), opacity=op)
        traces[base + 2] = dict(x=[cx], y=[cy + r * 1.5 + 1.2], text=[f"{name} {mo['k'][i]:.0%}"], opacity=op)
        base += 3
    px_, py_, tx, ops = [], [], [], []
    for name, p in d.planets.items():
        x, y = proj.xy(p["alt"][i], p["az"][i])
        px_.append(float(x)); py_.append(float(y)); tx.append(name); ops.append(1.0 if p["alt"][i] > 0 else 0.35)
    traces[base] = dict(x=px_, y=py_, text=tx, marker=dict(size=9, color=[COLORS.get(n, "#ccc") for n in tx], opacity=ops))
    for tr in traces.values():
        for key in ("x", "y"):
            if key in tr:
                tr[key] = [None if v is None else round(float(v), 2) for v in tr[key]]
    return traces


def sky_figure(d: SkyData, view: str = "panorama", center_az: float = 180.0, fov: float = 360.0, scale: float = 10.0,
               show_stars: bool = True, start_index: int = 0, ref: SkyData | None = None, frame_ms: int = 140) -> go.Figure:
    proj = Projection(view, center_az)
    ref = ref or d
    fig = go.Figure()
    first = _frame_traces(d, start_index, proj, scale, show_stars)
    n_moons = len(d.moons)
    fig.add_trace(go.Scatter(x=first[0]["x"], y=first[0]["y"], fill="toself", mode="lines", line=dict(width=0), fillcolor=first[0]["fillcolor"], hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=first[1]["x"], y=first[1]["y"], mode="markers", marker=first[1]["marker"], hoverinfo="skip", showlegend=False))
    if view == "dome":
        a = np.linspace(0, 2 * np.pi, 180)
        ring_out = np.concatenate([120 * np.cos(a), 90 * np.cos(a[::-1])]); ring_oy = np.concatenate([120 * np.sin(a), 90 * np.sin(a[::-1])])
        fig.add_trace(go.Scatter(x=list(ring_out), y=list(ring_oy), fill="toself", mode="lines", fillcolor="rgb(30,26,24)",
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        gx, gy, gt = [], [], []
        for alt in (30, 60):
            r = 90 - alt
            gx += list(r * np.cos(a)) + [None]; gy += list(r * np.sin(a)) + [None]
        for i, name in enumerate(COMPASS):
            az = i * 45
            x_, y_ = proj.xy(0.0, az)
            xl, yl = proj.xy(-8.0, az)
            gt.append((float(xl), float(yl), name))
        fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines", line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[g[0] for g in gt], y=[g[1] for g in gt], mode="text", text=[g[2] for g in gt],
                                 textfont=dict(color="#ced4da", size=13), hoverinfo="skip", showlegend=False))
    else:
        fig.add_trace(go.Scatter(x=[-540, 540, 540, -540, -540], y=[-60, -60, 0, 0, -60], fill="toself", mode="lines", fillcolor="rgb(30,26,24)",
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        gx, gy = [-540, 540, None], [0, 0, None]
        for alt in (30, 60):
            gx += [-540, 540, None]; gy += [alt, alt, None]
        fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines", line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dot"), hoverinfo="skip", showlegend=False))
        lab = []
        for i, name in enumerate(COMPASS):
            rel = (i * 45 - center_az + 180) % 360 - 180
            for off in (-360, 0, 360):
                lab.append((rel + off, name))
        fig.add_trace(go.Scatter(x=[l[0] for l in lab], y=[-4] * len(lab), mode="text", text=[l[1] for l in lab],
                                 textfont=dict(color="#ced4da", size=14), hoverinfo="skip", showlegend=False))
    # each body's path across the day, with an arrow showing which way it goes
    tracks = [("Aureth", ref.sun["alt"], ref.sun["az"], "255,209,102")] + [
        (n, m["alt"], m["az"], ",".join(str(v) for v in MOON_TINT[n])) for n, m in ref.moons.items()]
    for name, alt, az, rgb in tracks:
        X, Y = proj.xy(alt, az)
        px, py, run = [], [], []
        for q in range(len(alt)):
            if alt[q] > 0 and not (run and view != "dome" and abs(X[q] - X[run[-1]]) > 90):
                run.append(q)
            else:
                if len(run) > 1:
                    px += list(X[run]) + [None]; py += list(Y[run]) + [None]
                run = [q] if alt[q] > 0 else []
        if len(run) > 1:
            px += list(X[run]) + [None]; py += list(Y[run]) + [None]
        fig.add_trace(go.Scatter(x=px, y=py, mode="lines", line=dict(color=f"rgba({rgb},0.5)", width=2, dash="dot"),
                                 hovertemplate=f"{name}'s path today<extra></extra>", showlegend=False))
        # arrow three quarters of the way along the longest stretch
        best, cur = [], []
        for q in range(len(alt)):
            if alt[q] > 0:
                cur.append(q)
            else:
                best = cur if len(cur) > len(best) else best
                cur = []
        best = cur if len(cur) > len(best) else best
        if len(best) >= 4:
            m = best[int(0.7 * (len(best) - 1))]
            idx3 = [m - 1, m, m + 1]
            fig.add_trace(go.Scatter(x=[float(X[q]) for q in idx3], y=[float(Y[q]) for q in idx3], mode="markers",
                                     marker=dict(symbol="arrow", angleref="previous", size=[0, 13, 0], color=f"rgba({rgb},0.9)"),
                                     hoverinfo="skip", showlegend=False))
        else:
            fig.add_trace(go.Scatter(x=[], y=[], mode="markers", hoverinfo="skip", showlegend=False))
    # dynamic objects
    fig.add_trace(go.Scatter(x=first[4]["x"], y=first[4]["y"], fill="toself", mode="lines", line=dict(width=0), fillcolor=first[4]["fillcolor"], hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=first[5]["x"], y=first[5]["y"], fill="toself", mode="lines", line=dict(width=0), fillcolor=first[5]["fillcolor"], hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=first[6]["x"], y=first[6]["y"], fill="toself", mode="lines", line=dict(width=0), fillcolor=first[6]["fillcolor"], hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=first[7]["x"], y=first[7]["y"], mode="text", text=first[7]["text"], textfont=dict(color="#ffd166", size=13), hoverinfo="skip", showlegend=False))
    base = 8
    for name in d.moons:
        fig.add_trace(go.Scatter(x=first[base]["x"], y=first[base]["y"], fill="toself", mode="lines", fillcolor=first[base]["fillcolor"], line=first[base]["line"], hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=first[base + 1]["x"], y=first[base + 1]["y"], fill="toself", mode="lines", fillcolor=first[base + 1]["fillcolor"], line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=first[base + 2]["x"], y=first[base + 2]["y"], mode="text", text=first[base + 2]["text"], textfont=dict(color="#e9ecef", size=12), hoverinfo="skip", showlegend=False))
        base += 3
    fig.add_trace(go.Scatter(x=first[base]["x"], y=first[base]["y"], mode="markers+text", text=first[base]["text"], textposition="top center",
                             textfont=dict(color="#e9ecef", size=11), marker=first[base]["marker"], hoverinfo="skip", showlegend=False))
    # figure traces: 0 sky, 1 stars, 2 ground, 3 grid, 4 compass, then the day's paths and arrows, then the Sun's two glows, disc and label,
    # the moons, the planets. A frame key k >= 4 therefore lives at figure trace k + 1 + the number of path traces.
    n_track_traces = 2 * (1 + n_moons)           # a path line and an arrow for the Sun and each moon
    fig_idx = {0: 0, 1: 1}
    for k in [4, 5, 6, 7] + list(range(8, 8 + 3 * n_moons + 1)):
        fig_idx[k] = k + 1 + n_track_traces

    frames = []
    for i in range(len(d.clock)):
        tr = _frame_traces(d, i, proj, scale, show_stars)
        data = []
        for k, idx in fig_idx.items():
            v = tr[k]
            kw = {kk: vv for kk, vv in v.items()}
            data.append(go.Scatter(**kw))
        frames.append(go.Frame(data=data, traces=list(fig_idx.values()), name=str(i)))
    fig.frames = frames

    if view == "dome":
        xr, yr, h = [-122, 122], [-122, 122], 700
    else:
        half = fov / 2
        xr = [-half, half]
        yr = [-22, min(95, 10 + 0.62 * fov)] if fov < 360 else [-22, 92]
        h = 470
    ticks = dict(tickvals=list(range(-180, 181, 30)) if fov >= 180 else list(range(-180, 181, 15)))
    fig.update_xaxes(range=xr, showgrid=False, zeroline=False, visible=(view != "dome"), scaleanchor="y", constrain="domain",
                     tickvals=ticks["tickvals"], ticktext=[f"{int((center_az + v) % 360)}°" for v in ticks["tickvals"]])
    fig.update_yaxes(range=yr, showgrid=False, zeroline=False, visible=(view != "dome"), constrain="domain", title_text="height above the horizon (°)")
    every = max(1, int(math.ceil(len(frames) / 10)))
    steps = [dict(label=(d.labels[i] if i % every == 0 else ""), method="animate",
                  args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for i, f in enumerate(frames)]
    fig.update_layout(
        height=h, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgb(10,12,20)", paper_bgcolor="rgba(0,0,0,0)",
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.12, buttons=[
            dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=int(frame_ms), redraw=True), fromcurrent=True)]),
            dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=start_index, currentvalue=dict(prefix="", font=dict(size=14)), pad=dict(t=30), steps=steps)])
    return fig


def day_crossings(d: SkyData) -> dict:
    """For each body, how it crosses the sky today: which side of the zenith, and which way it drifts on the panorama."""
    out = {}
    series = {"Aureth": (d.sun["alt"], d.sun["az"])}
    series.update({n: (m["alt"], m["az"]) for n, m in d.moons.items()})
    series.update({n: (p["alt"], p["az"]) for n, p in d.planets.items()})
    for name, (alt, az) in series.items():
        up = alt > 0
        runs, i = [], 0
        while i < len(up):
            if up[i]:
                j = i
                while j + 1 < len(up) and up[j + 1]:
                    j += 1
                runs.append((i, j))
                i = j + 1
            else:
                i += 1
        if not runs:
            out[name] = None
            continue
        i, j = max(runs, key=lambda r: alt[r[0]:r[1] + 1].max())
        z = np.degrees(np.unwrap(np.radians(az[i:j + 1])))
        top = az[i:j + 1][int(np.argmax(alt[i:j + 1]))]
        north = top < 90 or top > 270
        out[name] = {"side": "north" if north else "south", "rightward": bool(z[-1] - z[0] > 0),
                     "peak": float(alt[i:j + 1].max())}
    return out


def readout(d: SkyData, i: int, ref: SkyData | None = None) -> list[dict]:
    def compass(az):
        return COMPASS[int(((az + 22.5) % 360) // 45)]
    cross = day_crossings(ref or d)
    def how(name):
        c = cross.get(name)
        if c is None:
            return "stays below the horizon today"
        return f"passes {c['side']} of the zenith ({'left to right' if c['rightward'] else 'right to left'} in the panorama)"
    rows = []
    sa = float(d.sun["alt"][i])
    rows.append({"Object": "Aureth", "Height above the horizon": f"{sa:+.1f}°", "Direction": f"{float(d.sun['az'][i]):.0f}° {compass(float(d.sun['az'][i]))}",
                 "Looks": f"{2 * float(d.sun['rad'][i]):.3f}° wide", "Notes": twilight_name(sa), "Today it": how("Aureth")})
    for name, mo in d.moons.items():
        a = float(mo["alt"][i])
        rows.append({"Object": name, "Height above the horizon": f"{a:+.1f}°", "Direction": f"{float(mo['az'][i]):.0f}° {compass(float(mo['az'][i]))}",
                     "Looks": f"{2 * float(mo['rad'][i]):.3f}° wide", "Notes": f"{float(mo['k'][i]):.0%} lit" + ("" if a > 0 else ", below the horizon"), "Today it": how(name)})
    for name, p in d.planets.items():
        a = float(p["alt"][i])
        rows.append({"Object": name, "Height above the horizon": f"{a:+.1f}°", "Direction": f"{float(p['az'][i]):.0f}° {compass(float(p['az'][i]))}",
                     "Looks": "a point of light", "Notes": "" if a > 0 else "below the horizon", "Today it": how(name)})
    return rows


def find_sight(d: SkyData, name: str, i: int) -> float:
    """Direction (azimuth, degrees) of a named object at frame i, for pointing the view at it."""
    if name == "Aureth":
        return float(d.sun["az"][i])
    if name in d.moons:
        return float(d.moons[name]["az"][i])
    return float(d.planets[name]["az"][i])
