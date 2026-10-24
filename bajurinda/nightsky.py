from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from .aureth import COLORS, load_system, state
from .dynamics import build_simulation
from .model import System, angular_diameter_deg
from .sky import illuminated_fraction, phase_disc, spin_axis, surface_normal
from .starfield import band_edges, circle_on_sky, ecl_vec, faint_stars, load_faint_settings, load_sky

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
    spin: np.ndarray                # the planet's spin axis (inertial unit vector)
    U: np.ndarray                   # local frame at each frame: up, east, north (inertial unit vectors)
    E: np.ndarray
    N: np.ndarray
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

    return SkyData(t, clock, labels, sun, moons, planets, s, U, E, N, lat, lon)


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


def _vis(sun_alt: float, mag: float) -> float:
    """How visible a star of magnitude `mag` is when the Sun is at `sun_alt`: bright stars show in twilight, faint ones need real dark."""
    return float(max(0.0, min(1.0, (-sun_alt - (4.0 + 2.5 * mag)) / 6.0)))


def _poly(x, y, view):
    """Cut a closed outline at the panorama's wrap-around seam so no streaks cross the picture."""
    if view == "dome":
        return list(x), list(y)
    ox, oy, run = [], [], []
    for q in range(len(x)):
        if run and abs(x[q] - x[run[-1]]) > 120:
            if len(run) > 1:
                ox += [x[r] for r in run] + [None]; oy += [y[r] for r in run] + [None]
            run = []
        run.append(q)
    if len(run) > 1:
        ox += [x[r] for r in run] + [None]; oy += [y[r] for r in run] + [None]
    return ox, oy


def _band_centre(mw, d, i, proj, n=96):
    """The galaxy's centre line projected on the picture, and its half-width in degrees at each point."""
    from .starfield import mw_basis, half_width
    C, E2, P = mw_basis(mw)
    l = np.linspace(0, 360, n + 1)
    u = np.cos(np.radians(l))[:, None] * C + np.sin(np.radians(l))[:, None] * E2
    U, E, N = d.U[i], d.E[i], d.N[i]
    alt, az = _altaz(u, np.broadcast_to(U, u.shape), np.broadcast_to(E, u.shape), np.broadcast_to(N, u.shape))
    x, y = proj.xy(alt, az)
    return np.asarray(x, float), np.asarray(y, float), half_width(l, mw), l


def _ribbon_screen(x, y, w, scale, view, l0=None, l1=None, l=None):
    """A ribbon of half-width scale*w drawn around the line (x, y), in picture degrees. Returns closed pieces."""
    ox, oy = [], []
    n = len(x)
    start = 0
    for q in range(1, n + 1):
        cut = q == n or (view != "dome" and abs(x[q] - x[q - 1]) > 60)
        if not cut:
            continue
        seg = np.arange(start, q)
        start = q
        if len(seg) < 3:
            continue
        if l0 is not None:
            ll = l[seg]
            inside = ((ll - l0) % 360) <= ((l1 - l0) % 360)
            if inside.sum() < 3:
                continue
            seg = seg[inside]
        sx, sy = x[seg], y[seg]
        tx, ty = np.gradient(sx), np.gradient(sy)
        norm = np.hypot(tx, ty)
        norm[norm == 0] = 1.0
        nx, ny = -ty / norm, tx / norm
        hw = w[seg] * scale
        px = list(sx + nx * hw) + list((sx - nx * hw)[::-1]) + [float(sx[0] + nx[0] * hw[0])]
        py = list(sy + ny * hw) + list((sy - ny * hw)[::-1]) + [float(sy[0] + ny[0] * hw[0])]
        ox += px + [None]
        oy += py + [None]
    return ox, oy


MW_LAYERS = tuple((f"mw_{k}", sc, col) for k, (sc, col) in enumerate((
    (2.3, "rgba(140,165,250,0.030)"), (1.85, "rgba(160,182,252,0.034)"), (1.5, "rgba(185,200,255,0.038)"),
    (1.2, "rgba(210,220,255,0.044)"), (0.9, "rgba(235,236,255,0.052)"), (0.6, "rgba(255,244,222,0.065)"))))


def _con_traces(cset: dict, stars, U, E, N, proj: Projection, sun_alt: float, labels: bool):
    """The lines and figure names of one constellation set, as seen at this moment. Lines follow great circles and stop short of each star."""
    view = proj.view
    vec = {st.name: ecl_vec(np.array(st.lon), np.array(st.lat)) for st in stars}
    fade = round(0.85 * _vis(sun_alt, 1.0), 2)
    xs, ys, lx, ly, lt = [], [], [], [], []
    t = np.linspace(0, 1, 16)[:, None]
    for fg in cset["figures"]:
        members = []
        for a, b in fg["lines"]:
            if a not in vec or b not in vec:
                continue
            va, vb = vec[a], vec[b]
            om = math.acos(float(np.clip(va @ vb, -1, 1)))
            if om < 1e-4 or om > 3.1:
                continue
            members += [a, b]
            trim = min(0.4, math.radians(1.3) / om)
            tt = trim + t * (1 - 2 * trim)
            pts = (np.sin((1 - tt) * om) * va + np.sin(tt * om) * vb) / math.sin(om)
            alt_, az_ = _altaz(pts, np.broadcast_to(U, pts.shape), np.broadcast_to(E, pts.shape), np.broadcast_to(N, pts.shape))
            up = alt_ > 0
            if up.sum() < 2:
                continue
            px, py = _poly(*proj.xy(alt_[up], az_[up]), view)
            xs += px + [None]; ys += py + [None]
        if labels and members:
            c = sum(vec[m] for m in set(members))
            nc = np.linalg.norm(c)
            if nc > 1e-6:
                c = (c / nc)[None, :]
                alt_, az_ = _altaz(c, U[None, :], E[None, :], N[None, :])
                if alt_[0] > 0:
                    x, y = proj.xy(alt_, az_)
                    lx.append(float(np.ravel(x)[0])); ly.append(float(np.ravel(y)[0])); lt.append(fg["name"])
    return dict(x=xs, y=ys, opacity=fade), dict(x=lx, y=ly, text=lt, opacity=fade)


def _frame_traces(d: SkyData, i: int, proj: Projection, scale: float, opts: dict):
    view = proj.view
    sun_alt = float(d.sun["alt"][i])
    sky = _rgba(sky_rgb(sun_alt), 1.0)
    traces = {}
    if view == "dome":
        a = np.linspace(0, 2 * np.pi, 120)
        traces["sky"] = dict(x=list(90 * np.cos(a)), y=list(90 * np.sin(a)), fillcolor=sky)
    else:
        traces["sky"] = dict(x=[-540, 540, 540, -540, -540], y=[0, 0, 120, 120, 0], fillcolor=sky)
    U, E, N = d.U[i], d.E[i], d.N[i]
    stars, mw = load_sky()

    # the galaxy
    mw_vis = max(0.0, min(1.0, (-sun_alt - 14) / 6)) * mw.brightness if opts["milky_way"] else 0.0
    if mw_vis > 0:
        cx_, cy_, wv, lv = _band_centre(mw, d, i, proj)
        for key, sc, _ in MW_LAYERS:
            x, y = _ribbon_screen(cx_, cy_, wv, sc, view)
            traces[key] = dict(x=x, y=y, opacity=mw_vis)
        x, y = _ribbon_screen(cx_, cy_, wv, 0.1, view, mw.rift_from, mw.rift_to, lv)
        traces["mw_rift"] = dict(x=x, y=y, opacity=mw_vis * mw.rift_strength)
        bx, by, dx, dy = [], [], [], []
        for lon_, lat_, size_, strength in mw.patches:
            for frac in (1.0, 0.55):
                circ = circle_on_sky(lon_, lat_, size_ * frac)
                alt_, az_ = _altaz(circ, np.broadcast_to(U, circ.shape), np.broadcast_to(E, circ.shape), np.broadcast_to(N, circ.shape))
                px, py = _poly(*proj.xy(alt_, az_), view)
                if strength > 0:
                    bx += px + [None]; by += py + [None]
                else:
                    dx += px + [None]; dy += py + [None]
        traces["mw_pb"] = dict(x=bx, y=by, opacity=mw_vis)
        traces["mw_pd"] = dict(x=dx, y=dy, opacity=mw_vis * 0.7)
    else:
        for key, _, _ in MW_LAYERS:
            traces[key] = dict(x=[], y=[], opacity=0)
        for key in ("mw_rift", "mw_pb", "mw_pd"):
            traces[key] = dict(x=[], y=[], opacity=0)

    # constellation lines (several sets can show at once, each in its own colour)
    for k, cset in enumerate(opts.get("cons") or []):
        traces[f"con{k}"], traces[f"conlab{k}"] = _con_traces(cset, stars, U, E, N, proj, sun_alt, opts.get("con_labels", True))

    # faint stars, crowded along the galaxy
    fv, fm = _faint(opts.get("seed"))
    if opts["faint"]:
        alt_, az_ = _altaz(fv, np.broadcast_to(U, fv.shape), np.broadcast_to(E, fv.shape), np.broadcast_to(N, fv.shape))
        vis = np.array([_vis(sun_alt, m) for m in fm])
        keep = (alt_ > 0) & (vis > 0)
        x, y = proj.xy(alt_[keep], az_[keep])
        traces["faint"] = dict(x=list(x), y=list(y), marker=dict(size=list(np.round(1.2 + (5.6 - fm[keep]) * 0.4, 2)), color="white",
                                                                  opacity=list(np.round(vis[keep] * 0.6, 2))))
    else:
        traces["faint"] = dict(x=[], y=[], marker=dict(size=[], color="white", opacity=[]))

    # the named stars
    if opts["named"] and stars:
        sv = ecl_vec(np.array([st.lon for st in stars]), np.array([st.lat for st in stars]))
        alt_, az_ = _altaz(sv, np.broadcast_to(U, sv.shape), np.broadcast_to(E, sv.shape), np.broadcast_to(N, sv.shape))
        vis = np.array([_vis(sun_alt, st.mag) for st in stars])
        keep = (alt_ > 0) & (vis > 0)
        x, y = proj.xy(alt_[keep], az_[keep])
        kept = [st for st, k in zip(stars, keep) if k]
        traces["named"] = dict(x=list(x), y=list(y),
                               text=[st.name if (st.label and vis[j] > 0.25) else "" for st, j in zip(kept, np.where(keep)[0])],
                               marker=dict(size=[round(float(np.clip(7 - 1.7 * st.mag, 3.5, 12)), 1) for st in kept],
                                           color=[st.color for st in kept], opacity=[round(float(vis[j]), 2) for j in np.where(keep)[0]]))
    else:
        traces["named"] = dict(x=[], y=[], text=[], marker=dict(size=[], color=[], opacity=[]))

    def place(alt, az, rad_deg):
        x, y = proj.xy(alt, az)
        return float(x), float(y), rad_deg * scale

    cx, cy, r = place(d.sun["alt"][i], d.sun["az"][i], d.sun["rad"][i])
    op = 1.0 if sun_alt > 0 else 0.35
    for key, mult, alpha in (("sun_g1", 6.0, 0.10), ("sun_g2", 2.5, 0.22)):
        gx, gy = _circle(cx, cy, r * mult)
        x, y = _copies(gx, gy, view)
        traces[key] = dict(x=x, y=y, fillcolor=f"rgba(255,209,102,{alpha})", opacity=op)
    gx, gy = _circle(cx, cy, r)
    x, y = _copies(gx, gy, view)
    traces["sun"] = dict(x=x, y=y, fillcolor="rgb(255,222,130)", opacity=op)
    lx, ly = proj.xy(d.sun["alt"][i], d.sun["az"][i])
    traces["sun_label"] = dict(x=[float(lx)], y=[float(ly) + r * 3.2 + 1.5], text=["Aureth" + ("" if sun_alt > 0 else " (below the horizon)")], opacity=op)

    for k, (name, mo) in enumerate(d.moons.items()):
        alt, az = float(mo["alt"][i]), float(mo["az"][i])
        cx, cy, r = place(alt, az, mo["rad"][i])
        op = 1.0 if alt > 0 else 0.35
        gx, gy = _circle(cx, cy, r)
        x, y = _copies(gx, gy, view)
        traces[f"m{k}_dark"] = dict(x=x, y=y, opacity=op)
        px, py = proj.xy(mo["limb_alt"][i], mo["limb_az"][i])
        dxv = ((float(px) - cx + 180) % 360) - 180 if view != "dome" else float(px) - cx
        phi = math.atan2(float(py) - cy, dxv)
        lx_, ly_ = phase_disc(float(mo["k"][i]), True, r, cx, cy)
        rx = cx + (lx_ - cx) * math.cos(phi) - (ly_ - cy) * math.sin(phi)
        ry = cy + (lx_ - cx) * math.sin(phi) + (ly_ - cy) * math.cos(phi)
        x, y = _copies(rx, ry, view)
        traces[f"m{k}_lit"] = dict(x=x, y=y, opacity=op)
        traces[f"m{k}_label"] = dict(x=[cx], y=[cy + r * 1.5 + 1.2], text=[f"{name} {mo['k'][i]:.0%}"], opacity=op)
    px_, py_, tx, ops = [], [], [], []
    for name, p in d.planets.items():
        x, y = proj.xy(p["alt"][i], p["az"][i])
        px_.append(float(x)); py_.append(float(y)); tx.append(name); ops.append(1.0 if p["alt"][i] > 0 else 0.35)
    traces["planets"] = dict(x=px_, y=py_, text=tx, marker=dict(size=9, color=[COLORS.get(n, "#ccc") for n in tx], opacity=ops))
    for tr in traces.values():
        for key in ("x", "y"):
            if key in tr:
                tr[key] = [None if v is None else round(float(v), 2) for v in tr[key]]
    return traces


_FAINT_CACHE: dict = {}


def _faint(seed: int | None):
    st = load_faint_settings()
    seed = st["seed"] if seed is None else int(seed)
    key = (seed, st["n_band"], st["n_field"])
    if key not in _FAINT_CACHE:
        _FAINT_CACHE[key] = faint_stars(load_sky()[1], st["n_band"], st["n_field"], seed)
    return _FAINT_CACHE[key]


def _panorama_yrange(fov: float) -> list[float]:
    return [-12, min(92, 10 + 0.62 * fov)] if fov < 360 else [-12, 90]


def _pick_grid(d: SkyData, i: int, proj: Projection, fov: float):
    """A fine grid over the picture whose hover text gives the sky coordinates under the pointer at frame i."""
    stars, _mw = load_sky()
    if proj.view == "dome":
        step = 1.5
        xs = np.arange(-108, 108.001, step)
        ys = np.arange(-108, 108.001, step)
        X, Y = np.meshgrid(xs, ys)
        r = np.hypot(X, Y)
        alt = 90.0 - r
        az = np.degrees(np.arctan2(-X, Y)) % 360
        outside = r > 108
    else:
        half = fov / 2
        step = max(0.5, fov / 170)
        y0, y1 = _panorama_yrange(fov)
        xs = np.arange(-half, half + 1e-9, step)
        ys = np.arange(y0, y1 + 1e-9, step)
        X, Y = np.meshgrid(xs, ys)
        alt = Y
        az = (proj.c + X) % 360
        outside = np.zeros_like(alt, dtype=bool)
    U, E, N = d.U[i], d.E[i], d.N[i]
    sa, ca = np.sin(np.radians(alt))[..., None], np.cos(np.radians(alt))[..., None]
    v = sa * U + ca * (np.sin(np.radians(az))[..., None] * E + np.cos(np.radians(az))[..., None] * N)
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360
    lat = np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))
    s_ax = d.spin
    dec = np.degrees(np.arcsin(np.clip(v @ s_ax, -1, 1)))
    near_txt = np.full(alt.shape, "", dtype=object)
    if stars:
        sv = ecl_vec(np.array([st.lon for st in stars]), np.array([st.lat for st in stars]))
        sep = np.degrees(np.arccos(np.clip(v @ sv.T, -1, 1)))
        j = sep.argmin(axis=-1)
        sm = np.take_along_axis(sep, j[..., None], axis=-1)[..., 0]
        for q in np.flatnonzero((sm < 2.5).ravel()):
            near_txt.ravel()[q] = f"<br>near {stars[int(j.ravel()[q])].name} ({sm.ravel()[q]:.1f}° away)"
    comp = np.array(COMPASS, dtype=object)[((az + 22.5) % 360 // 45).astype(int)]
    text = np.empty(alt.shape, dtype=object)
    flat = text.ravel()
    for q, (lo, la, al, az_, de, c, nt, out) in enumerate(zip(lon.ravel(), lat.ravel(), alt.ravel(), az.ravel(), dec.ravel(),
                                                               comp.ravel(), np.asarray(near_txt, dtype=object).ravel(), outside.ravel())):
        flat[q] = "" if out else (f"<b>longitude {lo:.1f}°  latitude {la:+.1f}°</b><br>(ecliptic)"
                                  f"<br>height {al:+.0f}°  direction {az_:.0f}° {c}<br>declination {de:+.1f}°{nt}")
    return xs, ys, text


def sky_figure(d: SkyData, view: str = "panorama", center_az: float = 180.0, fov: float = 360.0, scale: float = 10.0,
               named: bool = True, start_index: int = 0, ref: SkyData | None = None, frame_ms: int = 140,
               milky_way: bool = True, faint: bool = True, faint_seed: int | None = None, height: int = 0,
               pick: bool = False, preview: dict | None = None,
               cons: list[dict] | None = None, con_labels: bool = True) -> go.Figure:
    proj = Projection(view, center_az)
    ref = ref or d
    opts = {"named": named, "milky_way": milky_way, "faint": faint, "seed": faint_seed, "cons": cons or [], "con_labels": con_labels}
    first = _frame_traces(d, start_index, proj, scale, opts)
    n_moons = len(d.moons)
    fig = go.Figure()
    idx: dict[str, int] = {}

    def dyn(key, **kw):
        props = {k: v for k, v in first[key].items()}
        kw.update({k: props[k] for k in props if k in ("x", "y", "text", "fillcolor", "opacity", "marker")})
        if "fillcolor" in props and "fillcolor" in kw:
            kw["fillcolor"] = props["fillcolor"]
        fig.add_trace(go.Scatter(hoverinfo="skip", showlegend=False, **kw))
        idx[key] = len(fig.data) - 1

    filled = dict(fill="toself", mode="lines", line=dict(width=0))
    dyn("sky", **filled)
    for key, _, colour in MW_LAYERS:
        dyn(key, **{**filled, "fillcolor": colour})
    dyn("mw_rift", **{**filled, "fillcolor": "rgba(5,8,18,0.8)"})
    dyn("mw_pb", **{**filled, "fillcolor": "rgba(255,238,215,0.035)"})
    dyn("mw_pd", **{**filled, "fillcolor": "rgba(4,6,16,0.45)"})
    for k, cset in enumerate(cons or []):
        dyn(f"con{k}", mode="lines", line=dict(color=cset["color"], width=1.8))
        dyn(f"conlab{k}", mode="text", textposition="bottom center" if k % 2 == 0 else "top center", textfont=dict(color=cset["color"], size=13))
    dyn("faint", mode="markers")
    dyn("named", mode="markers+text", textposition="top right", textfont=dict(color="#ced4da", size=11))

    # ground, grid, compass
    if view == "dome":
        a = np.linspace(0, 2 * np.pi, 180)
        ring_x = np.concatenate([110 * np.cos(a), 90 * np.cos(a[::-1])]); ring_y = np.concatenate([110 * np.sin(a), 90 * np.sin(a[::-1])])
        fig.add_trace(go.Scatter(x=list(ring_x), y=list(ring_y), fill="toself", mode="lines", fillcolor="rgb(30,26,24)", line=dict(width=0), hoverinfo="skip", showlegend=False))
        mask_x = np.concatenate([320 * np.cos(a), 110 * np.cos(a[::-1])]); mask_y = np.concatenate([320 * np.sin(a), 110 * np.sin(a[::-1])])
        fig.add_trace(go.Scatter(x=list(mask_x), y=list(mask_y), fill="toself", mode="lines", fillcolor="rgb(10,12,20)", line=dict(width=0), hoverinfo="skip", showlegend=False))
        gx, gy = [], []
        for alt in (30, 60):
            r = 90 - alt
            gx += list(r * np.cos(a)) + [None]; gy += list(r * np.sin(a)) + [None]
        fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines", line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"), hoverinfo="skip", showlegend=False))
        lab = [(float(proj.xy(-8.0, i * 45)[0]), float(proj.xy(-8.0, i * 45)[1]), name) for i, name in enumerate(COMPASS)]
    else:
        fig.add_trace(go.Scatter(x=[-540, 540, 540, -540, -540], y=[-60, -60, 0, 0, -60], fill="toself", mode="lines", fillcolor="rgb(30,26,24)", line=dict(width=0), hoverinfo="skip", showlegend=False))
        gx, gy = [-540, 540, None], [0, 0, None]
        for alt in (30, 60):
            gx += [-540, 540, None]; gy += [alt, alt, None]
        fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines", line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dot"), hoverinfo="skip", showlegend=False))
        lab = []
        for i, name in enumerate(COMPASS):
            rel = (i * 45 - center_az + 180) % 360 - 180
            for off in (-360, 0, 360):
                lab.append((rel + off, -4, name))
    fig.add_trace(go.Scatter(x=[l[0] for l in lab], y=[l[1] for l in lab], mode="text", text=[l[2] for l in lab],
                             textfont=dict(color="#ced4da", size=14), hoverinfo="skip", showlegend=False))

    # each body's path across the reference day, with an arrow
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
                                 hoverinfo="skip" if pick else None, hovertemplate=None if pick else f"{name}'s path<extra></extra>", showlegend=False))
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
            i3 = [m - 1, m, m + 1]
            fig.add_trace(go.Scatter(x=[float(X[q]) for q in i3], y=[float(Y[q]) for q in i3], mode="markers",
                                     marker=dict(symbol="arrow", angleref="previous", size=[0, 13, 0], color=f"rgba({rgb},0.9)"),
                                     hoverinfo="skip", showlegend=False))
        else:
            fig.add_trace(go.Scatter(x=[], y=[], mode="markers", hoverinfo="skip", showlegend=False))

    dyn("sun_g1", **filled)
    dyn("sun_g2", **filled)
    dyn("sun", **filled)
    dyn("sun_label", mode="text", textfont=dict(color="#ffd166", size=13))
    for k, name in enumerate(d.moons):
        dyn(f"m{k}_dark", fill="toself", mode="lines", fillcolor="rgb(38,42,58)", line=dict(color=_rgba(MOON_TINT[name], 0.6), width=1))
        dyn(f"m{k}_lit", fill="toself", mode="lines", fillcolor=_rgba(MOON_TINT[name], 1.0), line=dict(width=0))
        dyn(f"m{k}_label", mode="text", textfont=dict(color="#e9ecef", size=12))
    dyn("planets", mode="markers+text", textposition="top center", textfont=dict(color="#e9ecef", size=11))

    frames = []
    for i in (range(len(d.clock)) if not pick else []):
        tr = _frame_traces(d, i, proj, scale, opts)
        data = []
        for key, fi in idx.items():
            v = tr[key]
            data.append(go.Scatter(**{k: v[k] for k in v if k in ("x", "y", "text", "fillcolor", "opacity", "marker")}))
        frames.append(go.Frame(data=data, traces=list(idx.values()), name=str(i)))
    fig.frames = frames

    if pick:
        gx, gy, gt = _pick_grid(d, start_index, proj, fov)
        if preview:
            pv = ecl_vec(np.array(preview["lon"]), np.array(preview["lat"]))[None, :]
            pa, pz = _altaz(pv, d.U[start_index][None, :], d.E[start_index][None, :], d.N[start_index][None, :])
            px, py = proj.xy(pa, pz)
            fig.add_trace(go.Scatter(x=[float(px[0])], y=[float(py[0])], mode="markers+text", text=[preview.get("name", "preview")],
                                     textposition="bottom right", textfont=dict(color="#ff6fa5", size=13),
                                     marker=dict(symbol="circle-open", size=max(18, 22 - 3 * preview.get("mag", 1.0)), color="#ff006e", line=dict(width=2)),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=[float(px[0])], y=[float(py[0])], mode="markers", marker=dict(size=5, color="#ff006e"),
                                     hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Heatmap(x=gx, y=gy, z=np.zeros((len(gy), len(gx))), text=gt, hovertemplate="%{text}<extra></extra>",
                                 colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False, hoverongaps=False, zsmooth=False))
    if view == "dome":
        xr, yr, h = [-108, 108], [-108, 108], height or 700
    else:
        half = fov / 2
        xr = [-half, half]
        yr = _panorama_yrange(fov)
        h = height or 470
    tv = list(range(-180, 181, 30)) if fov >= 180 else list(range(-180, 181, 15))
    fig.update_xaxes(range=xr, showgrid=False, zeroline=False, visible=(view != "dome"), scaleanchor="y", constrain="domain",
                     tickvals=tv, ticktext=[f"{int((center_az + v) % 360)}°" for v in tv])
    fig.update_yaxes(range=yr, showgrid=False, zeroline=False, visible=(view != "dome"), constrain="domain", title_text="height above the horizon (°)")
    every = max(1, int(math.ceil(len(frames) / 10)))
    steps = [dict(label=(d.labels[i] if i % every == 0 else ""), method="animate",
                  args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for i, f in enumerate(frames)]
    layout = dict(height=h, margin=dict(l=10, r=10, t=10 if pick else 44, b=10), plot_bgcolor="rgb(10,12,20)", paper_bgcolor="rgba(0,0,0,0)",
                  hoverlabel=dict(bgcolor="rgba(20,24,40,0.95)", font=dict(size=14, color="#f1f3f5"), bordercolor="#868e96"))
    if not pick:
        layout.update(updatemenus=[dict(type="buttons", direction="left", x=0.0, y=1.0, yanchor="bottom", buttons=[
            dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=int(frame_ms), redraw=True), fromcurrent=True)]),
            dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
            sliders=[dict(active=start_index, currentvalue=dict(prefix="", font=dict(size=14)), pad=dict(t=30), steps=steps)])
    fig.update_layout(**layout)
    if pick and view != "dome":
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="rgba(255,255,255,0.45)", spikethickness=1)
        fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="rgba(255,255,255,0.45)", spikethickness=1)
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
    if name in d.planets:
        return float(d.planets[name]["az"][i])
    for st in load_sky()[0]:
        if st.name == name:
            v = ecl_vec(np.array(st.lon), np.array(st.lat))[None, :]
            _, az = _altaz(v, d.U[i][None, :], d.E[i][None, :], d.N[i][None, :])
            return float(az[0])
    return 180.0
