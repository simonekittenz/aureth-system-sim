from __future__ import annotations

import math

import numpy as np

from .aureth import AU_KM, K2, Body, Star
from .aureth import state as _kepler_state

G1 = 9.80665 * 86400.0**2 / (AU_KM * 1e3)       # one g in AU / day^2
AU_DAY_KMS = AU_KM / 86400.0


def straight_flight(star: Star, origin: Body, dest: Body, t0, a_g: float, iters: int = 8, eph=None) -> dict:
    """Constant-acceleration flip-and-burn from `origin` to `dest`, leaving at t0 (Earth days from year 1, day 1).

    The ship accelerates at a_g until halfway along the line, then brakes to arrive at rest relative to the destination.
    The planets follow their exact orbits; their motion during the flight is included. The Sun's pull is ignored (it is
    under 1% of a pilot's acceleration at planetary distances), and the line is flown straight.
    """
    state = eph.state if eph is not None else _kepler_state
    t0 = np.atleast_1d(np.asarray(t0, float))
    a = a_g * G1
    ro, vo = state(star, origin, t0)
    rd0, _ = state(star, dest, t0)
    T = 2 * np.sqrt(np.linalg.norm(rd0 - ro, axis=1) / a)
    for _ in range(iters):
        rd, vd = state(star, dest, t0 + T)
        d = rd - ro
        D = np.linalg.norm(d, axis=1)
        w = np.einsum("ij,ij->i", vo - vd, d / D[:, None])           # initial speed toward the target, in its final frame
        T = (2 * np.sqrt(w * w / 2 + a * D) - w) / a
    rd, vd = state(star, dest, t0 + T)
    p, q = ro[:, :2], rd[:, :2]
    s = q - p
    u = np.clip(-np.einsum("ij,ij->i", p, s) / np.einsum("ij,ij->i", s, s), 0, 1)
    closest = np.linalg.norm(p + u[:, None] * s, axis=1)
    D = np.linalg.norm(rd - ro, axis=1)
    sep = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", ro, state(star, dest, t0)[0])
                                       / (np.linalg.norm(ro, axis=1) * np.linalg.norm(state(star, dest, t0)[0], axis=1)), -1, 1)))
    return {"t0": t0, "T": T, "dist_au": D, "closest_au": closest, "sep_deg": sep,
            "v_peak_kms": np.sqrt(a * D) * AU_DAY_KMS}


def detour_flight(star: Star, origin: Body, dest: Body, t0: float, a_g: float, keep_au: float,
                  N: int = 16, sub: int = 3, eph=None) -> float | None:
    """Fastest flight at constant thrust magnitude (free direction, Sun's gravity included) that never comes
    within `keep_au` of the star. Returns days, or None if the optimiser did not converge."""
    from scipy.optimize import minimize

    state = eph.state if eph is not None else _kepler_state
    a = a_g * G1
    mu = K2 * star.mass_msun
    sf = straight_flight(star, origin, dest, t0, a_g, eph=eph)
    T0 = float(sf["T"][0])
    ro, vo = state(star, origin, t0)
    ro, vo = ro[0][:2], vo[0][:2]

    def propagate(z):
        Tt, th = z[0], z[1:]
        dt = Tt / N / sub
        r, v = ro.copy(), vo.copy()
        rs = []
        for i in range(N):
            thrust = np.array([a * math.cos(th[i]), a * math.sin(th[i])])
            for _ in range(sub):
                acc = -mu * r / np.linalg.norm(r) ** 3 + thrust
                vh = v + 0.5 * dt * acc
                r = r + dt * vh
                v = vh + 0.5 * dt * (-mu * r / np.linalg.norm(r) ** 3 + thrust)
                rs.append(r.copy())
        return r, v, np.array(rs)

    def eq(z):
        r, v, _ = propagate(z)
        rd, vd = state(star, dest, t0 + z[0])
        return np.concatenate([r - rd[0][:2], (v - vd[0][:2]) * 10])

    def ineq(z):
        return np.linalg.norm(propagate(z)[2], axis=1) - keep_au

    rd, _ = state(star, dest, t0 + T0)
    base = math.atan2(rd[0][1] - ro[1], rd[0][0] - ro[0])
    ramp = np.sin(np.linspace(0, math.pi, N // 2))
    best = None
    for sign in (1.0, -1.0):
        th0 = np.concatenate([base + sign * 0.9 * ramp, base + math.pi - sign * 0.9 * ramp[: N - N // 2]])
        res = minimize(lambda z: z[0], np.concatenate([[T0 * 1.3], th0]), method="SLSQP",
                       constraints=[{"type": "eq", "fun": eq}, {"type": "ineq", "fun": ineq}],
                       options={"maxiter": 250, "ftol": 1e-9})
        ok = res.success and np.max(np.abs(eq(res.x))) < 1e-4 and ineq(res.x).min() > -1e-4
        if ok and (best is None or res.x[0] < best):
            best = float(res.x[0])
        if best is not None:
            break
    return best


def scan_route(star: Star, origin: Body, dest: Body, a_g: float, span_days: float, keep_au: float, step: float = 1.0, eph=None) -> dict:
    t0 = np.arange(0, span_days, step)
    f = straight_flight(star, origin, dest, t0, a_g, eph=eph)
    f["blocked"] = f["closest_au"] < keep_au
    return f


def route_extremes(star: Star, origin: Body, dest: Body, a_g: float, span_days: float, keep_au: float, eph=None) -> dict:
    """Best and worst departure over the span. The worst case is re-solved with a curved route when the Sun is in the way."""
    f = scan_route(star, origin, dest, a_g, span_days, keep_au, eph=eph)
    ib, iw = int(np.argmin(f["T"])), int(np.argmax(f["T"]))
    worst = float(f["T"][iw])
    detour = False
    if f["blocked"][iw]:
        d = detour_flight(star, origin, dest, float(f["t0"][iw]), a_g, keep_au, eph=eph)
        if d is not None:
            worst, detour = d, True
        else:
            worst *= 1.03
    return {"best_T": float(f["T"][ib]), "best_t0": float(f["t0"][ib]), "best_dist": float(f["dist_au"][ib]),
            "best_sep": float(f["sep_deg"][ib]), "worst_T": worst, "worst_t0": float(f["t0"][iw]),
            "worst_dist": float(f["dist_au"][iw]), "worst_sep": float(f["sep_deg"][iw]),
            "worst_detour": detour, "worst_blocked": bool(f["blocked"][iw]),
            "v_peak_best": float(f["v_peak_kms"][ib]), "v_peak_worst": float(f["v_peak_kms"][iw])}
