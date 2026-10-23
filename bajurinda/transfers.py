from __future__ import annotations

import math

import numpy as np

from .aureth import AU_PER_DAY_KMS, K2, Body, Star, state


def stumpff(z: np.ndarray):
    sq = np.sqrt(np.abs(z))
    big = np.abs(z) > 1e-8
    safe = np.where(big, z, 1.0)
    sq_safe = np.where(big, sq, 1.0)
    C = np.where(z > 1e-8, (1 - np.cos(sq_safe)) / safe, np.where(z < -1e-8, (np.cosh(sq_safe) - 1) / (-safe), 0.5 - z / 24))
    S = np.where(z > 1e-8, (sq_safe - np.sin(sq_safe)) / sq_safe**3,
                 np.where(z < -1e-8, (np.sinh(sq_safe) - sq_safe) / sq_safe**3, 1 / 6 - z / 120))
    return C, S


def lambert(r1: np.ndarray, r2: np.ndarray, dt: np.ndarray, mu: float):
    """Prograde zero-revolution Lambert solution (universal variables, vectorised bisection).

    r1, r2: (N, 3) positions; dt: (N,) flight times; returns v1, v2 (N, 3) and a validity mask.
    """
    r1n, r2n = np.linalg.norm(r1, axis=1), np.linalg.norm(r2, axis=1)
    c = np.einsum("ij,ij->i", r1, r2) / (r1n * r2n)
    cross_z = r1[:, 0] * r2[:, 1] - r1[:, 1] * r2[:, 0]
    dth = np.where(cross_z >= 0, np.arccos(np.clip(c, -1, 1)), 2 * np.pi - np.arccos(np.clip(c, -1, 1)))
    A = np.sin(dth) * np.sqrt(r1n * r2n / (1 - np.cos(dth)))

    def y_of(z):
        C, S = stumpff(z)
        return r1n + r2n + A * (z * S - 1) / np.sqrt(C)

    lo = np.full(len(dt), -20.0)
    for _ in range(200):
        bad = y_of(lo) < 0
        if not bad.any():
            break
        lo = np.where(bad, lo + 0.5, lo)
    hi = np.full(len(dt), (2 * np.pi) ** 2 - 1e-6)
    sqmu = math.sqrt(mu)
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        C, S = stumpff(mid)
        yy = y_of(mid)
        with np.errstate(invalid="ignore", divide="ignore"):
            F = (np.maximum(yy, 0) / C) ** 1.5 * S + A * np.sqrt(np.maximum(yy, 0)) - sqmu * dt
        low_side = (F < 0) | (yy < 0)
        lo = np.where(low_side, mid, lo)
        hi = np.where(low_side, hi, mid)
    z = 0.5 * (lo + hi)
    yy = y_of(z)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = 1 - yy / r1n
        g = A * np.sqrt(yy / mu)
        gdot = 1 - yy / r2n
        v1 = (r2 - f[:, None] * r1) / g[:, None]
        v2 = (gdot[:, None] * r2 - r1) / g[:, None]
    ok = np.isfinite(v1).all(axis=1) & np.isfinite(v2).all(axis=1) & (A > 0) & (yy > 0)
    return v1, v2, ok


def hohmann(a1: float, a2: float, mu: float) -> dict:
    """Circular, coplanar Hohmann transfer between radii a1 and a2 (AU). Speeds in km/s, times in days."""
    at = 0.5 * (a1 + a2)
    tof = math.pi * math.sqrt(at**3 / mu)
    v1, v2 = math.sqrt(mu / a1), math.sqrt(mu / a2)
    vp, va = math.sqrt(mu * (2 / a1 - 1 / at)), math.sqrt(mu * (2 / a2 - 1 / at))
    n2 = math.sqrt(mu / a2**3)
    lead = math.degrees(math.pi - n2 * tof)            # angle the destination must be ahead of the origin at departure
    return {"tof_days": tof, "vinf_dep": abs(vp - v1) * AU_PER_DAY_KMS, "vinf_arr": abs(v2 - va) * AU_PER_DAY_KMS,
            "lead_deg": lead}


def porkchop(star: Star, origin: Body, dest: Body, dep_days: np.ndarray, tof_days: np.ndarray) -> dict:
    mu = K2 * star.mass_msun
    D, T = np.meshgrid(dep_days, tof_days, indexing="ij")
    dep, tof = D.ravel(), T.ravel()
    r1, v1p = state(star, origin, dep)
    r2, v2p = state(star, dest, dep + tof)
    v1, v2, ok = lambert(r1, r2, tof, mu)
    vd = np.linalg.norm(v1 - v1p, axis=1) * AU_PER_DAY_KMS
    va = np.linalg.norm(v2 - v2p, axis=1) * AU_PER_DAY_KMS
    vd = np.where(ok, vd, np.nan).reshape(D.shape)
    va = np.where(ok, va, np.nan).reshape(D.shape)
    return {"dep": dep_days, "tof": tof_days, "vinf_dep": vd, "vinf_arr": va, "total": vd + va}


def find_windows(pc: dict, synodic_days: float, from_day: float, to_day: float, max_windows: int = 12) -> list[dict]:
    """One best (departure, flight time) per launch window: the cheapest departure within half a synodic
    period on each side. The grid in `pc` should extend beyond [from_day, to_day] by about half a synodic period."""
    total = np.where(np.isnan(pc["total"]), np.inf, pc["total"])
    best, arg = total.min(axis=1), total.argmin(axis=1)
    step = float(pc["dep"][1] - pc["dep"][0])
    half = max(int(0.4 * synodic_days / step), 1)
    out = []
    for i in range(half, len(best) - half):
        if np.isfinite(best[i]) and best[i] <= best[i - half:i + half + 1].min() + 1e-12:
            d = float(pc["dep"][i])
            if from_day <= d <= to_day:
                out.append({"dep_day": d, "tof_days": float(pc["tof"][arg[i]]),
                            "vinf_dep": float(pc["vinf_dep"][i, arg[i]]), "vinf_arr": float(pc["vinf_arr"][i, arg[i]]),
                            "total": float(best[i])})
    dedup = []
    for w in out:
        if not dedup or w["dep_day"] - dedup[-1]["dep_day"] > 0.5 * synodic_days:
            dedup.append(w)
        elif w["total"] < dedup[-1]["total"]:
            dedup[-1] = w
    return dedup[:max_windows]


def launch_speed(body: Body, vinf_kms: float) -> float | None:
    """Speed needed at the surface to leave with excess speed vinf (ignores the atmosphere and the planet's spin)."""
    if body.escape_kms is None:
        return None
    return math.sqrt(body.escape_kms**2 + vinf_kms**2)
