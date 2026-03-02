"""Angles-only IOD using Izzo Lambert with local SEZ->ECEF->ECI conversions.

This module performs:
- angle conversion into observer site position and LOS vectors in inertial axes
- 3-observation angles-only IOD using Izzo Lambert
- optional state covariance propagation from angle covariance
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from astropy import units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, GCRS, ITRS
from astropy.time import Time
from astropy.utils import iers

iers.conf.auto_download = False

try:
    from hapsira.core.iod import izzo as _hapsira_izzo
except Exception:  # pragma: no cover - optional dependency
    _hapsira_izzo = None

MU_EARTH_KM3_S2 = 398600.4418
EARTH_ROT_RATE = (7.2921150e-5 * u.rad / u.s).to_value(
    1 / u.s, equivalencies=u.dimensionless_angles()
)
WGS84_A = 6378.137 * u.km
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


class IzzoConvergenceError(RuntimeError):
    """Raised when the Izzo Lambert iterator fails to converge."""


class AnglesOnlyConvergenceError(RuntimeError):
    """Raised when the angles-only range iteration fails to converge."""


def _lat_lon_trig(lat: u.Quantity, lon: u.Quantity) -> tuple[float, float, float, float]:
    lat_rad = lat.to_value(u.rad)
    lon_rad = lon.to_value(u.rad)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    return sin_lat, cos_lat, sin_lon, cos_lon


def _az_el_trig(az: u.Quantity, el: u.Quantity) -> tuple[float, float, float, float]:
    az_rad = az.to_value(u.rad)
    el_rad = el.to_value(u.rad)
    return np.sin(az_rad), np.cos(az_rad), np.sin(el_rad), np.cos(el_rad)


def _site_position(
    sin_lat: float,
    cos_lat: float,
    sin_lon: float,
    cos_lon: float,
    height: u.Quantity,
) -> np.ndarray:
    a = WGS84_A.to_value(u.km)
    h = height.to_value(u.km)
    denom = np.sqrt(1 - WGS84_E2 * sin_lat**2)
    n = a / denom
    x = (n + h) * cos_lat * cos_lon
    y = (n + h) * cos_lat * sin_lon
    z = (n * (1 - WGS84_E2) + h) * sin_lat
    return np.array([x, y, z])


def _sez_to_ecef_matrix(
    sin_lat: float,
    cos_lat: float,
    sin_lon: float,
    cos_lon: float,
) -> np.ndarray:
    south_hat = np.array([sin_lat * cos_lon, sin_lat * sin_lon, -cos_lat])
    east_hat = np.array([-sin_lon, cos_lon, 0.0])
    zenith_hat = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    return np.stack((south_hat, east_hat, zenith_hat), axis=-1)


def _rotation_z(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _earth_rotation_angle(epoch: Time) -> float:
    jd_ut1 = epoch.ut1.jd
    d = jd_ut1 - 2451545.0
    theta = 2 * np.pi * (0.7790572732640 + 1.00273781191135448 * d)
    return np.mod(theta, 2 * np.pi)


def _los_unit_sez(
    sin_az: float,
    cos_az: float,
    sin_el: float,
    cos_el: float,
) -> np.ndarray:
    s = -cos_el * cos_az
    e = cos_el * sin_az
    z = sin_el
    return np.array([s, e, z])


def _site_itrs(
    *,
    lon: u.Quantity,
    lat: u.Quantity,
    height: u.Quantity,
    epoch_utc: Time,
) -> tuple[ITRS, np.ndarray, np.ndarray]:
    sin_lat, cos_lat, sin_lon, cos_lon = _lat_lon_trig(lat, lon)
    site_ecef = _site_position(sin_lat, cos_lat, sin_lon, cos_lon, height)
    sez_to_ecef = _sez_to_ecef_matrix(sin_lat, cos_lat, sin_lon, cos_lon)

    rep = CartesianRepresentation(
        x=site_ecef[0] * u.km,
        y=site_ecef[1] * u.km,
        z=site_ecef[2] * u.km,
        differentials=CartesianDifferential(
            d_x=0.0 * u.km / u.s,
            d_y=0.0 * u.km / u.s,
            d_z=0.0 * u.km / u.s,
        ),
    )
    return ITRS(rep, obstime=epoch_utc), site_ecef, sez_to_ecef


def _los_eci_from_ecef_astropy(
    site_ecef: np.ndarray,
    los_ecef: np.ndarray,
    epoch_utc: Time,
) -> np.ndarray:
    p0 = ITRS(
        CartesianRepresentation(
            x=site_ecef[0] * u.km,
            y=site_ecef[1] * u.km,
            z=site_ecef[2] * u.km,
        ),
        obstime=epoch_utc,
    ).transform_to(GCRS(obstime=epoch_utc))
    p1_ecef = site_ecef + los_ecef
    p1 = ITRS(
        CartesianRepresentation(
            x=p1_ecef[0] * u.km,
            y=p1_ecef[1] * u.km,
            z=p1_ecef[2] * u.km,
        ),
        obstime=epoch_utc,
    ).transform_to(GCRS(obstime=epoch_utc))

    v = p1.cartesian.xyz.to_value(u.km) - p0.cartesian.xyz.to_value(u.km)
    norm = np.linalg.norm(v)
    if norm <= 0.0:
        raise ValueError("LOS vector norm is zero after transform")
    return v / norm


def _angles_to_site_and_los_eci(
    *,
    lon: u.Quantity,
    lat: u.Quantity,
    height: u.Quantity,
    epoch_utc: Time,
    az: u.Quantity,
    el: u.Quantity,
    method: str = "astropy",
) -> tuple[np.ndarray, np.ndarray]:
    if epoch_utc.scale != "utc":
        raise ValueError("epoch_utc must be a Time with scale='utc'")
    if method not in ("astropy", "simple"):
        raise ValueError("method must be 'astropy' or 'simple'")

    sin_az, cos_az, sin_el, cos_el = _az_el_trig(az, el)
    los_sez = _los_unit_sez(sin_az, cos_az, sin_el, cos_el)
    site_itrs, site_ecef, sez_to_ecef = _site_itrs(
        lon=lon, lat=lat, height=height, epoch_utc=epoch_utc
    )
    los_ecef = sez_to_ecef @ los_sez

    if method == "simple":
        rot = _rotation_z(-_earth_rotation_angle(epoch_utc))
        site_eci = rot @ site_ecef
        los_eci = rot @ los_ecef
        los_eci = los_eci / np.linalg.norm(los_eci)
        return site_eci, los_eci

    site_gcrs = site_itrs.transform_to(GCRS(obstime=epoch_utc))
    site_eci = site_gcrs.cartesian.xyz.to_value(u.km)
    los_eci = _los_eci_from_ecef_astropy(site_ecef, los_ecef, epoch_utc)
    return site_eci, los_eci



def _norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))



def _stumpff_c(z: float) -> float:
    if z > 1e-8:
        sz = np.sqrt(z)
        return (1.0 - np.cos(sz)) / z
    if z < -1e-8:
        sz = np.sqrt(-z)
        return (np.cosh(sz) - 1.0) / (-z)
    return 0.5 - z / 24.0 + z**2 / 720.0 - z**3 / 40320.0



def _stumpff_s(z: float) -> float:
    if z > 1e-8:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / (sz**3)
    if z < -1e-8:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / (sz**3)
    return 1.0 / 6.0 - z / 120.0 + z**2 / 5040.0 - z**3 / 362880.0



def propagate_universal(
    mu_km3_s2: float,
    r0_km: np.ndarray,
    v0_km_s: np.ndarray,
    dt_s: float,
    *,
    maxiter: int = 50,
    rtol: float = 1e-11,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a two-body state with universal variables."""
    r0 = np.asarray(r0_km, dtype=float)
    v0 = np.asarray(v0_km_s, dtype=float)

    r0n = _norm(r0)
    v0n2 = float(np.dot(v0, v0))
    sqrt_mu = np.sqrt(mu_km3_s2)
    vr0 = float(np.dot(r0, v0)) / r0n
    rv_sqrt_mu = float(np.dot(r0, v0)) / sqrt_mu
    alpha = 2.0 / r0n - v0n2 / mu_km3_s2

    if abs(alpha) > 1e-10:
        chi = np.sign(dt_s) * np.sqrt(mu_km3_s2) * abs(alpha) * abs(dt_s)
    else:
        h = np.cross(r0, v0)
        p = float(np.dot(h, h)) / mu_km3_s2
        s = 0.5 * (np.pi / 2.0 - np.arctan(3.0 * np.sqrt(mu_km3_s2 / p**3) * dt_s))
        w = np.arctan(np.tan(s) ** (1.0 / 3.0))
        chi = np.sqrt(p) * 2.0 / np.tan(2.0 * w)

    for _ in range(maxiter):
        z = alpha * chi**2
        c = _stumpff_c(z)
        s = _stumpff_s(z)

        f = (
            rv_sqrt_mu * chi**2 * c
            + (1.0 - alpha * r0n) * chi**3 * s
            + r0n * chi
            - sqrt_mu * dt_s
        )

        r = (
            chi**2 * c
            + rv_sqrt_mu * chi * (1.0 - z * s)
            + r0n * (1.0 - z * c)
        )

        dchi = f / r
        chi -= dchi
        if abs(dchi) < rtol:
            break
    else:
        raise IzzoConvergenceError("Universal-variable propagation failed to converge")

    z = alpha * chi**2
    c = _stumpff_c(z)
    s = _stumpff_s(z)

    f_lagrange = 1.0 - chi**2 * c / r0n
    g_lagrange = dt_s - (chi**3 * s) / sqrt_mu
    r_vec = f_lagrange * r0 + g_lagrange * v0

    rn = _norm(r_vec)
    fdot = sqrt_mu * (alpha * chi**3 * s - chi) / (rn * r0n)
    gdot = 1.0 - chi**2 * c / rn
    v_vec = fdot * r0 + gdot * v0

    return r_vec, v_vec



def _compute_y(x: float, lam: float) -> float:
    inside = 1.0 - lam**2 * (1.0 - x**2)
    if inside < -1e-14:
        raise ValueError("Lambert geometry produced invalid y^2 < 0")
    return np.sqrt(max(0.0, inside))



def _compute_psi(x: float, y: float, lam: float) -> float:
    if -1.0 <= x < 1.0:
        arg = x * y + lam * (1.0 - x**2)
        return float(np.arccos(np.clip(arg, -1.0, 1.0)))
    if x > 1.0:
        return float(np.arcsinh((y - x * lam) * np.sqrt(x**2 - 1.0)))
    return float(np.arccosh((y - x * lam) * np.sqrt(x**2 - 1.0)))



def _hyp2f1_3_1_5_2(x: float, *, max_terms: int = 200000) -> float:
    """Evaluate 2F1(3, 1; 5/2; x) using the poliastro Battin series form."""
    if abs(x) >= 1.0:
        raise ValueError("Hypergeometric series requires |x| < 1")

    res = 1.0
    term = 1.0
    ii = 0
    for _ in range(max_terms):
        term = term * (3.0 + ii) * (1.0 + ii) / (2.5 + ii) * x / (ii + 1.0)
        res_old = res
        res += term
        if res_old == res:
            return res
        ii += 1

    raise IzzoConvergenceError("Hypergeometric series failed to converge")



def _tof_equation_y(x: float, y: float, T_target: float, lam: float, M: int) -> float:
    if M == 0 and -1.0 < x < 1.0:
        eta = y - lam * x
        s1 = 0.5 * (1.0 - lam - x * eta)
        q = (4.0 / 3.0) * _hyp2f1_3_1_5_2(s1)
        T_eval = 0.5 * (eta**3 * q + 4.0 * lam * eta)
    else:
        psi = _compute_psi(x, y, lam)
        T_eval = (psi + M * np.pi) / np.sqrt(abs(1.0 - x**2) ** 3) - x + lam * y

    return T_eval - T_target



def _tof_derivatives(x: float, y: float, T_eval: float, lam: float) -> tuple[float, float, float]:
    one_minus_x2 = 1.0 - x**2
    lam2 = lam**2
    lam3 = lam**3
    lam5 = lam**5

    dT = (3.0 * T_eval * x - 2.0 + 2.0 * lam3 * x / y) / one_minus_x2
    d2T = (3.0 * T_eval + 5.0 * x * dT + 2.0 * (1.0 - lam2) * lam3 / y**3) / one_minus_x2
    d3T = (
        7.0 * x * d2T + 8.0 * dT - 6.0 * (1.0 - lam2) * lam5 * x / y**5
    ) / one_minus_x2
    return dT, d2T, d3T



def _initial_guess(T: float, lam: float, M: int) -> float:
    if M != 0:
        raise NotImplementedError("This implementation supports only M=0 transfers")

    T0 = np.arccos(lam) + lam * np.sqrt(1.0 - lam**2)
    T1 = 2.0 * (1.0 - lam**3) / 3.0

    if T >= T0:
        return (T0 / T) ** (2.0 / 3.0) - 1.0
    if T < T1:
        return 2.5 * T1 * (T1 - T) / (T * (1.0 - lam**5)) + 1.0
    return np.exp(np.log(2.0) * np.log(T / T0) / np.log(T1 / T0)) - 1.0



def _householder(
    x0: float,
    T: float,
    lam: float,
    M: int,
    *,
    maxiter: int,
    rtol: float,
) -> float:
    x = x0
    for _ in range(maxiter):
        y = _compute_y(x, lam)
        fval = _tof_equation_y(x, y, T, lam, M)
        T_eval = _tof_equation_y(x, y, 0.0, lam, M)
        dT, d2T, d3T = _tof_derivatives(x, y, T_eval, lam)

        denom = dT * (dT**2 - fval * d2T) + (d3T * fval**2) / 6.0
        if abs(denom) < 1e-16:
            raise IzzoConvergenceError("Householder denominator collapsed")

        step = fval * ((dT**2 - 0.5 * fval * d2T) / denom)
        x_next = x - step
        if abs(x_next - x) < rtol:
            return x_next
        x = x_next

    raise IzzoConvergenceError("Izzo Householder iteration failed to converge")


def _solve_x(T: float, lam: float, M: int, *, maxiter: int, rtol: float) -> float:
    if M != 0:
        raise NotImplementedError("This implementation supports only M=0 transfers")

    x0 = _initial_guess(T, lam, M)
    try:
        return _householder(x0, T, lam, M, maxiter=maxiter, rtol=rtol)
    except IzzoConvergenceError:
        pass

    bracket_candidates: list[tuple[float, float, float]] = []

    def _scan_brackets(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vals = np.full_like(grid, np.nan, dtype=float)
        for i, x in enumerate(grid):
            y = _compute_y(float(x), lam)
            vals[i] = _tof_equation_y(float(x), y, T, lam, M)
        for i in range(len(grid) - 1):
            f0 = vals[i]
            f1 = vals[i + 1]
            if not np.isfinite(f0) or not np.isfinite(f1):
                continue
            if f0 == 0.0:
                bracket_candidates.append((0.0, float(grid[i]), float(grid[i])))
                continue
            if f0 * f1 < 0.0:
                score = abs(f0) + abs(f1)
                bracket_candidates.append((score, float(grid[i]), float(grid[i + 1])))
        return grid, vals

    grids_and_vals = []
    grids_and_vals.append(_scan_brackets(np.linspace(-0.999999, 0.999999, 3000)))
    grids_and_vals.append(_scan_brackets(np.linspace(1.000001, 50.0, 7000)))
    grids_and_vals.append(_scan_brackets(np.linspace(-50.0, -1.000001, 7000)))

    lo = hi = None
    if bracket_candidates:
        _, lo, hi = min(bracket_candidates, key=lambda t: t[0])
        if lo == hi:
            return lo

    if lo is None or hi is None:
        all_x = np.concatenate([gv[0] for gv in grids_and_vals])
        all_f = np.concatenate([gv[1] for gv in grids_and_vals])
        finite = np.isfinite(all_f)
        if not finite.any():
            raise IzzoConvergenceError("No finite Lambert TOF values during root search")
        x = float(all_x[finite][np.argmin(np.abs(all_f[finite]))])
        for _ in range(maxiter):
            y = _compute_y(x, lam)
            fx = _tof_equation_y(x, y, T, lam, M)
            if abs(fx) < 1e-12:
                return x
            dx = 1e-6
            xp = x + dx
            xm = x - dx
            fp = _tof_equation_y(xp, _compute_y(xp, lam), T, lam, M)
            fm = _tof_equation_y(xm, _compute_y(xm, lam), T, lam, M)
            dfdx = (fp - fm) / (xp - xm)
            if abs(dfdx) < 1e-14:
                break
            x_new = x - fx / dfdx
            if abs(x_new - x) < rtol:
                return x_new
            x = x_new
        raise IzzoConvergenceError("Failed to bracket Lambert x root")

    flo = _tof_equation_y(lo, _compute_y(lo, lam), T, lam, M)
    for _ in range(maxiter * 5):
        mid = 0.5 * (lo + hi)
        fmid = _tof_equation_y(mid, _compute_y(mid, lam), T, lam, M)
        if abs(fmid) < 1e-12 or abs(hi - lo) < rtol:
            return mid
        if flo * fmid < 0.0:
            hi = mid
        else:
            lo = mid
            flo = fmid

    raise IzzoConvergenceError("Bisection failed to converge for Lambert x root")



def _lambert_izzo_local(
    mu_km3_s2: float,
    r1_km: np.ndarray,
    r2_km: np.ndarray,
    tof_s: float,
    *,
    prograde: bool = True,
    lowpath: bool = True,
    maxiter: int = 35,
    rtol: float = 1e-10,
    M: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve Lambert's problem using Izzo's x-parameter method.

    Returns initial and final inertial velocities in km/s.
    """
    if tof_s <= 0.0:
        raise ValueError("tof_s must be positive")

    if M != 0:
        raise NotImplementedError("This implementation currently supports only M=0")

    r1 = np.asarray(r1_km, dtype=float)
    r2 = np.asarray(r2_km, dtype=float)

    r1n = _norm(r1)
    r2n = _norm(r2)
    c_vec = r2 - r1
    c = _norm(c_vec)

    if c == 0.0:
        raise ValueError("Lambert geometry is singular for identical endpoints")

    s = 0.5 * (r1n + r2n + c)
    if s <= 0.0:
        raise ValueError("Invalid Lambert geometry: semiperimeter must be positive")

    ir1 = r1 / r1n
    ir2 = r2 / r2n
    ih = np.cross(ir1, ir2)
    ihn = _norm(ih)
    if ihn < 1e-12:
        raise ValueError("Lambert plane is undefined (collinear endpoints)")
    ih = ih / ihn

    lam2 = max(0.0, 1.0 - c / s)
    lam = float(np.sqrt(lam2))

    if ih[2] < 0.0:
        lam = -lam
        it1 = np.cross(ir1, ih)
        it2 = np.cross(ir2, ih)
    else:
        it1 = np.cross(ih, ir1)
        it2 = np.cross(ih, ir2)

    if not prograde:
        lam = -lam
        it1 = -it1
        it2 = -it2

    if not lowpath and M > 0:
        lam = -lam

    T = np.sqrt(2.0 * mu_km3_s2 / s**3) * tof_s
    x = _solve_x(T, lam, M, maxiter=maxiter, rtol=rtol)
    y = _compute_y(x, lam)

    gamma = np.sqrt(mu_km3_s2 * s / 2.0)
    rho = (r1n - r2n) / c
    sigma2 = max(0.0, 1.0 - rho**2)
    sigma = np.sqrt(sigma2)

    vr1 = gamma * ((lam * y - x) - rho * (lam * y + x)) / r1n
    vr2 = -gamma * ((lam * y - x) + rho * (lam * y + x)) / r2n
    vt = gamma * sigma * (y + lam * x)
    vt1 = vt / r1n
    vt2 = vt / r2n

    v1 = vr1 * ir1 + vt1 * it1
    v2 = vr2 * ir2 + vt2 * it2

    return v1, v2


def _lambert_izzo_hapsira(
    mu_km3_s2: float,
    r1_km: np.ndarray,
    r2_km: np.ndarray,
    tof_s: float,
    *,
    prograde: bool = True,
    lowpath: bool = True,
    maxiter: int = 35,
    rtol: float = 1e-10,
    M: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if _hapsira_izzo is None:
        raise ImportError(
            "hapsira is not installed. Install it to use backend='hapsira', "
            "or use backend='local'."
        )

    try:
        v1, v2 = _hapsira_izzo(
            mu_km3_s2,
            np.asarray(r1_km, dtype=float),
            np.asarray(r2_km, dtype=float),
            float(tof_s),
            M=M,
            prograde=prograde,
            lowpath=lowpath,
            numiter=maxiter,
            rtol=rtol,
        )
    except TypeError:
        # Compatibility with versions that use maxiter keyword.
        v1, v2 = _hapsira_izzo(
            mu_km3_s2,
            np.asarray(r1_km, dtype=float),
            np.asarray(r2_km, dtype=float),
            float(tof_s),
            M=M,
            prograde=prograde,
            lowpath=lowpath,
            maxiter=maxiter,
            rtol=rtol,
        )

    return np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)


def lambert_izzo(
    mu_km3_s2: float,
    r1_km: np.ndarray,
    r2_km: np.ndarray,
    tof_s: float,
    *,
    prograde: bool = True,
    lowpath: bool = True,
    maxiter: int = 35,
    rtol: float = 1e-10,
    M: int = 0,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Solve Lambert's problem with Izzo using `hapsira` or local fallback.

    `backend` options:
    - ``\"auto\"``: use `hapsira` when installed, otherwise local solver
    - ``\"hapsira\"``: require `hapsira` and raise if unavailable
    - ``\"local\"``: force local NumPy implementation
    """
    if backend not in {"auto", "hapsira", "local"}:
        raise ValueError("backend must be one of: 'auto', 'hapsira', 'local'")

    if backend in {"auto", "hapsira"}:
        try:
            return _lambert_izzo_hapsira(
                mu_km3_s2,
                r1_km,
                r2_km,
                tof_s,
                prograde=prograde,
                lowpath=lowpath,
                maxiter=maxiter,
                rtol=rtol,
                M=M,
            )
        except ImportError:
            if backend == "hapsira":
                raise

    return _lambert_izzo_local(
        mu_km3_s2,
        r1_km,
        r2_km,
        tof_s,
        prograde=prograde,
        lowpath=lowpath,
        maxiter=maxiter,
        rtol=rtol,
        M=M,
    )



def _orthonormal_perp_basis(los: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    los = los / _norm(los)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, los)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(los, helper)
    e1 /= _norm(e1)
    e2 = np.cross(los, e1)
    return e1, e2



def _guess_range_for_radius(site_km: np.ndarray, los: np.ndarray, radius_km: float) -> float:
    b = float(np.dot(site_km, los))
    c = float(np.dot(site_km, site_km) - radius_km**2)
    disc = b * b - c
    if disc < 0.0:
        return max(1000.0, -b)
    sdisc = np.sqrt(disc)
    roots = [(-b + sdisc), (-b - sdisc)]
    positive = [r for r in roots if r > 0.0]
    if positive:
        return min(positive)
    return max(1000.0, -b)



def _as_seconds(epochs: Sequence[Time] | Sequence[float]) -> tuple[float, float]:
    if len(epochs) != 3:
        raise ValueError("Need exactly 3 epochs")

    if isinstance(epochs[0], Time):
        t1 = epochs[0]
        t2 = epochs[1]
        t3 = epochs[2]
        dt12 = (t2 - t1).to_value(u.s)
        dt13 = (t3 - t1).to_value(u.s)
        return float(dt12), float(dt13)

    t = np.asarray(epochs, dtype=float)
    return float(t[1] - t[0]), float(t[2] - t[0])


def _validate_three_observations(observations: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    obs_list = list(observations)
    if len(obs_list) != 3:
        raise ValueError("Exactly 3 observations are required")
    for obs in obs_list:
        required = {"lon", "lat", "height", "epoch_utc", "az", "el"}
        missing = sorted(required - set(obs.keys()))
        if missing:
            raise ValueError(f"Observation missing keys: {missing}")
    return obs_list


def _observations_to_geometry(
    observations: Iterable[dict[str, object]],
    *,
    method: str,
) -> tuple[np.ndarray, np.ndarray, list[Time]]:
    obs_list = _validate_three_observations(observations)
    site_positions = []
    los_vectors = []
    epochs: list[Time] = []

    for obs in obs_list:
        site_eci, los_eci = _angles_to_site_and_los_eci(
            lon=obs["lon"],
            lat=obs["lat"],
            height=obs["height"],
            epoch_utc=obs["epoch_utc"],
            az=obs["az"],
            el=obs["el"],
            method=method,
        )
        site_positions.append(site_eci)
        los_vectors.append(los_eci)
        epochs.append(obs["epoch_utc"])

    return np.vstack(site_positions), np.vstack(los_vectors), epochs


def _state_vector_from_solution(solution: dict[str, object]) -> np.ndarray:
    return np.hstack((np.asarray(solution["r2_km"], dtype=float), np.asarray(solution["v2_km_s"], dtype=float)))


def _coerce_angle_covariance(
    cov_angles: np.ndarray | Sequence[np.ndarray],
    *,
    cov_in_degrees: bool,
) -> np.ndarray:
    # Supports:
    # - full (6, 6) matrix in [az1, el1, az2, el2, az3, el3] order
    # - one (2, 2) matrix to be repeated for all 3 observations
    # - sequence of 3 matrices, each (2, 2)
    cov_arr = np.asarray(cov_angles, dtype=float)

    if cov_arr.shape == (6, 6):
        cov6 = cov_arr
    elif cov_arr.shape == (2, 2):
        cov6 = np.zeros((6, 6))
        for i in range(3):
            j = 2 * i
            cov6[j : j + 2, j : j + 2] = cov_arr
    elif cov_arr.shape == (3, 2, 2):
        cov6 = np.zeros((6, 6))
        for i in range(3):
            j = 2 * i
            cov6[j : j + 2, j : j + 2] = cov_arr[i]
    else:
        raise ValueError(
            "cov_angles must have shape (6,6), (2,2), or (3,2,2) "
            "for [az1, el1, az2, el2, az3, el3]"
        )

    if cov_in_degrees:
        rad_per_deg = np.deg2rad(1.0)
        conv = np.eye(6) * rad_per_deg
        cov6 = conv @ cov6 @ conv.T

    return cov6


def _copy_observations(observations: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    # Shallow copy is sufficient because we only perturb az/el quantities.
    return [dict(obs) for obs in observations]



def solve_angles_only_los_izzo(
    *,
    site_positions_km: np.ndarray,
    los_eci: np.ndarray,
    epochs: Sequence[Time] | Sequence[float],
    mu_km3_s2: float = MU_EARTH_KM3_S2,
    initial_ranges_km: tuple[float, float] | None = None,
    range_bounds_km: tuple[float, float] = (1.0, 100000.0),
    prograde: bool = True,
    lowpath: bool = True,
    lambert_backend: str = "auto",
    maxiter: int = 30,
    tol_km: float = 1e-3,
) -> dict[str, object]:
    """Solve 3-observation angles-only IOD using Izzo Lambert inside the loop.

    The unknowns are endpoint ranges ``rho1`` and ``rho3``. At each iterate:
    - build endpoint position vectors
    - solve Lambert(r1, r3, dt13) with Izzo
    - propagate to t2 and enforce the line-of-sight constraint at observation 2

    `lambert_backend` is forwarded to :func:`lambert_izzo`.
    """
    R = np.asarray(site_positions_km, dtype=float)
    L = np.asarray(los_eci, dtype=float)

    if R.shape != (3, 3):
        raise ValueError("site_positions_km must have shape (3, 3)")
    if L.shape != (3, 3):
        raise ValueError("los_eci must have shape (3, 3)")

    L = L / np.linalg.norm(L, axis=1, keepdims=True)
    dt12, dt13 = _as_seconds(epochs)
    if dt12 <= 0.0 or dt13 <= dt12:
        raise ValueError("Epochs must be strictly increasing")

    rho_min, rho_max = range_bounds_km
    if rho_min <= 0.0 or rho_max <= rho_min:
        raise ValueError("Invalid range_bounds_km")

    if initial_ranges_km is None:
        rho1 = _guess_range_for_radius(R[0], L[0], 7000.0)
        rho3 = _guess_range_for_radius(R[2], L[2], 7000.0)
    else:
        rho1, rho3 = initial_ranges_km

    rho1 = float(np.clip(rho1, rho_min, rho_max))
    rho3 = float(np.clip(rho3, rho_min, rho_max))

    e1, e2 = _orthonormal_perp_basis(L[1])

    def residual_and_state(r1_range: float, r3_range: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        r1 = R[0] + r1_range * L[0]
        r3 = R[2] + r3_range * L[2]

        v1, v3 = lambert_izzo(
            mu_km3_s2,
            r1,
            r3,
            dt13,
            prograde=prograde,
            lowpath=lowpath,
            backend=lambert_backend,
        )
        r2_pred, v2_pred = propagate_universal(mu_km3_s2, r1, v1, dt12)

        delta = r2_pred - R[1]
        rho2 = float(np.dot(delta, L[1]))
        perp = delta - rho2 * L[1]
        residual = np.array([float(np.dot(perp, e1)), float(np.dot(perp, e2))])

        return residual, {
            "r1_km": r1,
            "r2_km": r2_pred,
            "r3_km": r3,
            "v1_km_s": v1,
            "v2_km_s": v2_pred,
            "v3_km_s": v3,
            "rho2_km": np.array([rho2]),
            "perp_km": perp,
        }

    residual, state = residual_and_state(rho1, rho3)

    for _ in range(maxiter):
        res_norm = _norm(residual)
        if res_norm < tol_km:
            break

        jac = np.zeros((2, 2))
        for j, rho in enumerate((rho1, rho3)):
            eps = max(1e-3, 1e-6 * abs(rho))
            if j == 0:
                r_plus, _ = residual_and_state(rho1 + eps, rho3)
            else:
                r_plus, _ = residual_and_state(rho1, rho3 + eps)
            jac[:, j] = (r_plus - residual) / eps

        try:
            step = np.linalg.solve(jac, -residual)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(jac, -residual, rcond=None)[0]

        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.05):
            cand1 = float(np.clip(rho1 + alpha * step[0], rho_min, rho_max))
            cand3 = float(np.clip(rho3 + alpha * step[1], rho_min, rho_max))
            cand_residual, cand_state = residual_and_state(cand1, cand3)
            if _norm(cand_residual) < res_norm:
                rho1, rho3 = cand1, cand3
                residual, state = cand_residual, cand_state
                accepted = True
                break

        if not accepted:
            raise AnglesOnlyConvergenceError(
                "Angles-only Izzo iteration stagnated; try better initial_ranges_km"
            )
    else:
        raise AnglesOnlyConvergenceError(
            "Angles-only Izzo iteration did not converge within maxiter"
        )

    rho2 = float(state["rho2_km"][0])

    return {
        "ranges_km": np.array([rho1, rho2, rho3]),
        "residual_km": residual,
        "residual_norm_km": _norm(residual),
        "r1_km": state["r1_km"],
        "r2_km": state["r2_km"],
        "r3_km": state["r3_km"],
        "v1_km_s": state["v1_km_s"],
        "v2_km_s": state["v2_km_s"],
        "v3_km_s": state["v3_km_s"],
    }



def solve_angles_only_observations_izzo(
    observations: Iterable[dict[str, object]],
    *,
    method: str = "astropy",
    mu_km3_s2: float = MU_EARTH_KM3_S2,
    initial_ranges_km: tuple[float, float] | None = None,
    range_bounds_km: tuple[float, float] = (1.0, 100000.0),
    prograde: bool = True,
    lowpath: bool = True,
    lambert_backend: str = "auto",
    maxiter: int = 30,
    tol_km: float = 1e-3,
) -> dict[str, object]:
    """High-level angles-only solver using sitetrack_anglesonly measurement geometry.

    Each observation dict must include:
    - lon, lat, height, epoch_utc, az, el

    `lambert_backend` is forwarded to :func:`solve_angles_only_los_izzo`.
    """
    site_positions, los_vectors, epochs = _observations_to_geometry(observations, method=method)

    solution = solve_angles_only_los_izzo(
        site_positions_km=site_positions,
        los_eci=los_vectors,
        epochs=epochs,
        mu_km3_s2=mu_km3_s2,
        initial_ranges_km=initial_ranges_km,
        range_bounds_km=range_bounds_km,
        prograde=prograde,
        lowpath=lowpath,
        lambert_backend=lambert_backend,
        maxiter=maxiter,
        tol_km=tol_km,
    )

    r2 = solution["r2_km"]
    v2 = solution["v2_km_s"]
    epoch2 = epochs[1]
    rep = CartesianRepresentation(r2 * u.km, differentials=CartesianDifferential(v2 * u.km / u.s))

    out = dict(solution)
    out["state_gcrs_t2"] = GCRS(rep, obstime=epoch2)
    out["state_vector_t2_km_km_s"] = _state_vector_from_solution(solution)
    return out


def estimate_state_covariance_izzo(
    observations: Iterable[dict[str, object]],
    *,
    cov_angles: np.ndarray | Sequence[np.ndarray] | None = None,
    cov_in_degrees: bool = False,
    method: str = "astropy",
    mu_km3_s2: float = MU_EARTH_KM3_S2,
    initial_ranges_km: tuple[float, float] | None = None,
    range_bounds_km: tuple[float, float] = (1.0, 100000.0),
    prograde: bool = True,
    lowpath: bool = True,
    lambert_backend: str = "auto",
    maxiter: int = 30,
    tol_km: float = 1e-3,
    jacobian_eps_rad: float = 1e-7,
    return_jacobian: bool = False,
) -> dict[str, object]:
    """Primary API: solve full state at t2 and optionally compute 6x6 covariance.

    When ``cov_angles`` is provided, it is mapped through a numerical Jacobian:
    ``P_state = J * P_angles * J^T`` where angle order is
    ``[az1, el1, az2, el2, az3, el3]`` in radians.
    """
    obs_list = _validate_three_observations(observations)
    nominal = solve_angles_only_observations_izzo(
        obs_list,
        method=method,
        mu_km3_s2=mu_km3_s2,
        initial_ranges_km=initial_ranges_km,
        range_bounds_km=range_bounds_km,
        prograde=prograde,
        lowpath=lowpath,
        lambert_backend=lambert_backend,
        maxiter=maxiter,
        tol_km=tol_km,
    )

    out = dict(nominal)
    if cov_angles is None:
        return out

    if jacobian_eps_rad <= 0.0:
        raise ValueError("jacobian_eps_rad must be positive")

    cov6 = _coerce_angle_covariance(cov_angles, cov_in_degrees=cov_in_degrees)
    jac = np.zeros((6, 6))
    pert_pairs = ((0, "az"), (0, "el"), (1, "az"), (1, "el"), (2, "az"), (2, "el"))
    eps_q = jacobian_eps_rad * u.rad
    warm_start = (float(out["ranges_km"][0]), float(out["ranges_km"][2]))

    for col, (obs_idx, key) in enumerate(pert_pairs):
        obs_plus = _copy_observations(obs_list)
        obs_minus = _copy_observations(obs_list)
        obs_plus[obs_idx][key] = obs_plus[obs_idx][key] + eps_q
        obs_minus[obs_idx][key] = obs_minus[obs_idx][key] - eps_q

        plus = solve_angles_only_observations_izzo(
            obs_plus,
            method=method,
            mu_km3_s2=mu_km3_s2,
            initial_ranges_km=warm_start,
            range_bounds_km=range_bounds_km,
            prograde=prograde,
            lowpath=lowpath,
            lambert_backend=lambert_backend,
            maxiter=maxiter,
            tol_km=tol_km,
        )
        minus = solve_angles_only_observations_izzo(
            obs_minus,
            method=method,
            mu_km3_s2=mu_km3_s2,
            initial_ranges_km=warm_start,
            range_bounds_km=range_bounds_km,
            prograde=prograde,
            lowpath=lowpath,
            lambert_backend=lambert_backend,
            maxiter=maxiter,
            tol_km=tol_km,
        )

        jac[:, col] = (
            plus["state_vector_t2_km_km_s"] - minus["state_vector_t2_km_km_s"]
        ) / (2.0 * jacobian_eps_rad)

    cov_state = jac @ cov6 @ jac.T
    out["covariance_t2_km_km_s"] = cov_state
    if return_jacobian:
        out["state_jacobian_t2_wrt_angles"] = jac
    return out


def solve_angles_only_izzo(
    observations: Iterable[dict[str, object]],
    **kwargs: object,
) -> dict[str, object]:
    """Alias for :func:`estimate_state_covariance_izzo` as the primary API."""
    return estimate_state_covariance_izzo(observations, **kwargs)


def main() -> None:
    observations = [
        {
            "lon": -100.0 * u.deg,
            "lat": 40.0 * u.deg,
            "height": 1.0 * u.km,
            "epoch_utc": Time("2020-01-01T00:00:00", scale="utc"),
            "az": 10.0 * u.deg,
            "el": 20.0 * u.deg,
        },
        {
            "lon": -99.5 * u.deg,
            "lat": 40.1 * u.deg,
            "height": 1.1 * u.km,
            "epoch_utc": Time("2020-01-01T00:02:00", scale="utc"),
            "az": 11.0 * u.deg,
            "el": 19.5 * u.deg,
        },
        {
            "lon": -99.0 * u.deg,
            "lat": 40.2 * u.deg,
            "height": 1.2 * u.km,
            "epoch_utc": Time("2020-01-01T00:04:30", scale="utc"),
            "az": 12.0 * u.deg,
            "el": 19.0 * u.deg,
        },
    ]

    out = estimate_state_covariance_izzo(
        observations,
        cov_angles=np.diag([0.01, 0.01]),
        cov_in_degrees=True,
        return_jacobian=True,
    )

    print("Ranges (km):", out["ranges_km"])
    print("Residual (km):", out["residual_km"])
    print("Residual norm (km):", out["residual_norm_km"])
    print("State vector at t2 [x y z vx vy vz] (km, km/s):")
    print(out["state_vector_t2_km_km_s"])
    print("State covariance at t2 (6x6):")
    print(out["covariance_t2_km_km_s"])
    print("State Jacobian wrt [az1 el1 az2 el2 az3 el3] (6x6):")
    print(out["state_jacobian_t2_wrt_angles"])


if __name__ == "__main__":
    main()
