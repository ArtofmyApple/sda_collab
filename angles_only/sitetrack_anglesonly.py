"""Angles-only SITE-TRACK utilities for SEZ -> inertial line-of-sight conversion.

This module mirrors the structure of sitetrack.py, but handles angles-only
measurements (azimuth/elevation, optional angle rates). Without range/range-rate,
the observable is a line of sight, not a full Cartesian state.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    GCRS,
    ITRS,
)
from astropy.time import Time
from astropy.utils import iers

# Use local IERS data only.
iers.conf.auto_download = False

EARTH_ROT_RATE = (7.2921150e-5 * u.rad / u.s).to_value(
    1 / u.s, equivalencies=u.dimensionless_angles()
)
WGS84_A = 6378.137 * u.km
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


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
    # Unit line-of-sight in SEZ (Vallado pg 434). 
    s = -cos_el * cos_az
    e = cos_el * sin_az
    z = sin_el
    return np.array([s, e, z])


def _los_rate_sez(
    az_rate: u.Quantity,
    el_rate: u.Quantity,
    sin_az: float,
    cos_az: float,
    sin_el: float,
    cos_el: float,
) -> np.ndarray:
    az_rate_val = az_rate.to_value(1 / u.s, equivalencies=u.dimensionless_angles())
    el_rate_val = el_rate.to_value(1 / u.s, equivalencies=u.dimensionless_angles())

    s_dot = sin_el * cos_az * el_rate_val + cos_el * sin_az * az_rate_val
    e_dot = -sin_el * sin_az * el_rate_val + cos_el * cos_az * az_rate_val
    z_dot = cos_el * el_rate_val
    return np.array([s_dot, e_dot, z_dot])


def _site_itrs(
    *,
    lon: u.Quantity,
    lat: u.Quantity,
    height: u.Quantity,
    epoch_utc: Time,
    trig: tuple[float, float, float, float] | None = None,
) -> tuple[ITRS, np.ndarray, np.ndarray]:
    if trig is None:
        sin_lat, cos_lat, sin_lon, cos_lon = _lat_lon_trig(lat, lon)
    else:
        sin_lat, cos_lat, sin_lon, cos_lon = trig

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
    # Convert LOS direction by transforming two nearby points.
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
    if norm <= 0:
        raise ValueError("LOS vector norm is zero after transform")
    return v / norm


def _angles_only_core(
    *,
    lon: u.Quantity,
    lat: u.Quantity,
    height: u.Quantity,
    epoch_utc: Time,
    az: u.Quantity,
    el: u.Quantity,
    az_rate: u.Quantity | None = None,
    el_rate: u.Quantity | None = None,
    delta_ut1: u.Quantity | None = None,
    delta_at: u.Quantity | None = None,
    method: str = "astropy",
) -> tuple[GCRS, np.ndarray, np.ndarray | None]:
    if epoch_utc.scale != "utc":
        raise ValueError("epoch_utc must be a Time with scale='utc'")
    if (az_rate is None) != (el_rate is None):
        raise ValueError("Provide both az_rate and el_rate, or neither")
    if method not in ("astropy", "simple"):
        raise ValueError("method must be 'astropy' or 'simple'")

    if delta_ut1 is not None:
        epoch_utc.delta_ut1_utc = delta_ut1.to_value(u.s)
    if delta_at is not None:
        epoch_utc.delta_tai = delta_at.to_value(u.s)

    sin_az, cos_az, sin_el, cos_el = _az_el_trig(az, el)
    los_sez = _los_unit_sez(sin_az, cos_az, sin_el, cos_el)
    los_rate_sez = None
    if az_rate is not None and el_rate is not None:
        los_rate_sez = _los_rate_sez(
            az_rate, el_rate, sin_az, cos_az, sin_el, cos_el
        )

    trig = _lat_lon_trig(lat, lon)
    site_itrs, site_ecef, sez_to_ecef = _site_itrs(
        lon=lon, lat=lat, height=height, epoch_utc=epoch_utc, trig=trig
    )
    los_ecef = sez_to_ecef @ los_sez

    if method == "simple":
        rot = _rotation_z(-_earth_rotation_angle(epoch_utc))
        site_eci = rot @ site_ecef
        site_vel_eci = np.cross(np.array([0.0, 0.0, EARTH_ROT_RATE]), site_eci)

        site_gcrs = GCRS(
            CartesianRepresentation(
                site_eci * u.km,
                differentials=CartesianDifferential(site_vel_eci * u.km / u.s),
            ),
            obstime=epoch_utc,
        )

        los_eci = rot @ los_ecef
        los_eci = los_eci / np.linalg.norm(los_eci)

        los_rate_eci = None
        if los_rate_sez is not None:
            los_rate_ecef = sez_to_ecef @ los_rate_sez
            los_rate_eci = rot @ los_rate_ecef + np.cross(
                np.array([0.0, 0.0, EARTH_ROT_RATE]), los_eci
            )
            # Keep derivative orthogonal to unit vector.
            los_rate_eci = los_rate_eci - np.dot(los_rate_eci, los_eci) * los_eci

        return site_gcrs, los_eci, los_rate_eci

    site_gcrs = site_itrs.transform_to(GCRS(obstime=epoch_utc))
    los_eci = _los_eci_from_ecef_astropy(site_ecef, los_ecef, epoch_utc)
    los_rate_eci = None

    if los_rate_sez is not None:
        # Centered finite-difference LOS rate in GCRS so astropy Earth orientation is respected.
        dt_fd = 0.1  # seconds
        epoch_plus = epoch_utc + dt_fd * u.s
        epoch_minus = epoch_utc - dt_fd * u.s
        az_plus = az + az_rate * dt_fd * u.s
        az_minus = az - az_rate * dt_fd * u.s
        el_plus = el + el_rate * dt_fd * u.s
        el_minus = el - el_rate * dt_fd * u.s
        _, los_eci_plus, _ = _angles_only_core(
            lon=lon,
            lat=lat,
            height=height,
            epoch_utc=epoch_plus,
            az=az_plus,
            el=el_plus,
            delta_ut1=delta_ut1,
            delta_at=delta_at,
            method="astropy",
        )
        _, los_eci_minus, _ = _angles_only_core(
            lon=lon,
            lat=lat,
            height=height,
            epoch_utc=epoch_minus,
            az=az_minus,
            el=el_minus,
            delta_ut1=delta_ut1,
            delta_at=delta_at,
            method="astropy",
        )
        los_rate_eci = (los_eci_plus - los_eci_minus) / (2.0 * dt_fd)
        los_rate_eci = los_rate_eci - np.dot(los_rate_eci, los_eci) * los_eci

    return site_gcrs, los_eci, los_rate_eci


def jacobian_angles_to_los_eci(
    *,
    lon: u.Quantity,
    lat: u.Quantity,
    height: u.Quantity,
    epoch_utc: Time,
    az: u.Quantity,
    el: u.Quantity,
    delta_ut1: u.Quantity | None = None,
    delta_at: u.Quantity | None = None,
    method: str = "astropy",
) -> np.ndarray:
    # Numerical Jacobian d(los_eci) / d([az, el]_rad).
    eps = 1e-7 * u.rad
    _, base_los, _ = _angles_only_core(
        lon=lon,
        lat=lat,
        height=height,
        epoch_utc=epoch_utc,
        az=az,
        el=el,
        delta_ut1=delta_ut1,
        delta_at=delta_at,
        method=method,
    )
    _, los_az, _ = _angles_only_core(
        lon=lon,
        lat=lat,
        height=height,
        epoch_utc=epoch_utc,
        az=az + eps,
        el=el,
        delta_ut1=delta_ut1,
        delta_at=delta_at,
        method=method,
    )
    _, los_el, _ = _angles_only_core(
        lon=lon,
        lat=lat,
        height=height,
        epoch_utc=epoch_utc,
        az=az,
        el=el + eps,
        delta_ut1=delta_ut1,
        delta_at=delta_at,
        method=method,
    )

    jac = np.zeros((3, 2))
    eps_val = eps.to_value(u.rad)
    jac[:, 0] = (los_az - base_los) / eps_val
    jac[:, 1] = (los_el - base_los) / eps_val
    return jac


def covariance_angles_to_los_eci(
    cov_angles: np.ndarray,
    *,
    lon: u.Quantity,
    lat: u.Quantity,
    height: u.Quantity,
    epoch_utc: Time,
    az: u.Quantity,
    el: u.Quantity,
    cov_in_degrees: bool = False,
    delta_ut1: u.Quantity | None = None,
    delta_at: u.Quantity | None = None,
    method: str = "astropy",
) -> np.ndarray:
    cov_angles = np.asarray(cov_angles, dtype=float)
    if cov_angles.shape != (2, 2):
        raise ValueError("cov_angles must be 2x2 (az, el)")

    if cov_in_degrees:
        scale = np.deg2rad(1.0)
        conv = np.diag([scale, scale])
        cov_angles = conv @ cov_angles @ conv.T

    jac = jacobian_angles_to_los_eci(
        lon=lon,
        lat=lat,
        height=height,
        epoch_utc=epoch_utc,
        az=az,
        el=el,
        delta_ut1=delta_ut1,
        delta_at=delta_at,
        method=method,
    )
    return jac @ cov_angles @ jac.T


def angles_only_to_eci(
    *,
    lon: u.Quantity,
    lat: u.Quantity,
    height: u.Quantity,
    epoch_utc: Time,
    az: u.Quantity,
    el: u.Quantity,
    az_rate: u.Quantity | None = None,
    el_rate: u.Quantity | None = None,
    delta_ut1: u.Quantity | None = None,
    delta_at: u.Quantity | None = None,
    method: str = "astropy",
    cov_angles: np.ndarray | None = None,
    cov_in_degrees: bool = False,
) -> dict[str, object] | tuple[dict[str, object], np.ndarray]:
    """Convert angles-only measurement to inertial LOS geometry.

    Returns a dict with:
    - site_gcrs: GCRS coordinate of observing site
    - los_eci: unit LOS vector in GCRS axes
    - los_rate_eci: LOS time derivative (1/s), or None when no angle rates
    """
    site_gcrs, los_eci, los_rate_eci = _angles_only_core(
        lon=lon,
        lat=lat,
        height=height,
        epoch_utc=epoch_utc,
        az=az,
        el=el,
        az_rate=az_rate,
        el_rate=el_rate,
        delta_ut1=delta_ut1,
        delta_at=delta_at,
        method=method,
    )

    result = {
        "site_gcrs": site_gcrs,
        "los_eci": los_eci,
        "los_rate_eci": los_rate_eci,
    }

    if cov_angles is None:
        return result

    cov_los_eci = covariance_angles_to_los_eci(
        cov_angles,
        lon=lon,
        lat=lat,
        height=height,
        epoch_utc=epoch_utc,
        az=az,
        el=el,
        cov_in_degrees=cov_in_degrees,
        delta_ut1=delta_ut1,
        delta_at=delta_at,
        method=method,
    )
    return result, cov_los_eci


def angles_only_with_assumed_range_to_eci(
    *,
    lon: u.Quantity,
    lat: u.Quantity,
    height: u.Quantity,
    epoch_utc: Time,
    az: u.Quantity,
    el: u.Quantity,
    assumed_rho: u.Quantity,
    assumed_rho_rate: u.Quantity = 0.0 * u.km / u.s,
    az_rate: u.Quantity | None = None,
    el_rate: u.Quantity | None = None,
    delta_ut1: u.Quantity | None = None,
    delta_at: u.Quantity | None = None,
    method: str = "astropy",
) -> GCRS:
    """Build an inertial state from angles-only plus assumed range information."""
    site_gcrs, los_eci, los_rate_eci = _angles_only_core(
        lon=lon,
        lat=lat,
        height=height,
        epoch_utc=epoch_utc,
        az=az,
        el=el,
        az_rate=az_rate,
        el_rate=el_rate,
        delta_ut1=delta_ut1,
        delta_at=delta_at,
        method=method,
    )

    cart_site = site_gcrs.represent_as(CartesianRepresentation, CartesianDifferential)
    site_pos = cart_site.xyz.to_value(u.km)
    site_vel = cart_site.differentials["s"].d_xyz.to_value(u.km / u.s)

    rho = assumed_rho.to_value(u.km)
    rho_rate = assumed_rho_rate.to_value(u.km / u.s)

    if los_rate_eci is None:
        los_rate_eci = np.zeros(3)

    pos = site_pos + rho * los_eci
    vel = site_vel + rho_rate * los_eci + rho * los_rate_eci

    rep = CartesianRepresentation(
        pos * u.km,
        differentials=CartesianDifferential(vel * (u.km / u.s)),
    )
    return GCRS(rep, obstime=epoch_utc)


def main() -> None:
    result = angles_only_to_eci(
        lon=-104.883 * u.deg,
        lat=39.007 * u.deg,
        height=2187 * u.m,
        epoch_utc=Time("1995-05-20T03:17:02", scale="utc"),
        az=205.6 * u.deg,
        el=30.7 * u.deg,
        az_rate=0.15 * u.deg / u.s,
        el_rate=0.17 * u.deg / u.s,
        delta_ut1=0 * u.s,
        delta_at=28 * u.s,
        cov_angles=np.diag([0.1, 0.1]),
        cov_in_degrees=True,
    )

    if isinstance(result, tuple):
        data, cov_los = result
    else:
        data, cov_los = result, None

    site = data["site_gcrs"].represent_as(CartesianRepresentation, CartesianDifferential)
    print("Observer GCRS position:", site.xyz.to(u.km))
    print("Observer GCRS velocity:", site.differentials["s"].d_xyz.to(u.km / u.s))
    print("LOS unit vector (GCRS axes):", data["los_eci"])
    if data["los_rate_eci"] is not None:
        print("LOS rate (1/s):", data["los_rate_eci"])
    if cov_los is not None:
        print("LOS covariance in GCRS axes:")
        print(cov_los)


if __name__ == "__main__":
    main()
