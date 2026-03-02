"""Standalone example for converting SEZ measurements into inertial states."""

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

# We use a static astropy-iers-data package from Jan 2026. Do not query internet each time this is run. 
from astropy.utils import iers
iers.conf.auto_download = False

EARTH_ROT_RATE = (7.2921150e-5 * u.rad / u.s).to_value(
    1 / u.s, equivalencies=u.dimensionless_angles()
)
WGS84_A = 6378.137 * u.km
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F) # e⊕^2


def _lat_lon_trig(lat: u.Quantity, lon: u.Quantity) -> tuple[float, float, float, float]:
    """Pre-calculate trig functions (Vallado Eq. 3-14, Eq. 7-1) once pe176r call."""
    lat_rad = lat.to_value(u.rad)
    lon_rad = lon.to_value(u.rad)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    return sin_lat, cos_lat, sin_lon, cos_lon


def _az_el_trig(az: u.Quantity, el: u.Quantity) -> tuple[float, float, float, float]:
    """Return sin/cos pairs for azimuth and elevation."""
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
    # Calculates the ECEF site position vector using the oblate Earth model. This accounts for the radius of curvature in the meridian (N or C⊕).
    a = WGS84_A.to_value(u.km)
    h = height.to_value(u.km)
    denom = np.sqrt(1 - WGS84_E2 * sin_lat**2)
    N = a / denom # C⊕ in Vallado Eq 3-7, radius of curvature in the meridian 
    x = (N + h) * cos_lat * cos_lon # x, y, z in ECEF frame defined in Eq 7-1 
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1 - WGS84_E2) + h) * sin_lat # S⊕ = C⊕ (1- e⊕^2) in Eq 3-7
    return np.array([x, y, z])


def _sez_vector(
    rho: u.Quantity,
    sin_az: float,
    cos_az: float,
    sin_el: float,
    cos_el: float,
) -> u.Quantity:
    # Converts range (ρ), azimuth (β), and elevation (el) into a Topocentric Horizon (SEZ) vector.
    # Note: Negative 's' component accounts for azimuth measured clockwise from North.
    rho_val = rho.to_value(u.km)
    s = -rho_val * cos_el * cos_az # Eq 4-4
    e = rho_val * cos_el * sin_az
    z = rho_val * sin_el
    return u.Quantity([s, e, z], u.km)


def _sez_velocity(
    rho: u.Quantity,
    rho_rate: u.Quantity,
    az_rate: u.Quantity,
    el_rate: u.Quantity,
    sin_az: float,
    cos_az: float,
    cos_el: float,
    sez_pos: np.ndarray,
) -> u.Quantity:
    # Calculates the SEZ velocity vector by substituting Cartesian SEZ (_sez_vector function) position components into Vallado's Eq (4-5).
    rho_val = max(rho.to_value(u.km), 1e-9) # Prevent divide by zero 
    rho_rate_val = rho_rate.to_value(u.km / u.s)
    az_rate_val = az_rate.to_value(1 / u.s, equivalencies=u.dimensionless_angles())
    el_rate_val = el_rate.to_value(1 / u.s, equivalencies=u.dimensionless_angles())

    s, e, z = sez_pos
    # s = - rho cos(el) cos(az)
    # e = rho cos(el) sin(az)
    # z = rho sin(el)

    # s_dot = - rho_dot cos(el) cos(az) + rho sin(el) cos(az) el_dot + rho cos(el) sin(az) az_dot 
    # s_dot, with substitution, equals the sum of the following 3 terms
    #       rho_dot / rho * (- rho cos(el) cos(az)) = - rho_dot cos(el) cos(az)
    #       z el_dot cos(az) = rho sin(el) cos(az) el_dot
    #       e az_dot = rho cos(el) sin(az) az_dot
    s_dot = (rho_rate_val / rho_val) * s + z * el_rate_val * cos_az + e * az_rate_val

    # e_dot = rho_dot cos(el) sin(az) - rho sin(el) sin(az) el_dot + rho cos(el) cos(az) az_dot
    #       rho_dot / rho * e = rho_dot cos(el) sin(az)
    #       -z el_dot sin(az) = - rho sin(el) sin(az) el_dot
    #       -s az_dot = rho cos(el) cos(az) az_dot
    e_dot = (rho_rate_val / rho_val) * e - z * el_rate_val * sin_az - s * az_rate_val

    # z_dot = rho_dot sin(el) + rho cos(el) el-dot
    #       rho_dot / rho * z = rho_dot sin(el)
    #       cos(el) rho el_dot
    z_dot = (rho_rate_val / rho_val) * z + cos_el * rho_val * el_rate_val

    return u.Quantity([s_dot, e_dot, z_dot], u.km / u.s)


def _sez_to_ecef_matrix(
    sin_lat: float,
    cos_lat: float,
    sin_lon: float,
    cos_lon: float,
) -> np.ndarray:
    # SEZ to ECEF rotation matrix. Ref: Vallado Eq. (3-28), Algorithm 51.
    south_hat = np.array([sin_lat * cos_lon, sin_lat * sin_lon, -cos_lat])
    east_hat = np.array([-sin_lon, cos_lon, 0.0])
    zenith_hat = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    return np.stack((south_hat, east_hat, zenith_hat), axis=-1)


def _rotation_z(angle: float) -> np.ndarray:
    # Standard Z-axis rotation matrix (ROT3). Ref: Vallado Eq. (3-15).
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _earth_rotation_angle(epoch: Time) -> float:
    # IAU-2000 Earth Rotation Angle (theta_ERA). Ref: Vallado Eq. (3-62).
    jd_ut1 = epoch.ut1.jd
    d = jd_ut1 - 2451545.0
    theta = 2 * np.pi * (0.7790572732640 + 1.00273781191135448 * d)
    return np.mod(theta, 2 * np.pi)


def _ecef_to_gcrs_simple(rep: CartesianRepresentation, epoch: Time) -> GCRS:
    # Transforms ECEF (ITRF) to ECI (GCRS) using a simplified Earth orientation model. Earth Precession and Nutation are ignored.
    # Position rotation uses ROT3 and Earth Rotation Angle (theta_ERA).
    # Velocity is transformed using the Transport Theorem (Kinematic Equation).
    # Reference: Vallado Section 3.4.2, Eq (3-15), Eq (3-25), and Eq (3-62).
    
    # Rotate ECEF position vector to be parallel with ECI (Vallado Section 3.4.1)
    rot = _rotation_z(-_earth_rotation_angle(epoch)) # Rot3 in Eq (3-15); epoch is theta_ERA
    pos_ecef = rep.xyz.to_value(u.km) 
    pos_eci = rot @ pos_ecef
    new_rep = CartesianRepresentation(pos_eci * u.km)

    vel = None
    if "s" in rep.differentials:
        vel = rep.differentials["s"].d_xyz.to_value(u.km / u.s)

    # Rotate velocity vector and add add Coriolis effect to account for the motion of the rotating ECEF frame, Eq (3-25)
    if vel is not None:
        vel_eci = rot @ vel # Rotated velocity, [B][P][N][R][W] v_ITRF
        vel_eci = vel_eci + np.cross(
            np.array([0.0, 0.0, EARTH_ROT_RATE]), pos_eci
        ) # [B][P][N][R][W] v_ITRF + w x r_ITRS, Eq (3-76)
        new_rep = new_rep.with_differentials(
            CartesianDifferential(vel_eci * (u.km / u.s))
        )

    return GCRS(new_rep, obstime=epoch)


def _sez_state_to_itrs(
    lon: u.Quantity,
    lat: u.Quantity,
    height: u.Quantity,
    epoch_utc: Time,
    sez_position: np.ndarray,
    sez_velocity: np.ndarray | None = None,
    trig: tuple[float, float, float, float] | None = None,
) -> tuple[ITRS, np.ndarray, np.ndarray, np.ndarray]:
    # Implements the SEZ-to-ITRS (ECEF) portion of the SITE-TRACK procedure. Vallado Section 7.2.1 and Equations (3-28), (7-1), (7-2), and (7-3).

    # Store pre-computed sin/cos
    if trig is None:
        sin_lat, cos_lat, sin_lon, cos_lon = _lat_lon_trig(lat, lon)
    else:
        sin_lat, cos_lat, sin_lon, cos_lon = trig
    site_ecef = _site_position(sin_lat, cos_lat, sin_lon, cos_lon, height) # Geocentric Site Pos
    sez_to_ecef = _sez_to_ecef_matrix(sin_lat, cos_lat, sin_lon, cos_lon) # Rotation matrix for ECEF

    r_ecef = site_ecef + sez_to_ecef @ sez_position # Eq (7-2)
    if sez_velocity is None:
        sez_velocity = np.zeros(3)
    v_ecef = sez_to_ecef @ sez_velocity # Eq (7-3) and SITETRACK algorithm. We sez_velocity, rho_dot_SEZ in Vallado notation, and apply rotation into ECEF, to obtain rho-dot_ECEF. 

    rep = CartesianRepresentation(
        x=r_ecef[0] * u.km,
        y=r_ecef[1] * u.km,
        z=r_ecef[2] * u.km,
        differentials=CartesianDifferential(
            d_x=v_ecef[0] * u.km / u.s,
            d_y=v_ecef[1] * u.km / u.s,
            d_z=v_ecef[2] * u.km / u.s,
        ),
    )

    return ITRS(rep, obstime=epoch_utc), sez_to_ecef, r_ecef, v_ecef


def _eci_state_from_ecef(
    r_ecef: np.ndarray,
    v_ecef: np.ndarray,
    epoch_utc: Time,
    method: str,
) -> np.ndarray:
    rep = CartesianRepresentation(
        x=r_ecef[0] * u.km,
        y=r_ecef[1] * u.km,
        z=r_ecef[2] * u.km,
        differentials=CartesianDifferential(
            d_x=v_ecef[0] * u.km / u.s,
            d_y=v_ecef[1] * u.km / u.s,
            d_z=v_ecef[2] * u.km / u.s,
        ),
    )

    if method == "simple":
        coord = _ecef_to_gcrs_simple(rep, epoch_utc)
    else:
        coord = ITRS(rep, obstime=epoch_utc).transform_to(GCRS(obstime=epoch_utc))

    cart = coord.cartesian
    pos = cart.xyz.to_value(u.km)
    vel = cart.differentials["s"].d_xyz.to_value(u.km / u.s)
    return np.hstack((pos, vel))


def _jacobian_ecef_to_eci(
    r_ecef: np.ndarray,
    v_ecef: np.ndarray,
    epoch_utc: Time,
    method: str,
) -> np.ndarray:
    base_state = _eci_state_from_ecef(r_ecef, v_ecef, epoch_utc, method)
    jac = np.zeros((6, 6))
    pos_eps = 1e-6  # km
    vel_eps = 1e-9  # km / s

    for i in range(3):
        delta = np.zeros(3)
        delta[i] = pos_eps
        state = _eci_state_from_ecef(r_ecef + delta, v_ecef, epoch_utc, method)
        jac[:, i] = (state - base_state) / pos_eps

    for i in range(3):
        delta = np.zeros(3)
        delta[i] = vel_eps
        state = _eci_state_from_ecef(r_ecef, v_ecef + delta, epoch_utc, method)
        jac[:, 3 + i] = (state - base_state) / vel_eps

    return jac


def measurement_to_eci(
    *,
    lon: u.Quantity,
    lat: u.Quantity,
    height: u.Quantity,
    epoch_utc: Time,
    rho: u.Quantity,
    az: u.Quantity,
    el: u.Quantity,
    rho_rate: u.Quantity | None = None,
    az_rate: u.Quantity | None = None,
    el_rate: u.Quantity | None = None,
    delta_ut1: u.Quantity | None = None,
    delta_at: u.Quantity | None = None,
    method: str = "astropy",
    cov_sez: np.ndarray | None = None,
) -> GCRS | tuple[GCRS, np.ndarray]:
    if epoch_utc.scale != "utc":
        raise ValueError("epoch_utc must be a Time with scale='utc'")

    sin_az, cos_az, sin_el, cos_el = _az_el_trig(az, el)

    sez_pos = _sez_vector(rho, sin_az, cos_az, sin_el, cos_el).to_value(u.km)
    sez_vel = None
    if (
        rho_rate is not None
        and az_rate is not None
        and el_rate is not None
    ):
        sez_vel = _sez_velocity(
            rho,
            rho_rate,
            az_rate,
            el_rate,
            sin_az,
            cos_az,
            cos_el,
            sez_pos,
        ).to_value(u.km / u.s)

    if delta_ut1 is not None:
        epoch_utc.delta_ut1_utc = delta_ut1.to_value(u.s)
    if delta_at is not None:
        epoch_utc.delta_tai = delta_at.to_value(u.s)

    trig = _lat_lon_trig(lat, lon)
    itrs, _, _, _ = _sez_state_to_itrs(
        lon=lon,
        lat=lat,
        height=height,
        epoch_utc=epoch_utc,
        sez_position=sez_pos,
        sez_velocity=sez_vel,
        trig=trig,
    )

    if method == "simple":
        state = _ecef_to_gcrs_simple(itrs.cartesian, epoch_utc)
    else:
        if method != "astropy":
            raise ValueError("method must be 'astropy' or 'simple'")
        state = itrs.transform_to(GCRS(obstime=epoch_utc))

    if cov_sez is None:
        return state

    cov_eci = covariance_sez_to_eci(
        cov_sez,
        lon=lon,
        lat=lat,
        height=height,
        epoch_utc=epoch_utc,
        rho=rho,
        az=az,
        el=el,
        rho_rate=rho_rate,
        az_rate=az_rate,
        el_rate=el_rate,
        delta_ut1=delta_ut1,
        delta_at=delta_at,
        method=method,
    )

    return state, cov_eci


def jacobian_sez_to_eci(
    *,
    lon: u.Quantity,
    lat: u.Quantity,
    height: u.Quantity,
    epoch_utc: Time,
    rho: u.Quantity,
    az: u.Quantity,
    el: u.Quantity,
    rho_rate: u.Quantity | None = None,
    az_rate: u.Quantity | None = None,
    el_rate: u.Quantity | None = None,
    delta_ut1: u.Quantity | None = None,
    delta_at: u.Quantity | None = None,
    method: str = "astropy",
) -> np.ndarray:
    if epoch_utc.scale != "utc":
        raise ValueError("epoch_utc must be a Time with scale='utc'")

    sin_az, cos_az, sin_el, cos_el = _az_el_trig(az, el)

    sez_pos = _sez_vector(rho, sin_az, cos_az, sin_el, cos_el).to_value(u.km)
    sez_vel = None
    if (
        rho_rate is not None
        and az_rate is not None
        and el_rate is not None
    ):
        sez_vel = _sez_velocity(
            rho,
            rho_rate,
            az_rate,
            el_rate,
            sin_az,
            cos_az,
            cos_el,
            sez_pos,
        ).to_value(u.km / u.s)

    if delta_ut1 is not None:
        epoch_utc.delta_ut1_utc = delta_ut1.to_value(u.s)
    if delta_at is not None:
        epoch_utc.delta_tai = delta_at.to_value(u.s)

    trig = _lat_lon_trig(lat, lon)
    itrs, sez_to_ecef, r_ecef, v_ecef = _sez_state_to_itrs(
        lon=lon,
        lat=lat,
        height=height,
        epoch_utc=epoch_utc,
        sez_position=sez_pos,
        sez_velocity=sez_vel,
        trig=trig,
    )

    jac_ecef = _jacobian_ecef_to_eci(r_ecef, v_ecef, epoch_utc, method)
    block = np.zeros((6, 6))
    block[:3, :3] = sez_to_ecef
    block[3:, 3:] = sez_to_ecef
    return jac_ecef @ block


def covariance_sez_to_eci(
    cov_sez: np.ndarray,
    *,
    lon: u.Quantity,
    lat: u.Quantity,
    height: u.Quantity,
    epoch_utc: Time,
    rho: u.Quantity,
    az: u.Quantity,
    el: u.Quantity,
    rho_rate: u.Quantity | None = None,
    az_rate: u.Quantity | None = None,
    el_rate: u.Quantity | None = None,
    delta_ut1: u.Quantity | None = None,
    delta_at: u.Quantity | None = None,
    method: str = "astropy",
) -> np.ndarray:
    cov_sez = np.asarray(cov_sez, dtype=float)
    jac = jacobian_sez_to_eci(
        lon=lon,
        lat=lat,
        height=height,
        epoch_utc=epoch_utc,
        rho=rho,
        az=az,
        el=el,
        rho_rate=rho_rate,
        az_rate=az_rate,
        el_rate=el_rate,
        delta_ut1=delta_ut1,
        delta_at=delta_at,
        method=method,
    )

    if cov_sez.shape == (3, 3):
        jac = jac[:3, :3]
    elif cov_sez.shape != (6, 6):
        raise ValueError("cov_sez must be either 3x3 or 6x6")

    return jac @ cov_sez @ jac.T

def main() -> None:
    cov_input = np.diag([0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4])
    result = measurement_to_eci(
        lon=-104.883*u.deg,
        lat=39.007*u.deg,
        height=2187*u.m,
        epoch_utc=Time('1995-05-20T03:17:02', scale='utc'),
        rho=604.68*u.km,
        az=205.6*u.deg,
        el=30.7*u.deg,
        rho_rate=2.08*u.km/u.s,
        az_rate=0.15*u.deg/u.s,
        el_rate=0.17*u.deg/u.s,
        delta_ut1=0*u.s,
        delta_at=28*u.s,
        cov_sez=cov_input,
    )

    if isinstance(result, tuple):
        measurement, cov_eci = result
    else:

        measurement, cov_eci = result, None

    cart = measurement.represent_as(CartesianRepresentation, CartesianDifferential)
    print("GCRS position:", cart.xyz.to(u.km))
    if "s" in cart.differentials:
        print("GCRS velocity:", cart.differentials["s"].d_xyz.to(u.km / u.s))

    if cov_eci is not None:
        print("ECI covariance matrix:")
        print(cov_eci)


if __name__ == "__main__":
    main()
