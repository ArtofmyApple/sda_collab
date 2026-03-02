import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    EarthLocation,
    GCRS,
    ITRS,
)
from astropy.time import Time

pytestmark = pytest.mark.filterwarnings("ignore:leap-second file is expired.*")

SITETRACK_PATH = Path(
    "/Users/blau/Library/CloudStorage/OneDrive-RANDCorporation/Projects/SDA/Angles_Only/sitetrack.py"
)
spec = importlib.util.spec_from_file_location("sitetrack_under_test", SITETRACK_PATH)
s = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(s)


def _measurement_kwargs(case: dict, *, with_rates: bool) -> dict:
    kwargs = {
        "lon": case["lon_deg"] * u.deg,
        "lat": case["lat_deg"] * u.deg,
        "height": case["height_km"] * u.km,
        "epoch_utc": Time(case["epoch_utc"], scale="utc"),
        "rho": case["rho_km"] * u.km,
        "az": case["az_deg"] * u.deg,
        "el": case["el_deg"] * u.deg,
    }
    if with_rates:
        kwargs["rho_rate"] = case["rho_rate_km_s"] * u.km / u.s
        kwargs["az_rate"] = case["az_rate_deg_s"] * u.deg / u.s
        kwargs["el_rate"] = case["el_rate_deg_s"] * u.deg / u.s
    return kwargs


# Manual case expansion point for shared measurement-driven tests:
# add more dict entries here for additional manual checks.
GENERIC_MEASUREMENT_CASES = [
    {
        "id": "denver_1995",
        "lon_deg": -104.883,
        "lat_deg": 39.007,
        "height_km": 2.187,
        "epoch_utc": "1995-05-20T03:17:02",
        "rho_km": 604.68,
        "az_deg": 205.6,
        "el_deg": 30.7,
        "rho_rate_km_s": 2.08,
        "az_rate_deg_s": 0.15,
        "el_rate_deg_s": 0.17,
    },
    {
        "id": "midlat_2012",
        "lon_deg": -175.9,
        "lat_deg": 37.8,
        "height_km": 0.0,
        "epoch_utc": "2012-10-08T19:05:15",
        "rho_km": 1000.0,
        "az_deg": 315.0,
        "el_deg": 45.0,
        "rho_rate_km_s": -1.5,
        "az_rate_deg_s": -0.2,
        "el_rate_deg_s": -0.3,
    },
    {
        "id": "highlat_2020",
        "lon_deg": 140.0,
        "lat_deg": 65.0,
        "height_km": 1.2,
        "epoch_utc": "2020-01-01T06:00:00",
        "rho_km": 2500.0,
        "az_deg": 300.0,
        "el_deg": 55.0,
        "rho_rate_km_s": 0.8,
        "az_rate_deg_s": 0.12,
        "el_rate_deg_s": -0.08,
    },
]


# Manual case expansion point for test__lat_lon_trig_returns_expected_values.
LAT_LON_TRIG_CASES = [
    {"lat_deg": 0.0, "lon_deg": 0.0},
    {"lat_deg": 45.0, "lon_deg": -90.0},
    {"lat_deg": -73.25, "lon_deg": 121.4},
]


@pytest.mark.parametrize("case", LAT_LON_TRIG_CASES)
def test__lat_lon_trig_returns_expected_values(case):
    sin_lat, cos_lat, sin_lon, cos_lon = s._lat_lon_trig(
        case["lat_deg"] * u.deg, case["lon_deg"] * u.deg
    )
    assert np.isclose(sin_lat, np.sin(np.deg2rad(case["lat_deg"])))
    assert np.isclose(cos_lat, np.cos(np.deg2rad(case["lat_deg"])))
    assert np.isclose(sin_lon, np.sin(np.deg2rad(case["lon_deg"])))
    assert np.isclose(cos_lon, np.cos(np.deg2rad(case["lon_deg"])))


# Manual case expansion point for test__az_el_trig_returns_expected_values.
AZ_EL_TRIG_CASES = [
    {"az_deg": 0.0, "el_deg": 0.0},
    {"az_deg": 90.0, "el_deg": 30.0},
    {"az_deg": 250.0, "el_deg": -12.5},
]


@pytest.mark.parametrize("case", AZ_EL_TRIG_CASES)
def test__az_el_trig_returns_expected_values(case):
    sin_az, cos_az, sin_el, cos_el = s._az_el_trig(
        case["az_deg"] * u.deg, case["el_deg"] * u.deg
    )
    assert np.isclose(sin_az, np.sin(np.deg2rad(case["az_deg"])))
    assert np.isclose(cos_az, np.cos(np.deg2rad(case["az_deg"])))
    assert np.isclose(sin_el, np.sin(np.deg2rad(case["el_deg"])))
    assert np.isclose(cos_el, np.cos(np.deg2rad(case["el_deg"])))


# Manual case expansion point for test__site_position_matches_earthlocation.
SITE_POSITION_CASES = [
    {"lon_deg": -104.883, "lat_deg": 39.007, "height_km": 2.187},
    {"lon_deg": 10.0, "lat_deg": 0.0, "height_km": 0.0},
    {"lon_deg": 145.2, "lat_deg": -61.5, "height_km": 0.75},
]


@pytest.mark.parametrize("case", SITE_POSITION_CASES)
def test__site_position_matches_earthlocation(case):
    lat_rad = np.deg2rad(case["lat_deg"])
    lon_rad = np.deg2rad(case["lon_deg"])
    site = s._site_position(
        np.sin(lat_rad),
        np.cos(lat_rad),
        np.sin(lon_rad),
        np.cos(lon_rad),
        case["height_km"] * u.km,
    )

    loc = EarthLocation.from_geodetic(
        lon=case["lon_deg"] * u.deg,
        lat=case["lat_deg"] * u.deg,
        height=case["height_km"] * u.km,
    )
    truth = np.array([loc.x.to_value(u.km), loc.y.to_value(u.km), loc.z.to_value(u.km)])
    assert np.allclose(site, truth, atol=1e-6)


# Manual case expansion point for test__sez_vector_matches_closed_form.
SEZ_VECTOR_CASES = [
    {"rho_km": 1.0, "az_deg": 0.0, "el_deg": 0.0, "expected": np.array([-1.0, 0.0, 0.0])},
    {"rho_km": 2.0, "az_deg": 90.0, "el_deg": 0.0, "expected": np.array([0.0, 2.0, 0.0])},
    {"rho_km": 5.0, "az_deg": 33.0, "el_deg": 90.0, "expected": np.array([0.0, 0.0, 5.0])},
]


@pytest.mark.parametrize("case", SEZ_VECTOR_CASES)
def test__sez_vector_matches_closed_form(case):
    sin_az, cos_az, sin_el, cos_el = s._az_el_trig(case["az_deg"] * u.deg, case["el_deg"] * u.deg)
    sez = s._sez_vector(case["rho_km"] * u.km, sin_az, cos_az, sin_el, cos_el)
    assert np.allclose(sez.to_value(u.km), case["expected"], atol=1e-12)


# Manual case expansion point for test__sez_velocity_matches_central_difference.
SEZ_VELOCITY_CASES = [
    {"rho_km": 604.68, "rho_rate": 2.08, "az": 205.6, "el": 30.7, "az_rate": 0.15, "el_rate": 0.17},
    {"rho_km": 1000.0, "rho_rate": -1.5, "az": 315.0, "el": 45.0, "az_rate": -0.2, "el_rate": -0.3},
    {"rho_km": 0.0, "rho_rate": 1.2, "az": 120.0, "el": 5.0, "az_rate": 0.05, "el_rate": -0.02},
]


@pytest.mark.parametrize("case", SEZ_VELOCITY_CASES)
def test__sez_velocity_matches_central_difference(case):
    rho = case["rho_km"] * u.km
    rho_rate = case["rho_rate"] * u.km / u.s
    az = case["az"] * u.deg
    el = case["el"] * u.deg
    az_rate = case["az_rate"] * u.deg / u.s
    el_rate = case["el_rate"] * u.deg / u.s

    sin_az, cos_az, sin_el, cos_el = s._az_el_trig(az, el)
    sez_pos = s._sez_vector(rho, sin_az, cos_az, sin_el, cos_el).to_value(u.km)
    analytic = s._sez_velocity(rho, rho_rate, az_rate, el_rate, sin_az, cos_az, cos_el, sez_pos).to_value(u.km / u.s)

    dt = 1e-6
    rho_plus = rho + rho_rate * dt * u.s
    rho_minus = rho - rho_rate * dt * u.s
    az_plus = az + az_rate * dt * u.s
    az_minus = az - az_rate * dt * u.s
    el_plus = el + el_rate * dt * u.s
    el_minus = el - el_rate * dt * u.s

    sp, cp, sep, cep = s._az_el_trig(az_plus, el_plus)
    sm, cm, sem, cem = s._az_el_trig(az_minus, el_minus)
    vec_plus = s._sez_vector(rho_plus, sp, cp, sep, cep).to_value(u.km)
    vec_minus = s._sez_vector(rho_minus, sm, cm, sem, cem).to_value(u.km)
    fd = (vec_plus - vec_minus) / (2 * dt)

    assert np.allclose(analytic, fd, atol=2e-4)


# Manual case expansion point for test__sez_to_ecef_matrix_known_reference_cases.
SEZ_TO_ECEF_REFERENCE_CASES = [
    {
        "lat_deg": 0.0,
        "lon_deg": 0.0,
        "expected": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
    },
    {
        "lat_deg": 0.0,
        "lon_deg": 90.0,
        "expected": np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]),
    },
]


@pytest.mark.parametrize("case", SEZ_TO_ECEF_REFERENCE_CASES)
def test__sez_to_ecef_matrix_known_reference_cases(case):
    lat_rad = np.deg2rad(case["lat_deg"])
    lon_rad = np.deg2rad(case["lon_deg"])
    mat = s._sez_to_ecef_matrix(np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad))
    assert np.allclose(mat, case["expected"], atol=1e-12)


# Manual case expansion point for test__sez_to_ecef_matrix_is_proper_rotation.
SEZ_TO_ECEF_ORTHONORMAL_CASES = [
    {"lat_deg": 39.007, "lon_deg": -104.883},
    {"lat_deg": -22.1, "lon_deg": 130.3},
    {"lat_deg": 65.0, "lon_deg": 10.0},
]


@pytest.mark.parametrize("case", SEZ_TO_ECEF_ORTHONORMAL_CASES)
def test__sez_to_ecef_matrix_is_proper_rotation(case):
    lat_rad = np.deg2rad(case["lat_deg"])
    lon_rad = np.deg2rad(case["lon_deg"])
    mat = s._sez_to_ecef_matrix(np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad))
    assert np.allclose(mat.T @ mat, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(mat), 1.0, atol=1e-12)


# Manual case expansion point for test__rotation_z_rotates_vectors_as_expected.
ROTATION_Z_CASES = [
    {"angle": 0.0, "vec": np.array([2.0, -1.0, 0.5]), "expected": np.array([2.0, -1.0, 0.5])},
    {"angle": np.pi / 2, "vec": np.array([1.0, 0.0, 0.0]), "expected": np.array([0.0, 1.0, 0.0])},
    {"angle": -np.pi / 2, "vec": np.array([0.0, 1.0, 0.0]), "expected": np.array([1.0, 0.0, 0.0])},
]


@pytest.mark.parametrize("case", ROTATION_Z_CASES)
def test__rotation_z_rotates_vectors_as_expected(case):
    rot = s._rotation_z(case["angle"])
    assert np.allclose(rot @ case["vec"], case["expected"], atol=1e-12)
    assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-12)


# Manual case expansion point for test__earth_rotation_angle_in_expected_range.
EARTH_ROTATION_ANGLE_CASES = [
    Time(2451545.0, format="jd", scale="ut1"),
    Time(2457754.12345, format="jd", scale="ut1"),
    Time(2460000.5, format="jd", scale="ut1"),
]


@pytest.mark.parametrize("epoch", EARTH_ROTATION_ANGLE_CASES)
def test__earth_rotation_angle_in_expected_range(epoch):
    theta = s._earth_rotation_angle(epoch)
    assert 0.0 <= theta < 2 * np.pi


def test__earth_rotation_angle_matches_reference_and_sidereal_periodicity():
    j2000_ut1 = Time(2451545.0, format="jd", scale="ut1")
    theta_ref = np.mod(2 * np.pi * 0.7790572732640, 2 * np.pi)
    assert np.isclose(s._earth_rotation_angle(j2000_ut1), theta_ref, atol=1e-12)

    sidereal_days = 1.0 / 1.00273781191135448
    later = Time(j2000_ut1.jd + sidereal_days, format="jd", scale="ut1")
    assert np.isclose(s._earth_rotation_angle(j2000_ut1), s._earth_rotation_angle(later), atol=1e-10)


@pytest.mark.parametrize("case", GENERIC_MEASUREMENT_CASES[:2], ids=lambda c: c["id"])
def test__ecef_to_gcrs_simple_matches_manual_transform(case):
    kwargs = _measurement_kwargs(case, with_rates=True)
    trig = s._lat_lon_trig(kwargs["lat"], kwargs["lon"])
    sez_pos = s._sez_vector(kwargs["rho"], *s._az_el_trig(kwargs["az"], kwargs["el"]))
    sez_vel = s._sez_velocity(
        kwargs["rho"],
        kwargs["rho_rate"],
        kwargs["az_rate"],
        kwargs["el_rate"],
        *s._az_el_trig(kwargs["az"], kwargs["el"])[0:2],
        s._az_el_trig(kwargs["az"], kwargs["el"])[3],
        sez_pos.to_value(u.km),
    )
    _, _, r_ecef, v_ecef = s._sez_state_to_itrs(
        lon=kwargs["lon"],
        lat=kwargs["lat"],
        height=kwargs["height"],
        epoch_utc=kwargs["epoch_utc"],
        sez_position=sez_pos.to_value(u.km),
        sez_velocity=sez_vel.to_value(u.km / u.s),
        trig=trig,
    )
    rep = CartesianRepresentation(r_ecef * u.km, differentials=CartesianDifferential(v_ecef * u.km / u.s))

    gcrs = s._ecef_to_gcrs_simple(rep, kwargs["epoch_utc"])

    rot = s._rotation_z(-s._earth_rotation_angle(kwargs["epoch_utc"]))
    omega = np.array([0.0, 0.0, s.EARTH_ROT_RATE])
    exp_pos = rot @ r_ecef
    exp_vel = rot @ v_ecef + np.cross(omega, exp_pos)

    cart = gcrs.represent_as(CartesianRepresentation, CartesianDifferential)
    assert np.allclose(cart.xyz.to_value(u.km), exp_pos, atol=1e-12)
    assert np.allclose(cart.differentials["s"].d_xyz.to_value(u.km / u.s), exp_vel, atol=1e-12)


@pytest.mark.parametrize("case", GENERIC_MEASUREMENT_CASES[:2], ids=lambda c: c["id"])
def test__sez_state_to_itrs_matches_direct_equations(case):
    kwargs = _measurement_kwargs(case, with_rates=True)
    sin_az, cos_az, sin_el, cos_el = s._az_el_trig(kwargs["az"], kwargs["el"])
    sez_pos = s._sez_vector(kwargs["rho"], sin_az, cos_az, sin_el, cos_el).to_value(u.km)
    sez_vel = s._sez_velocity(
        kwargs["rho"],
        kwargs["rho_rate"],
        kwargs["az_rate"],
        kwargs["el_rate"],
        sin_az,
        cos_az,
        cos_el,
        sez_pos,
    ).to_value(u.km / u.s)

    trig = s._lat_lon_trig(kwargs["lat"], kwargs["lon"])
    itrs, mat, r_ecef, v_ecef = s._sez_state_to_itrs(
        lon=kwargs["lon"],
        lat=kwargs["lat"],
        height=kwargs["height"],
        epoch_utc=kwargs["epoch_utc"],
        sez_position=sez_pos,
        sez_velocity=sez_vel,
        trig=trig,
    )

    site = s._site_position(*trig, kwargs["height"])
    assert np.allclose(r_ecef, site + mat @ sez_pos, atol=1e-12)
    assert np.allclose(v_ecef, mat @ sez_vel, atol=1e-12)
    assert isinstance(itrs, ITRS)


@pytest.mark.parametrize("case", GENERIC_MEASUREMENT_CASES[:2], ids=lambda c: c["id"])
@pytest.mark.parametrize("method", ["simple", "astropy"])
def test__eci_state_from_ecef_matches_measurement_to_eci(case, method):
    kwargs = _measurement_kwargs(case, with_rates=True)
    trig = s._lat_lon_trig(kwargs["lat"], kwargs["lon"])
    sin_az, cos_az, sin_el, cos_el = s._az_el_trig(kwargs["az"], kwargs["el"])
    sez_pos = s._sez_vector(kwargs["rho"], sin_az, cos_az, sin_el, cos_el).to_value(u.km)
    sez_vel = s._sez_velocity(
        kwargs["rho"], kwargs["rho_rate"], kwargs["az_rate"], kwargs["el_rate"], sin_az, cos_az, cos_el, sez_pos
    ).to_value(u.km / u.s)
    _, _, r_ecef, v_ecef = s._sez_state_to_itrs(
        lon=kwargs["lon"], lat=kwargs["lat"], height=kwargs["height"], epoch_utc=kwargs["epoch_utc"], sez_position=sez_pos, sez_velocity=sez_vel, trig=trig
    )

    state_vec = s._eci_state_from_ecef(r_ecef, v_ecef, kwargs["epoch_utc"], method)
    meas = s.measurement_to_eci(**kwargs, method=method)
    cart = meas.represent_as(CartesianRepresentation, CartesianDifferential)
    truth = np.hstack((cart.xyz.to_value(u.km), cart.differentials["s"].d_xyz.to_value(u.km / u.s)))

    assert np.allclose(state_vec, truth, atol=1e-8)


@pytest.mark.parametrize("case", GENERIC_MEASUREMENT_CASES[:2], ids=lambda c: c["id"])
@pytest.mark.parametrize("method", ["simple", "astropy"])
def test__jacobian_ecef_to_eci_matches_finite_difference(case, method):
    kwargs = _measurement_kwargs(case, with_rates=True)
    trig = s._lat_lon_trig(kwargs["lat"], kwargs["lon"])
    sin_az, cos_az, sin_el, cos_el = s._az_el_trig(kwargs["az"], kwargs["el"])
    sez_pos = s._sez_vector(kwargs["rho"], sin_az, cos_az, sin_el, cos_el).to_value(u.km)
    sez_vel = s._sez_velocity(
        kwargs["rho"], kwargs["rho_rate"], kwargs["az_rate"], kwargs["el_rate"], sin_az, cos_az, cos_el, sez_pos
    ).to_value(u.km / u.s)
    _, _, r_ecef, v_ecef = s._sez_state_to_itrs(
        lon=kwargs["lon"], lat=kwargs["lat"], height=kwargs["height"], epoch_utc=kwargs["epoch_utc"], sez_position=sez_pos, sez_velocity=sez_vel, trig=trig
    )

    jac = s._jacobian_ecef_to_eci(r_ecef, v_ecef, kwargs["epoch_utc"], method)
    assert jac.shape == (6, 6)

    base = s._eci_state_from_ecef(r_ecef, v_ecef, kwargs["epoch_utc"], method)
    eps_pos = 1e-6
    eps_vel = 1e-9
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps_pos
        fd = (s._eci_state_from_ecef(r_ecef + d, v_ecef, kwargs["epoch_utc"], method) - base) / eps_pos
        assert np.allclose(jac[:, i], fd, atol=1e-4)
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps_vel
        fd = (s._eci_state_from_ecef(r_ecef, v_ecef + d, kwargs["epoch_utc"], method) - base) / eps_vel
        assert np.allclose(jac[:, 3 + i], fd, atol=1e-4)


@pytest.mark.parametrize("case", GENERIC_MEASUREMENT_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("method", ["simple", "astropy"])
def test_measurement_to_eci_returns_gcrs_and_velocity(case, method):
    out = s.measurement_to_eci(**_measurement_kwargs(case, with_rates=True), method=method)
    assert isinstance(out, GCRS)
    cart = out.represent_as(CartesianRepresentation, CartesianDifferential)
    assert cart.xyz.shape == (3,)
    assert cart.differentials["s"].d_xyz.shape == (3,)


@pytest.mark.parametrize("case", GENERIC_MEASUREMENT_CASES[:2], ids=lambda c: c["id"])
def test_measurement_to_eci_rejects_non_utc_epochs(case):
    kwargs = _measurement_kwargs(case, with_rates=True)
    kwargs["epoch_utc"] = Time(case["epoch_utc"], scale="tt")
    with pytest.raises(ValueError, match="scale='utc'"):
        s.measurement_to_eci(**kwargs)


# Manual case expansion point for invalid method checks.
INVALID_METHOD_CASES = ["bad", "", "eci"]


@pytest.mark.parametrize("method", INVALID_METHOD_CASES)
def test_measurement_to_eci_rejects_invalid_method(method):
    kwargs = _measurement_kwargs(GENERIC_MEASUREMENT_CASES[0], with_rates=True)
    with pytest.raises(ValueError, match="method must be"):
        s.measurement_to_eci(**kwargs, method=method)


@pytest.mark.parametrize("case", GENERIC_MEASUREMENT_CASES[:2], ids=lambda c: c["id"])
@pytest.mark.parametrize("method", ["simple", "astropy"])
def test_jacobian_sez_to_eci_matches_central_difference(case, method):
    kwargs = _measurement_kwargs(case, with_rates=True)
    jac = s.jacobian_sez_to_eci(**kwargs, method=method)
    assert jac.shape == (6, 6)

    sin_az, cos_az, sin_el, cos_el = s._az_el_trig(kwargs["az"], kwargs["el"])
    sez_pos = s._sez_vector(kwargs["rho"], sin_az, cos_az, sin_el, cos_el).to_value(u.km)
    sez_vel = s._sez_velocity(
        kwargs["rho"], kwargs["rho_rate"], kwargs["az_rate"], kwargs["el_rate"], sin_az, cos_az, cos_el, sez_pos
    ).to_value(u.km / u.s)

    trig = s._lat_lon_trig(kwargs["lat"], kwargs["lon"])
    _, mat, r_ecef, v_ecef = s._sez_state_to_itrs(
        lon=kwargs["lon"], lat=kwargs["lat"], height=kwargs["height"], epoch_utc=kwargs["epoch_utc"], sez_position=sez_pos, sez_velocity=sez_vel, trig=trig
    )

    jac_ecef = s._jacobian_ecef_to_eci(r_ecef, v_ecef, kwargs["epoch_utc"], method)
    block = np.zeros((6, 6))
    block[:3, :3] = mat
    block[3:, 3:] = mat
    assert np.allclose(jac, jac_ecef @ block, atol=1e-10)


# Manual case expansion point for covariance propagation tests.
COVARIANCE_CASES = [
    {"id": "cov3", "cov": np.diag([0.1, 0.2, 0.3])},
    {"id": "cov6", "cov": np.diag([0.1, 0.2, 0.3, 1e-4, 2e-4, 3e-4])},
]


@pytest.mark.parametrize("cov_case", COVARIANCE_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("method", ["simple", "astropy"])
def test_covariance_sez_to_eci_matches_jacobian_propagation(cov_case, method):
    kwargs = _measurement_kwargs(GENERIC_MEASUREMENT_CASES[0], with_rates=True)
    cov = s.covariance_sez_to_eci(cov_case["cov"], **kwargs, method=method)

    jac = s.jacobian_sez_to_eci(**kwargs, method=method)
    if cov_case["cov"].shape == (3, 3):
        expected = jac[:3, :3] @ cov_case["cov"] @ jac[:3, :3].T
    else:
        expected = jac @ cov_case["cov"] @ jac.T
    assert np.allclose(cov, expected, atol=1e-10)
    assert np.allclose(cov, cov.T, atol=1e-12)


# Manual case expansion point for covariance shape errors.
BAD_COVARIANCE_SHAPES = [
    np.array([1.0, 2.0]),
    np.zeros((2, 2)),
    np.zeros((4, 4)),
]


@pytest.mark.parametrize("bad_cov", BAD_COVARIANCE_SHAPES)
def test_covariance_sez_to_eci_rejects_bad_shapes(bad_cov):
    kwargs = _measurement_kwargs(GENERIC_MEASUREMENT_CASES[0], with_rates=True)
    with pytest.raises(ValueError, match="cov_sez must be either 3x3 or 6x6"):
        s.covariance_sez_to_eci(bad_cov, **kwargs)


@pytest.mark.parametrize("method", ["simple", "astropy"])
def test_measurement_to_eci_with_covariance_returns_tuple(method):
    kwargs = _measurement_kwargs(GENERIC_MEASUREMENT_CASES[0], with_rates=True)
    cov_in = np.diag([0.1, 0.2, 0.3, 1e-4, 2e-4, 3e-4])
    state, cov = s.measurement_to_eci(**kwargs, method=method, cov_sez=cov_in)
    assert isinstance(state, GCRS)
    assert cov.shape == (6, 6)


@pytest.mark.parametrize("case", GENERIC_MEASUREMENT_CASES[:2], ids=lambda c: c["id"])
def test_jacobian_sez_to_eci_rejects_non_utc_epochs(case):
    kwargs = _measurement_kwargs(case, with_rates=True)
    kwargs["epoch_utc"] = Time(case["epoch_utc"], scale="tai")
    with pytest.raises(ValueError, match="scale='utc'"):
        s.jacobian_sez_to_eci(**kwargs)


def test_main_runs_and_prints(capsys):
    s.main()
    out = capsys.readouterr().out
    assert "GCRS position:" in out
    assert "GCRS velocity:" in out
    assert "ECI covariance matrix:" in out
