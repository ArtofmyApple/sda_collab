import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import (
    AltAz,
    CartesianDifferential,
    CartesianRepresentation,
    EarthLocation,
    GCRS,
    ITRS,
    SkyCoord,
)
from astropy.time import Time

import sitetrack_anglesonly as s

pytestmark = pytest.mark.filterwarnings("ignore:leap-second file is expired.*")


def _measurement_kwargs(case: dict, *, with_rates: bool) -> dict:
    kwargs = {
        "lon": case["lon_deg"] * u.deg,
        "lat": case["lat_deg"] * u.deg,
        "height": case["height_km"] * u.km,
        "epoch_utc": Time(case["epoch_utc"], scale="utc"),
        "az": case["az_deg"] * u.deg,
        "el": case["el_deg"] * u.deg,
    }
    if with_rates:
        kwargs["az_rate"] = case["az_rate_deg_s"] * u.deg / u.s
        kwargs["el_rate"] = case["el_rate_deg_s"] * u.deg / u.s
    return kwargs


# Manual case expansion point for shared geometry-driven tests:
# add more station/measurement dictionaries here for extra manual checks.
GENERIC_MEASUREMENT_CASES = [
    {
        "id": "denver_1995",
        "lon_deg": -104.883,
        "lat_deg": 39.007,
        "height_km": 2.187,
        "epoch_utc": "1995-05-20T03:17:02",
        "az_deg": 205.6,
        "el_deg": 30.7,
        "az_rate_deg_s": 0.15,
        "el_rate_deg_s": 0.17,
    },
    {
        "id": "midlat_2012",
        "lon_deg": -175.9,
        "lat_deg": 37.8,
        "height_km": 0.0,
        "epoch_utc": "2012-10-08T19:05:15",
        "az_deg": 315.0,
        "el_deg": 45.0,
        "az_rate_deg_s": -0.2,
        "el_rate_deg_s": -0.3,
    },
    {
        "id": "highlat_2020",
        "lon_deg": 140.0,
        "lat_deg": 65.0,
        "height_km": 1.2,
        "epoch_utc": "2020-01-01T06:00:00",
        "az_deg": 300.0,
        "el_deg": 55.0,
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
    truth = np.array(
        [
            loc.x.to_value(u.km),
            loc.y.to_value(u.km),
            loc.z.to_value(u.km),
        ]
    )
    assert np.allclose(site, truth, atol=1e-6)


# Manual case expansion point for test__sez_to_ecef_matrix_known_reference_cases.
SEZ_TO_ECEF_REFERENCE_CASES = [
    {
        "lat_deg": 0.0,
        "lon_deg": 0.0,
        "expected": np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        ),
    },
    {
        "lat_deg": 0.0,
        "lon_deg": 90.0,
        "expected": np.array(
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ]
        ),
    },
]


@pytest.mark.parametrize("case", SEZ_TO_ECEF_REFERENCE_CASES)
def test__sez_to_ecef_matrix_known_reference_cases(case):
    lat_rad = np.deg2rad(case["lat_deg"])
    lon_rad = np.deg2rad(case["lon_deg"])
    mat = s._sez_to_ecef_matrix(
        np.sin(lat_rad),
        np.cos(lat_rad),
        np.sin(lon_rad),
        np.cos(lon_rad),
    )
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
    mat = s._sez_to_ecef_matrix(
        np.sin(lat_rad),
        np.cos(lat_rad),
        np.sin(lon_rad),
        np.cos(lon_rad),
    )
    assert np.allclose(mat.T @ mat, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(mat), 1.0, atol=1e-12)


# Manual case expansion point for test__rotation_z_rotates_vectors_as_expected.
ROTATION_Z_CASES = [
    {"angle": 0.0, "vec": np.array([2.0, -1.0, 0.5]), "expected": np.array([2.0, -1.0, 0.5])},
    {
        "angle": np.pi / 2,
        "vec": np.array([1.0, 0.0, 0.0]),
        "expected": np.array([0.0, 1.0, 0.0]),
    },
    {
        "angle": -np.pi / 2,
        "vec": np.array([0.0, 1.0, 0.0]),
        "expected": np.array([1.0, 0.0, 0.0]),
    },
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
    assert np.isclose(
        s._earth_rotation_angle(j2000_ut1), s._earth_rotation_angle(later), atol=1e-10
    )


# Manual case expansion point for test__los_unit_sez_known_directions_and_norm.
LOS_UNIT_CASES = [
    {"az_deg": 0.0, "el_deg": 0.0, "expected": np.array([-1.0, 0.0, 0.0])},
    {"az_deg": 90.0, "el_deg": 0.0, "expected": np.array([0.0, 1.0, 0.0])},
    {"az_deg": 180.0, "el_deg": 0.0, "expected": np.array([1.0, 0.0, 0.0])},
    {"az_deg": 33.0, "el_deg": 90.0, "expected": np.array([0.0, 0.0, 1.0])},
]


@pytest.mark.parametrize("case", LOS_UNIT_CASES)
def test__los_unit_sez_known_directions_and_norm(case):
    az_rad = np.deg2rad(case["az_deg"])
    el_rad = np.deg2rad(case["el_deg"])
    los = s._los_unit_sez(np.sin(az_rad), np.cos(az_rad), np.sin(el_rad), np.cos(el_rad))
    assert np.allclose(los, case["expected"], atol=1e-12)
    assert np.isclose(np.linalg.norm(los), 1.0, atol=1e-12)


# Manual case expansion point for test__los_rate_sez_matches_central_difference.
LOS_RATE_CASES = [
    {"az_deg": 205.6, "el_deg": 30.7, "az_rate_deg_s": 0.15, "el_rate_deg_s": 0.17},
    {"az_deg": 315.0, "el_deg": 45.0, "az_rate_deg_s": -0.2, "el_rate_deg_s": -0.3},
    {"az_deg": 120.0, "el_deg": 5.0, "az_rate_deg_s": 0.05, "el_rate_deg_s": -0.02},
]


@pytest.mark.parametrize("case", LOS_RATE_CASES)
def test__los_rate_sez_matches_central_difference(case):
    az = case["az_deg"] * u.deg
    el = case["el_deg"] * u.deg
    az_rate = case["az_rate_deg_s"] * u.deg / u.s
    el_rate = case["el_rate_deg_s"] * u.deg / u.s

    sin_az, cos_az, sin_el, cos_el = s._az_el_trig(az, el)
    analytic = s._los_rate_sez(az_rate, el_rate, sin_az, cos_az, sin_el, cos_el)

    dt = 1e-6
    az_plus = az + az_rate * dt * u.s
    az_minus = az - az_rate * dt * u.s
    el_plus = el + el_rate * dt * u.s
    el_minus = el - el_rate * dt * u.s
    sp, cp, sep, cep = s._az_el_trig(az_plus, el_plus)
    sm, cm, sem, cem = s._az_el_trig(az_minus, el_minus)
    fd = (s._los_unit_sez(sp, cp, sep, cep) - s._los_unit_sez(sm, cm, sem, cem)) / (2 * dt)

    assert np.allclose(analytic, fd, atol=1e-7)


# Manual case expansion point for test__site_itrs_matches_astropy_and_trig_override.
SITE_ITRS_CASES = [
    {"lon_deg": -104.883, "lat_deg": 39.007, "height_km": 2.187, "epoch_utc": "1995-05-20T03:17:02"},
    {"lon_deg": 140.0, "lat_deg": 65.0, "height_km": 1.2, "epoch_utc": "2020-01-01T06:00:00"},
]


@pytest.mark.parametrize("case", SITE_ITRS_CASES)
def test__site_itrs_matches_astropy_and_trig_override(case):
    epoch = Time(case["epoch_utc"], scale="utc")
    lon = case["lon_deg"] * u.deg
    lat = case["lat_deg"] * u.deg
    height = case["height_km"] * u.km

    site_no_trig, ecef_no_trig, sez_no_trig = s._site_itrs(
        lon=lon, lat=lat, height=height, epoch_utc=epoch
    )
    trig = s._lat_lon_trig(lat, lon)
    site_with_trig, ecef_with_trig, sez_with_trig = s._site_itrs(
        lon=lon, lat=lat, height=height, epoch_utc=epoch, trig=trig
    )

    assert isinstance(site_no_trig, ITRS)
    assert isinstance(site_with_trig, ITRS)
    assert np.allclose(ecef_no_trig, ecef_with_trig, atol=1e-12)
    assert np.allclose(sez_no_trig, sez_with_trig, atol=1e-12)
    assert np.allclose(site_no_trig.cartesian.xyz.to_value(u.km), ecef_no_trig, atol=1e-12)

    loc = EarthLocation.from_geodetic(lon=lon, lat=lat, height=height)
    truth = np.array([loc.x.to_value(u.km), loc.y.to_value(u.km), loc.z.to_value(u.km)])
    assert np.allclose(ecef_no_trig, truth, atol=1e-6)


@pytest.mark.parametrize(
    "case",
    GENERIC_MEASUREMENT_CASES[:2],
    ids=lambda c: c["id"],
)
def test__los_eci_from_ecef_astropy_is_unit_and_scale_stable(case):
    meas = _measurement_kwargs(case, with_rates=False)
    trig = s._lat_lon_trig(meas["lat"], meas["lon"])
    site_ecef = s._site_position(*trig, meas["height"])
    sez_to_ecef = s._sez_to_ecef_matrix(*trig)
    los_ecef = sez_to_ecef @ s._los_unit_sez(*s._az_el_trig(meas["az"], meas["el"]))

    los1 = s._los_eci_from_ecef_astropy(site_ecef, los_ecef, meas["epoch_utc"])
    los2 = s._los_eci_from_ecef_astropy(site_ecef, 2.0 * los_ecef, meas["epoch_utc"])
    assert np.isclose(np.linalg.norm(los1), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(los2), 1.0, atol=1e-12)
    assert np.allclose(los1, los2, atol=1e-8)


# Manual case expansion point for test__los_eci_from_ecef_astropy_raises_on_zero_vector.
LOS_ECI_ZERO_VECTOR_CASES = [
    {"lon_deg": -104.883, "lat_deg": 39.007, "height_km": 2.187},
    {"lon_deg": 140.0, "lat_deg": 65.0, "height_km": 1.2},
]


@pytest.mark.parametrize("case", LOS_ECI_ZERO_VECTOR_CASES)
def test__los_eci_from_ecef_astropy_raises_on_zero_vector(case):
    trig = s._lat_lon_trig(case["lat_deg"] * u.deg, case["lon_deg"] * u.deg)
    site_ecef = s._site_position(*trig, case["height_km"] * u.km)
    with pytest.raises(ValueError, match="LOS vector norm is zero after transform"):
        s._los_eci_from_ecef_astropy(
            site_ecef, np.zeros(3), Time("2020-01-01T00:00:00", scale="utc")
        )


@pytest.mark.parametrize(
    "case",
    GENERIC_MEASUREMENT_CASES[:2],
    ids=lambda c: c["id"],
)
@pytest.mark.parametrize("method", ["simple", "astropy"])
def test__angles_only_core_returns_valid_geometry(case, method):
    site_gcrs, los_eci, los_rate_eci = s._angles_only_core(
        **_measurement_kwargs(case, with_rates=True),
        method=method,
    )

    assert isinstance(site_gcrs, GCRS)
    assert np.isclose(np.linalg.norm(los_eci), 1.0, atol=1e-10)
    assert los_rate_eci is not None
    assert los_rate_eci.shape == (3,)
    assert np.isclose(np.dot(los_eci, los_rate_eci), 0.0, atol=1e-8)


@pytest.mark.parametrize(
    "case",
    GENERIC_MEASUREMENT_CASES[:2],
    ids=lambda c: c["id"],
)
def test__angles_only_core_simple_matches_manual_rotation_math(case):
    kwargs = _measurement_kwargs(case, with_rates=True)
    site_gcrs, los_eci, los_rate_eci = s._angles_only_core(**kwargs, method="simple")

    sin_az, cos_az, sin_el, cos_el = s._az_el_trig(kwargs["az"], kwargs["el"])
    los_sez = s._los_unit_sez(sin_az, cos_az, sin_el, cos_el)
    los_rate_sez = s._los_rate_sez(
        kwargs["az_rate"], kwargs["el_rate"], sin_az, cos_az, sin_el, cos_el
    )
    trig = s._lat_lon_trig(kwargs["lat"], kwargs["lon"])
    site_ecef = s._site_position(*trig, kwargs["height"])
    sez_to_ecef = s._sez_to_ecef_matrix(*trig)
    los_ecef = sez_to_ecef @ los_sez
    los_rate_ecef = sez_to_ecef @ los_rate_sez

    rot = s._rotation_z(-s._earth_rotation_angle(kwargs["epoch_utc"]))
    omega = np.array([0.0, 0.0, s.EARTH_ROT_RATE])
    expected_site_pos = rot @ site_ecef
    expected_site_vel = np.cross(omega, expected_site_pos)
    expected_los = rot @ los_ecef
    expected_los = expected_los / np.linalg.norm(expected_los)
    expected_los_rate = rot @ los_rate_ecef + np.cross(omega, expected_los)
    expected_los_rate = expected_los_rate - np.dot(expected_los_rate, expected_los) * expected_los

    site_cart = site_gcrs.represent_as(CartesianRepresentation, CartesianDifferential)
    assert np.allclose(site_cart.xyz.to_value(u.km), expected_site_pos, atol=1e-9)
    assert np.allclose(
        site_cart.differentials["s"].d_xyz.to_value(u.km / u.s), expected_site_vel, atol=1e-12
    )
    assert np.allclose(los_eci, expected_los, atol=1e-12)
    assert np.allclose(los_rate_eci, expected_los_rate, atol=1e-12)


@pytest.mark.parametrize(
    "case",
    GENERIC_MEASUREMENT_CASES[:2],
    ids=lambda c: c["id"],
)
def test__angles_only_core_astropy_matches_helper_path(case):
    kwargs = _measurement_kwargs(case, with_rates=False)
    site_gcrs, los_eci, _ = s._angles_only_core(**kwargs, method="astropy")

    trig = s._lat_lon_trig(kwargs["lat"], kwargs["lon"])
    site_itrs, site_ecef, sez_to_ecef = s._site_itrs(
        lon=kwargs["lon"],
        lat=kwargs["lat"],
        height=kwargs["height"],
        epoch_utc=kwargs["epoch_utc"],
        trig=trig,
    )
    los_sez = s._los_unit_sez(*s._az_el_trig(kwargs["az"], kwargs["el"]))
    los_ecef = sez_to_ecef @ los_sez
    expected_site = site_itrs.transform_to(GCRS(obstime=kwargs["epoch_utc"]))
    expected_los = s._los_eci_from_ecef_astropy(site_ecef, los_ecef, kwargs["epoch_utc"])

    assert np.allclose(
        site_gcrs.cartesian.xyz.to_value(u.km), expected_site.cartesian.xyz.to_value(u.km), atol=1e-9
    )
    assert np.allclose(los_eci, expected_los, atol=1e-12)


# Manual case expansion point for Vallado Example 7-2 RA/Dec cross-reference checks.
VALLADO_EX7_2_RADEC_OBS = [
    {"id": "obs3", "epoch_utc": "2012-08-20T11:40:28", "ra_deg": 0.939913, "dec_deg": 18.667717},
    {"id": "obs5", "epoch_utc": "2012-08-20T11:48:28", "ra_deg": 45.025748, "dec_deg": 35.664741},
    {"id": "obs6", "epoch_utc": "2012-08-20T11:52:28", "ra_deg": 67.886655, "dec_deg": 36.996583},
]


@pytest.mark.parametrize("obs", VALLADO_EX7_2_RADEC_OBS, ids=lambda c: c["id"])
def test_vallado_ex7_2_radec_cross_reference(obs):
    # Vallado Example 7-2 setup: 40N, 110W, 2000 m; date August 20, 2012.
    lon = -110.0 * u.deg
    lat = 40.0 * u.deg
    height = 2.0 * u.km
    delta_ut1 = -0.609641 * u.s
    delta_at = 35.0 * u.s

    location = EarthLocation.from_geodetic(lon=lon, lat=lat, height=height)
    epoch_ref = Time(obs["epoch_utc"], scale="utc")
    epoch_ref.delta_ut1_utc = delta_ut1.to_value(u.s)
    epoch_ref.delta_tai = delta_at.to_value(u.s)

    # Build topocentric RA/Dec LOS from the published table, then convert to az/el.
    topo_radec = SkyCoord(
        ra=obs["ra_deg"] * u.deg,
        dec=obs["dec_deg"] * u.deg,
        frame="cirs",
        obstime=epoch_ref,
        location=location,
    )
    altaz = topo_radec.transform_to(AltAz(obstime=epoch_ref, location=location))

    _, los_eci, _ = s._angles_only_core(
        lon=lon,
        lat=lat,
        height=height,
        epoch_utc=Time(obs["epoch_utc"], scale="utc"),
        az=altaz.az,
        el=altaz.alt,
        delta_ut1=delta_ut1,
        delta_at=delta_at,
        method="astropy",
    )

    expected_los = topo_radec.transform_to(GCRS(obstime=epoch_ref)).cartesian.xyz.value
    expected_los = expected_los / np.linalg.norm(expected_los)

    assert np.isclose(np.linalg.norm(los_eci), 1.0, atol=1e-12)
    assert np.allclose(los_eci, expected_los, atol=2e-6)


# Manual case expansion point for test__angles_only_core_rejects_non_utc_epochs.
NON_UTC_EPOCH_CASES = [
    Time("1995-05-20T03:17:02", scale="tai"),
    Time("2012-10-08T19:05:15", scale="tt"),
]


@pytest.mark.parametrize("bad_epoch", NON_UTC_EPOCH_CASES)
def test__angles_only_core_rejects_non_utc_epochs(bad_epoch):
    case = GENERIC_MEASUREMENT_CASES[0]
    kwargs = _measurement_kwargs(case, with_rates=False)
    kwargs["epoch_utc"] = bad_epoch
    with pytest.raises(ValueError, match="scale='utc'"):
        s._angles_only_core(**kwargs)


# Manual case expansion point for test__angles_only_core_rejects_mismatched_rate_inputs.
MISMATCHED_RATE_CASES = [
    {"az_rate": 0.1 * u.deg / u.s, "el_rate": None},
    {"az_rate": None, "el_rate": -0.2 * u.deg / u.s},
]


@pytest.mark.parametrize("rate_case", MISMATCHED_RATE_CASES)
def test__angles_only_core_rejects_mismatched_rate_inputs(rate_case):
    kwargs = _measurement_kwargs(GENERIC_MEASUREMENT_CASES[0], with_rates=False)
    kwargs["az_rate"] = rate_case["az_rate"]
    kwargs["el_rate"] = rate_case["el_rate"]
    with pytest.raises(ValueError, match="Provide both az_rate and el_rate"):
        s._angles_only_core(**kwargs)


# Manual case expansion point for test__angles_only_core_rejects_invalid_method.
INVALID_METHOD_CASES = ["bad", "", "eci"]


@pytest.mark.parametrize("method", INVALID_METHOD_CASES)
def test__angles_only_core_rejects_invalid_method(method):
    kwargs = _measurement_kwargs(GENERIC_MEASUREMENT_CASES[0], with_rates=False)
    with pytest.raises(ValueError, match="method must be"):
        s._angles_only_core(**kwargs, method=method)


@pytest.mark.parametrize(
    "case",
    GENERIC_MEASUREMENT_CASES[:2],
    ids=lambda c: c["id"],
)
@pytest.mark.parametrize("method", ["simple", "astropy"])
def test_jacobian_angles_to_los_eci_matches_central_difference(case, method):
    kwargs = _measurement_kwargs(case, with_rates=False)
    jac = s.jacobian_angles_to_los_eci(**kwargs, method=method)
    assert jac.shape == (3, 2)

    _, base_los, _ = s._angles_only_core(**kwargs, method=method)
    assert np.isclose(np.dot(base_los, jac[:, 0]), 0.0, atol=5e-5)
    assert np.isclose(np.dot(base_los, jac[:, 1]), 0.0, atol=5e-5)

    h = 1e-6 * u.rad
    kwargs_az_plus = {**kwargs, "az": kwargs["az"] + h}
    kwargs_az_minus = {**kwargs, "az": kwargs["az"] - h}
    kwargs_el_plus = {**kwargs, "el": kwargs["el"] + h}
    kwargs_el_minus = {**kwargs, "el": kwargs["el"] - h}
    _, los_az_plus, _ = s._angles_only_core(**kwargs_az_plus, method=method)
    _, los_az_minus, _ = s._angles_only_core(**kwargs_az_minus, method=method)
    _, los_el_plus, _ = s._angles_only_core(**kwargs_el_plus, method=method)
    _, los_el_minus, _ = s._angles_only_core(**kwargs_el_minus, method=method)

    h_val = h.to_value(u.rad)
    fd_az = (los_az_plus - los_az_minus) / (2 * h_val)
    fd_el = (los_el_plus - los_el_minus) / (2 * h_val)

    assert np.allclose(jac[:, 0], fd_az, atol=2e-4)
    assert np.allclose(jac[:, 1], fd_el, atol=2e-4)


# Manual case expansion point for covariance propagation tests.
COVARIANCE_PROPAGATION_CASES = [
    {
        "id": "rad_uncorrelated",
        "cov_angles": np.diag([1e-4, 2e-4]),
        "cov_in_degrees": False,
    },
    {
        "id": "deg_correlated",
        "cov_angles": np.array([[0.04, 0.01], [0.01, 0.09]]),
        "cov_in_degrees": True,
    },
]


@pytest.mark.parametrize(
    "cov_case",
    COVARIANCE_PROPAGATION_CASES,
    ids=lambda c: c["id"],
)
def test_covariance_angles_to_los_eci_matches_jacobian_propagation(cov_case):
    kwargs = _measurement_kwargs(GENERIC_MEASUREMENT_CASES[0], with_rates=False)
    cov = s.covariance_angles_to_los_eci(
        cov_case["cov_angles"], **kwargs, cov_in_degrees=cov_case["cov_in_degrees"]
    )

    jac = s.jacobian_angles_to_los_eci(**kwargs)
    if cov_case["cov_in_degrees"]:
        scale = np.deg2rad(1.0)
        cov_angles_rad = np.diag([scale, scale]) @ cov_case["cov_angles"] @ np.diag(
            [scale, scale]
        )
    else:
        cov_angles_rad = cov_case["cov_angles"]

    expected = jac @ cov_angles_rad @ jac.T
    assert np.allclose(cov, expected, atol=1e-10)
    assert np.allclose(cov, cov.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(cov)) >= -1e-12


# Manual case expansion point for test_covariance_angles_to_los_eci_rejects_bad_shapes.
BAD_COVARIANCE_SHAPES = [
    np.array([1.0, 2.0]),
    np.zeros((3, 3)),
    np.zeros((2, 3)),
]


@pytest.mark.parametrize("bad_cov", BAD_COVARIANCE_SHAPES)
def test_covariance_angles_to_los_eci_rejects_bad_shapes(bad_cov):
    kwargs = _measurement_kwargs(GENERIC_MEASUREMENT_CASES[0], with_rates=False)
    with pytest.raises(ValueError, match="cov_angles must be 2x2"):
        s.covariance_angles_to_los_eci(bad_cov, **kwargs)


@pytest.mark.parametrize(
    "case",
    GENERIC_MEASUREMENT_CASES[:2],
    ids=lambda c: c["id"],
)
@pytest.mark.parametrize("with_rates", [False, True])
def test_angles_only_to_eci_without_covariance_matches_core(case, with_rates):
    kwargs = _measurement_kwargs(case, with_rates=with_rates)
    out = s.angles_only_to_eci(**kwargs)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"site_gcrs", "los_eci", "los_rate_eci"}

    site_ref, los_ref, los_rate_ref = s._angles_only_core(**kwargs)
    assert np.allclose(
        out["site_gcrs"].cartesian.xyz.to_value(u.km),
        site_ref.cartesian.xyz.to_value(u.km),
        atol=1e-9,
    )
    assert np.allclose(out["los_eci"], los_ref, atol=1e-12)
    if with_rates:
        assert out["los_rate_eci"] is not None
        assert np.allclose(out["los_rate_eci"], los_rate_ref, atol=1e-12)
    else:
        assert out["los_rate_eci"] is None


@pytest.mark.parametrize(
    "cov_case",
    COVARIANCE_PROPAGATION_CASES,
    ids=lambda c: c["id"],
)
def test_angles_only_to_eci_with_covariance_returns_tuple(cov_case):
    kwargs = _measurement_kwargs(GENERIC_MEASUREMENT_CASES[0], with_rates=False)
    out, cov = s.angles_only_to_eci(
        **kwargs,
        cov_angles=cov_case["cov_angles"],
        cov_in_degrees=cov_case["cov_in_degrees"],
    )
    assert isinstance(out, dict)
    expected_cov = s.covariance_angles_to_los_eci(
        cov_case["cov_angles"], **kwargs, cov_in_degrees=cov_case["cov_in_degrees"]
    )
    assert np.allclose(cov, expected_cov, atol=1e-12)


# Manual case expansion point for test_angles_only_with_assumed_range_zero_range_matches_site_state.
ZERO_RANGE_CASES = [
    {"id": "no_rates", "with_rates": False},
    {"id": "with_rates", "with_rates": True},
]


@pytest.mark.parametrize("mode", ZERO_RANGE_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize(
    "case",
    GENERIC_MEASUREMENT_CASES[:2],
    ids=lambda c: c["id"],
)
def test_angles_only_with_assumed_range_zero_range_matches_site_state(case, mode):
    kwargs = _measurement_kwargs(case, with_rates=mode["with_rates"])
    reconstructed = s.angles_only_with_assumed_range_to_eci(
        **kwargs,
        assumed_rho=0.0 * u.km,
        assumed_rho_rate=0.0 * u.km / u.s,
    )
    site_gcrs, _, _ = s._angles_only_core(**kwargs)

    rec_cart = reconstructed.represent_as(CartesianRepresentation, CartesianDifferential)
    site_cart = site_gcrs.represent_as(CartesianRepresentation, CartesianDifferential)
    assert np.allclose(rec_cart.xyz.to_value(u.km), site_cart.xyz.to_value(u.km), atol=1e-9)
    assert np.allclose(
        rec_cart.differentials["s"].d_xyz.to_value(u.km / u.s),
        site_cart.differentials["s"].d_xyz.to_value(u.km / u.s),
        atol=1e-12,
    )


# Manual case expansion point for test_angles_only_with_assumed_range_matches_manual_reconstruction.
ASSUMED_RANGE_CASES = [
    {"rho_km": 300.0, "rho_rate_km_s": -1.5},
    {"rho_km": 1200.0, "rho_rate_km_s": 2.2},
]


@pytest.mark.parametrize("range_case", ASSUMED_RANGE_CASES)
@pytest.mark.parametrize(
    "case",
    GENERIC_MEASUREMENT_CASES[:2],
    ids=lambda c: c["id"],
)
def test_angles_only_with_assumed_range_matches_manual_reconstruction(case, range_case):
    kwargs = _measurement_kwargs(case, with_rates=True)
    reconstructed = s.angles_only_with_assumed_range_to_eci(
        **kwargs,
        assumed_rho=range_case["rho_km"] * u.km,
        assumed_rho_rate=range_case["rho_rate_km_s"] * u.km / u.s,
    )

    site_gcrs, los_eci, los_rate_eci = s._angles_only_core(**kwargs)
    site_cart = site_gcrs.represent_as(CartesianRepresentation, CartesianDifferential)
    site_pos = site_cart.xyz.to_value(u.km)
    site_vel = site_cart.differentials["s"].d_xyz.to_value(u.km / u.s)
    expected_pos = site_pos + range_case["rho_km"] * los_eci
    expected_vel = (
        site_vel
        + range_case["rho_rate_km_s"] * los_eci
        + range_case["rho_km"] * los_rate_eci
    )

    rec_cart = reconstructed.represent_as(CartesianRepresentation, CartesianDifferential)
    assert np.allclose(rec_cart.xyz.to_value(u.km), expected_pos, atol=1e-9)
    assert np.allclose(
        rec_cart.differentials["s"].d_xyz.to_value(u.km / u.s), expected_vel, atol=1e-9
    )


def _vallado_benchmark_cases():
    # Manual case expansion point for test_vallado_benchmark_cases_recover_truth_state:
    # add new benchmark entries to this list for additional manual checks.
    return [
        {
            "name": "vallado_ex7_1_1995",
            "case": dict(
                lon=-104.883 * u.deg,
                lat=39.007 * u.deg,
                height=2.187 * u.km,
                epoch_utc=Time("1995-05-20T03:17:02", scale="utc"),
                az=205.59999989525 * u.deg,
                el=30.700056859976 * u.deg,
                az_rate=0.149999998783 * u.deg / u.s,
                el_rate=0.169999703966 * u.deg / u.s,
                assumed_rho=604.680358525329 * u.km,
                assumed_rho_rate=2.080001775301 * u.km / u.s,
                method="astropy",
            ),
            "truth_recef_km": np.array([-1629.66031896, -5257.36055584, 3824.24387960]),
            "truth_vecef_km_s": np.array([-2.10270711, -1.66481320, 1.48496628]),
        },
        {
            "name": "vallado_testastf_2012_case205",
            "case": dict(
                lon=-175.900 * u.deg,
                lat=37.800 * u.deg,
                height=0.0 * u.km,
                epoch_utc=Time("2012-10-08T19:05:15", scale="utc"),
                az=315.000009727117 * u.deg,
                el=45.000100087568 * u.deg,
                az_rate=-0.199999929074 * u.deg / u.s,
                el_rate=-0.299997852105 * u.deg / u.s,
                assumed_rho=300.000466460201 * u.km,
                assumed_rho_rate=-5.000002827196 * u.km / u.s,
                method="astropy",
            ),
            "truth_recef_km": np.array([-5119.36386550, -216.57575162, 4136.46763724]),
            "truth_vecef_km_s": np.array([2.37872134, -1.02355003, -4.61624767]),
        },
    ]


@pytest.mark.parametrize(
    "benchmark",
    _vallado_benchmark_cases(),
    ids=lambda b: b["name"],
)
def test_vallado_benchmark_cases_recover_truth_state(benchmark):
    case = benchmark["case"]
    epoch_utc = case["epoch_utc"]

    target_truth_gcrs = ITRS(
        CartesianRepresentation(
            benchmark["truth_recef_km"] * u.km,
            differentials=CartesianDifferential(benchmark["truth_vecef_km_s"] * (u.km / u.s)),
        ),
        obstime=epoch_utc,
    ).transform_to(GCRS(obstime=epoch_utc))

    recovered = s.angles_only_with_assumed_range_to_eci(**case)
    recovered_cart = recovered.represent_as(CartesianRepresentation, CartesianDifferential)
    truth_cart = target_truth_gcrs.represent_as(CartesianRepresentation, CartesianDifferential)

    pos_err_m = (
        np.linalg.norm(recovered_cart.xyz.to_value(u.km) - truth_cart.xyz.to_value(u.km))
        * 1000.0
    )
    vel_err_m_s = (
        np.linalg.norm(
            recovered_cart.differentials["s"].d_xyz.to_value(u.km / u.s)
            - truth_cart.differentials["s"].d_xyz.to_value(u.km / u.s)
        )
        * 1000.0
    )

    assert pos_err_m < 1.0
    assert vel_err_m_s < 0.01


# Manual case expansion point for test_main_prints_expected_sections.
MAIN_BRANCH_CASES = [
    {"id": "dict_return", "return_tuple": False, "include_los_rate": False},
    {"id": "tuple_return", "return_tuple": True, "include_los_rate": True},
]


@pytest.mark.parametrize("case", MAIN_BRANCH_CASES, ids=lambda c: c["id"])
def test_main_prints_expected_sections(monkeypatch, capsys, case):
    fake_site = GCRS(
        CartesianRepresentation(
            np.array([1.0, 2.0, 3.0]) * u.km,
            differentials=CartesianDifferential(np.array([0.1, 0.2, 0.3]) * u.km / u.s),
        ),
        obstime=Time("2000-01-01T00:00:00", scale="utc"),
    )

    los_rate = np.array([0.01, 0.02, 0.03]) if case["include_los_rate"] else None

    def _fake_angles_only_to_eci(**_kwargs):
        payload = {"site_gcrs": fake_site, "los_eci": np.array([0.0, 0.0, 1.0]), "los_rate_eci": los_rate}
        if case["return_tuple"]:
            return payload, np.eye(3)
        return payload

    monkeypatch.setattr(s, "angles_only_to_eci", _fake_angles_only_to_eci)
    s.main()

    out = capsys.readouterr().out
    assert "Observer GCRS position:" in out
    assert "Observer GCRS velocity:" in out
    assert "LOS unit vector (GCRS axes):" in out
    assert ("LOS rate (1/s):" in out) is case["include_los_rate"]
    assert ("LOS covariance in GCRS axes:" in out) is case["return_tuple"]
