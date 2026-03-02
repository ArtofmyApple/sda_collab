import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation, GCRS, SkyCoord
from astropy.time import Time

import sitetrack_anglesonly_izzo as iz
import sitetrack_anglesonly as base


def test_lambert_izzo_matches_reference_case():
    # Common Lambert benchmark from Vallado/Poliastro examples.
    mu = 398600.4418
    r1 = np.array([5000.0, 10000.0, 2100.0])
    r2 = np.array([-14600.0, 2500.0, 7000.0])
    tof = 3600.0

    v1, v2 = iz.lambert_izzo(mu, r1, r2, tof, prograde=True, backend="local")

    v1_ref = np.array([-5.99249503, 1.92536671, 3.24563805])
    v2_ref = np.array([-3.31245851, -4.19661901, -0.38528906])

    assert np.allclose(v1, v1_ref, atol=2e-6)
    assert np.allclose(v2, v2_ref, atol=2e-6)



def test_angles_only_los_izzo_recovers_synthetic_mid_state():
    mu = 398600.4418

    # Truth state at middle epoch.
    r2_truth = np.array([7000.0, 1000.0, 1300.0])
    v2_truth = np.array([-1.5, 7.2, 1.0])

    # Observation timings (seconds).
    t1 = 0.0
    t2 = 120.0
    t3 = 270.0

    r1_truth, _ = iz.propagate_universal(mu, r2_truth, v2_truth, t1 - t2)
    r3_truth, _ = iz.propagate_universal(mu, r2_truth, v2_truth, t3 - t2)

    site_positions = np.array(
        [
            [6378.0, 0.0, 0.0],
            [0.0, 6378.0, 0.0],
            [-4000.0, 4500.0, 1200.0],
        ]
    )

    los = np.vstack(
        [
            (r1_truth - site_positions[0]) / np.linalg.norm(r1_truth - site_positions[0]),
            (r2_truth - site_positions[1]) / np.linalg.norm(r2_truth - site_positions[1]),
            (r3_truth - site_positions[2]) / np.linalg.norm(r3_truth - site_positions[2]),
        ]
    )

    rho1_true = np.linalg.norm(r1_truth - site_positions[0])
    rho3_true = np.linalg.norm(r3_truth - site_positions[2])

    solution = iz.solve_angles_only_los_izzo(
        site_positions_km=site_positions,
        los_eci=los,
        epochs=[t1, t2, t3],
        mu_km3_s2=mu,
        initial_ranges_km=(1.1 * rho1_true, 0.9 * rho3_true),
        tol_km=1e-5,
        maxiter=40,
    )

    assert np.linalg.norm(solution["r2_km"] - r2_truth) < 2e-3
    assert np.linalg.norm(solution["v2_km_s"] - v2_truth) < 2e-5
    assert solution["residual_norm_km"] < 1e-5


def test_angles_only_los_izzo_vallado_example_7_2_obs_3_5_6():
    # Vallado Example 7-2 setup.
    lon = -110.0 * u.deg
    lat = 40.0 * u.deg
    height = 2.0 * u.km
    delta_ut1 = -0.609641 * u.s
    delta_at = 35.0 * u.s

    # Published topocentric right ascension/declination observations.
    obs = [
        ("2012-08-20T11:40:28", 0.939913, 18.667717),   # obs 3
        ("2012-08-20T11:48:28", 45.025748, 35.664741),  # obs 5
        ("2012-08-20T11:52:28", 67.886655, 36.996583),  # obs 6
    ]

    location = EarthLocation.from_geodetic(lon=lon, lat=lat, height=height)
    site_positions = []
    los_vectors = []
    epochs = []

    for epoch_iso, ra_deg, dec_deg in obs:
        epoch = Time(epoch_iso, scale="utc")
        epoch.delta_ut1_utc = delta_ut1.to_value(u.s)
        epoch.delta_tai = delta_at.to_value(u.s)

        topo_radec = SkyCoord(
            ra=ra_deg * u.deg,
            dec=dec_deg * u.deg,
            frame="cirs",
            obstime=epoch,
            location=location,
        )
        los_eci = topo_radec.transform_to(GCRS(obstime=epoch)).cartesian.xyz.value
        los_eci = los_eci / np.linalg.norm(los_eci)

        site_gcrs = location.get_itrs(obstime=epoch).transform_to(GCRS(obstime=epoch))
        site_positions.append(site_gcrs.cartesian.xyz.to_value(u.km))
        los_vectors.append(los_eci)
        epochs.append(epoch)

    site_positions_km = np.asarray(site_positions)
    los_eci = np.asarray(los_vectors)

    solution = iz.solve_angles_only_los_izzo(
        site_positions_km=site_positions_km,
        los_eci=los_eci,
        epochs=epochs,
        tol_km=1e-3,
        maxiter=60,
    )

    assert solution["residual_norm_km"] < 1e-3

    reconstructed = np.vstack([solution["r1_km"], solution["r2_km"], solution["r3_km"]])
    los_from_solution = reconstructed - site_positions_km
    los_from_solution /= np.linalg.norm(los_from_solution, axis=1, keepdims=True)
    assert np.allclose(los_from_solution, los_eci, atol=5e-7)


def test_lambert_izzo_auto_matches_local_without_hapsira():
    if iz._hapsira_izzo is not None:
        pytest.skip("hapsira is installed; auto backend may differ slightly from local")

    mu = 398600.4418
    r1 = np.array([5000.0, 10000.0, 2100.0])
    r2 = np.array([-14600.0, 2500.0, 7000.0])
    tof = 3600.0

    v1_auto, v2_auto = iz.lambert_izzo(mu, r1, r2, tof, backend="auto")
    v1_local, v2_local = iz.lambert_izzo(mu, r1, r2, tof, backend="local")

    assert np.allclose(v1_auto, v1_local, atol=1e-12)
    assert np.allclose(v2_auto, v2_local, atol=1e-12)


def test_lambert_izzo_hapsira_backend_raises_if_missing():
    if iz._hapsira_izzo is not None:
        pytest.skip("hapsira is installed in this environment")

    mu = 398600.4418
    r1 = np.array([5000.0, 10000.0, 2100.0])
    r2 = np.array([-14600.0, 2500.0, 7000.0])
    tof = 3600.0

    with pytest.raises(ImportError):
        iz.lambert_izzo(mu, r1, r2, tof, backend="hapsira")


@pytest.mark.parametrize("method", ["astropy", "simple"])
def test_local_conversion_pipeline_matches_sitetrack_anglesonly(method):
    lon = -104.883 * u.deg
    lat = 39.007 * u.deg
    height = 2.187 * u.km
    epoch_utc = Time("1995-05-20T03:17:02", scale="utc")
    az = 205.6 * u.deg
    el = 30.7 * u.deg

    site_local, los_local = iz._angles_to_site_and_los_eci(
        lon=lon, lat=lat, height=height, epoch_utc=epoch_utc, az=az, el=el, method=method
    )
    site_base, los_base, _ = base._angles_only_core(
        lon=lon, lat=lat, height=height, epoch_utc=epoch_utc, az=az, el=el, method=method
    )
    site_base_xyz = site_base.cartesian.xyz.to_value(u.km)

    assert np.allclose(site_local, site_base_xyz, atol=1e-9)
    assert np.allclose(los_local, los_base, atol=1e-12)


def test_estimate_state_covariance_izzo_propagates_linearized_model(monkeypatch):
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

    base_state = np.array([7000.0, 1200.0, 800.0, -1.5, 7.1, 0.9])
    model_jac = np.array(
        [
            [10.0, 0.2, -0.1, 0.0, 0.0, 0.0],
            [0.1, 9.5, 0.0, -0.2, 0.0, 0.0],
            [0.0, 0.0, 8.0, 0.1, -0.1, 0.0],
            [0.02, 0.01, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.02, 0.01, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.02, 0.01, 0.0, 0.0],
        ]
    )

    def _fake_solve(obs, **_kwargs):
        ang = np.array(
            [
                obs[0]["az"].to_value(u.rad),
                obs[0]["el"].to_value(u.rad),
                obs[1]["az"].to_value(u.rad),
                obs[1]["el"].to_value(u.rad),
                obs[2]["az"].to_value(u.rad),
                obs[2]["el"].to_value(u.rad),
            ]
        )
        state = base_state + model_jac @ ang
        return {
            "ranges_km": np.array([1000.0, 1100.0, 1200.0]),
            "r2_km": state[:3],
            "v2_km_s": state[3:],
            "state_vector_t2_km_km_s": state,
        }

    monkeypatch.setattr(iz, "solve_angles_only_observations_izzo", _fake_solve)

    cov_angles_deg = np.diag([0.01, 0.01])
    out = iz.estimate_state_covariance_izzo(
        observations,
        cov_angles=cov_angles_deg,
        cov_in_degrees=True,
        return_jacobian=True,
    )

    cov6 = np.zeros((6, 6))
    for i in range(3):
        j = 2 * i
        cov6[j : j + 2, j : j + 2] = cov_angles_deg
    rad_per_deg = np.deg2rad(1.0)
    conv = np.eye(6) * rad_per_deg
    cov6 = conv @ cov6 @ conv.T
    expected_cov = model_jac @ cov6 @ model_jac.T

    assert np.allclose(out["state_jacobian_t2_wrt_angles"], model_jac, atol=5e-6)
    assert np.allclose(out["covariance_t2_km_km_s"], expected_cov, atol=1e-10)
    assert np.allclose(
        out["state_vector_t2_km_km_s"],
        _fake_solve(observations)["state_vector_t2_km_km_s"],
        atol=1e-12,
    )
