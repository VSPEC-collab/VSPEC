#!/usr/bin/env python

"""
Tests for `VSPEC.geometry` module
"""

import astropy.units as u
import pytest
import numpy as np
from matplotlib.figure import Figure
from pathlib import Path

from VSPEC.geometry import SystemGeometry


def compare_angles(angle1, angle2, abs=None):
    """
    Use pytest to compare two angles, return `True` if they are equal.
    """
    delta_angle = (np.abs(angle1-angle2)+180*u.deg) % (360*u.deg) - 180*u.deg
    return delta_angle.to_value(u.deg) == pytest.approx(0, abs=abs.to_value(u.deg))


def test_compare_angles():
    """
    Run tests for `compare_angles()`
    """
    assert compare_angles(0*u.deg, 0*u.deg, abs=1*u.deg)
    assert compare_angles(30*u.deg, 31*u.deg, abs=2*u.deg)
    assert compare_angles(230*u.deg, 231*u.deg, abs=2*u.deg)
    assert compare_angles(180*u.deg, 181*u.deg, abs=2*u.deg)
    assert not compare_angles(0*u.deg, 1*u.deg, abs=0.1*u.deg)
    assert compare_angles(0*u.deg, 359*u.deg, abs=2*u.deg)
    assert compare_angles(0*u.deg, -1*u.deg, abs=2*u.deg)


def test_default_init():
    """
    Test `__init__` for `SystemGeometry` with default parameters.
    """
    geo = SystemGeometry()
    assert geo.inclination == 0*u.deg
    assert geo.init_stellar_lon == 0*u.deg
    assert geo.init_planet_phase == 0*u.deg
    assert geo.stellar_period == 80*u.day
    assert geo.orbital_period == 11*u.day
    assert geo.semimajor_axis == 0.05*u.AU
    assert geo.planetary_rot_period == 11*u.day
    assert geo.planetary_init_substellar_lon == 0*u.deg
    assert geo.alpha == 0*u.deg
    assert geo.beta == 0*u.deg
    assert geo.eccentricity == 0
    assert geo.phase_of_periasteron == 0*u.deg
    assert geo.system_distance == 1.3*u.pc
    assert geo.obliquity == 0*u.deg
    assert geo.obliquity_direction == 0*u.deg


def test_custon_init():
    """
    Test `__init__` for `SystemGeometry` with custom parameters.
    """
    geo = SystemGeometry(
        inclination=30*u.deg,
        init_stellar_lon=45*u.deg,
        init_planet_phase=90*u.deg,
        stellar_period=40*u.day,
        orbital_period=20*u.day,
        semimajor_axis=0.1*u.AU,
        planetary_rot_period=20*u.day,
        planetary_init_substellar_lon=60*u.deg,
        stellar_offset_amp=15*u.deg,
        stellar_offset_phase=75*u.deg,
        eccentricity=0.5,
        phase_of_periasteron=120*u.deg,
        system_distance=2.0*u.pc,
        obliquity=45*u.deg,
        obliquity_direction=30*u.deg
    )
    assert geo.inclination == 30*u.deg
    assert geo.init_stellar_lon == 45*u.deg
    assert geo.init_planet_phase == 90*u.deg
    assert geo.stellar_period == 40*u.day
    assert geo.orbital_period == 20*u.day
    assert geo.semimajor_axis == 0.1*u.AU
    assert geo.planetary_rot_period == 20*u.day
    assert geo.planetary_init_substellar_lon == 60*u.deg
    assert geo.alpha == 15*u.deg
    assert geo.beta == 75*u.deg
    assert geo.eccentricity == 0.5
    assert geo.phase_of_periasteron == 120*u.deg
    assert geo.system_distance == 2.0*u.pc
    assert geo.obliquity == 45*u.deg
    assert geo.obliquity_direction == 30*u.deg


def get_sub_obs_test(inclination: u.Quantity):
    """
    Run tests for `SystemGeometry.sub_obs()` for a single inclination
    """
    geo = SystemGeometry(inclination=inclination)
    time = 0*u.s
    init_lon = geo.init_stellar_lon
    assert geo.sub_obs(time)['lon'].to_value(u.deg) == pytest.approx(init_lon.to_value(u.deg), rel=1e-6)
    assert geo.sub_obs(time)['lat'].to_value(u.deg) == pytest.approx(-1*(90*u.deg-inclination).to_value(u.deg), rel=1e-6)
    time = 0.1*geo.stellar_period
    coords = geo.sub_obs(time)
    assert coords['lon'].to_value(u.deg) == pytest.approx(
        ((init_lon-0.1*360*u.deg) % (360*u.deg)).value, rel=1e-6)
    assert coords['lat'].to_value(u.deg) == pytest.approx(-1 *
                                                           (90*u.deg-inclination).to_value(u.deg), rel=1e-6)
    time = 0.5*geo.stellar_period
    coords = geo.sub_obs(time)
    assert coords['lon'].to_value(u.deg) == pytest.approx(
        ((init_lon-0.5*360*u.deg) % (360*u.deg)).value, rel=1e-6)
    assert coords['lat'].to_value(u.deg) == pytest.approx(-1 *
                                                           (90*u.deg-inclination).to_value(u.deg), rel=1e-6)
    time = geo.stellar_period
    coords = geo.sub_obs(time)
    assert coords['lon'].to_value(u.deg) == pytest.approx(init_lon, rel=1e-6)
    assert coords['lat'].to_value(u.deg) == pytest.approx(-1 *
                                                           (90*u.deg-inclination).to_value(u.deg), rel=1e-6)


def test_sub_obs():
    """
    Run tests for `SystemGeometry.sub_obs()`
    """
    for i in [0, 30, 45, 60, 90, 120, 135, 150, 180]*u.deg:
        get_sub_obs_test(i)

    geo = SystemGeometry(stellar_offset_amp=90*u.deg,
                         stellar_offset_phase=0*u.deg, inclination=90*u.deg)
    time = 0*u.s
    coords = geo.sub_obs(time)
    assert coords['lat'].to_value(u.deg) == pytest.approx(90, rel=1e-6)
    time = 0.4*geo.stellar_period
    coords = geo.sub_obs(time)
    assert coords['lat'].to_value(u.deg) == pytest.approx(90, rel=1e-6)

    geo = SystemGeometry(stellar_offset_amp=45*u.deg,
                         stellar_offset_phase=90*u.deg, inclination=90*u.deg)
    time = 0*u.s
    coords = geo.sub_obs(time)
    assert coords['lat'].to_value(u.deg) == pytest.approx(0, rel=1e-6)
    time = 0.4*geo.stellar_period
    coords = geo.sub_obs(time)
    assert coords['lat'].to_value(u.deg) == pytest.approx(0, rel=1e-6)

    geo = SystemGeometry(stellar_offset_amp=90*u.deg,
                         stellar_offset_phase=180*u.deg, inclination=90*u.deg)
    time = 0*u.s
    coords = geo.sub_obs(time)
    assert coords['lat'].to_value(u.deg) == pytest.approx(-90, rel=1e-6)
    time = 0.4*geo.stellar_period
    coords = geo.sub_obs(time)
    assert coords['lat'].to_value(u.deg) == pytest.approx(-90, rel=1e-6)


def test_mean_motion():
    """
    Run tests for `SystemGeometry.mean_motion()`
    """
    geo = SystemGeometry()
    assert geo.mean_motion().to_value(u.deg/u.day) == pytest.approx(
        (360*u.deg/geo.orbital_period).to_value(u.deg/u.day), rel=1e-6)


def test_mean_anomaly():
    """
    Run tests for `SystemGeometry.mean_anomaly()`
    """
    geo = SystemGeometry(init_planet_phase=0*u.deg,
                         phase_of_periasteron=0*u.deg)
    for time in np.linspace(0, 2, 11):
        assert (geo.mean_anomaly(time*geo.orbital_period)).to_value(u.deg) == pytest.approx(360 * (time % 1), rel=1e-6)


def test_eccentric_anomaly():
    """
    Run tests for `SystemGeometry.eccentric_anomaly()`
    """
    geo = SystemGeometry(init_planet_phase=0*u.deg,
                         phase_of_periasteron=0*u.deg, eccentricity=0)
    for time in np.linspace(0, 2, 11):
        assert geo.eccentric_anomaly(time*geo.orbital_period).to_value(u.deg) == pytest.approx(
            geo.mean_anomaly(time*geo.orbital_period).to_value(u.deg), rel=1e-6)
    eccentricity = 0.5
    geo = SystemGeometry(init_planet_phase=0*u.deg,
                         phase_of_periasteron=0*u.deg, eccentricity=eccentricity)
    for time in np.linspace(0, 2, 11):
        mean_anom = geo.mean_anomaly(time*geo.orbital_period)
        eccentric_anom = geo.eccentric_anomaly(time*geo.orbital_period)
        lhs = eccentric_anom.to_value(u.rad) - eccentricity * \
            np.sin(eccentric_anom).to_value(u.dimensionless_unscaled)
        rhs = mean_anom.to_value(u.rad)
        assert lhs == pytest.approx(rhs, rel=1e-6)


def test_true_anomaly():
    """
    Run tests for `SystemGeometry.true_anomaly()`
    """
    geo = SystemGeometry(init_planet_phase=0*u.deg,
                         phase_of_periasteron=0*u.deg, eccentricity=0)
    for time in np.linspace(0, 2, 11):
        true_anomaly = geo.true_anomaly(time*geo.orbital_period)
        mean_anomaly = geo.mean_anomaly(time*geo.orbital_period)
        assert true_anomaly.to_value(u.deg) == pytest.approx(
            mean_anomaly.to_value(u.deg), rel=1e-6)
    eccentricity = 0.5
    geo = SystemGeometry(init_planet_phase=0*u.deg,
                         phase_of_periasteron=0*u.deg, eccentricity=eccentricity)
    for time in np.linspace(0, 2, 11):
        true_anomaly = geo.true_anomaly(time*geo.orbital_period)
        eccentric_anomaly = geo.eccentric_anomaly(time*geo.orbital_period)
        lhs = np.tan(eccentric_anomaly).to_value(u.dimensionless_unscaled)
        numerator = np.sqrt(1-eccentricity**2) * np.sin(true_anomaly)
        denominator = eccentricity + np.cos(true_anomaly)
        rhs = (numerator/denominator).to_value(u.dimensionless_unscaled)
        assert lhs == pytest.approx(rhs, rel=1e-6)


def test_phase():
    """
    Run tests for `SystemGeometry.true_anomaly()`
    """
    geo = SystemGeometry(init_planet_phase=0*u.deg,
                         phase_of_periasteron=0*u.deg, eccentricity=0)
    for time in np.linspace(0, 2, 11):
        phase = geo.phase(time*geo.orbital_period)
        assert phase.to_value(u.deg) == pytest.approx(360*(time % 1), 1e-6)

    geo = SystemGeometry(init_planet_phase=0*u.deg,
                         phase_of_periasteron=0*u.deg, eccentricity=0)
    for time in np.linspace(0, 2, 11):
        phase = geo.phase(time*geo.orbital_period)
        true_anomaly = geo.true_anomaly(time*geo.orbital_period)
        assert phase.to_value(u.deg) == pytest.approx(
            true_anomaly.to_value(u.deg))
    for init_planet_phase in np.linspace(0, 360, 4)*u.deg:
        for phase_of_periasteron in np.linspace(0, 360, 7)*u.deg:
            geo = SystemGeometry(init_planet_phase=init_planet_phase,
                                 phase_of_periasteron=phase_of_periasteron, eccentricity=0)
            for time in np.linspace(0, 2, 11):
                phase = geo.phase(time*geo.orbital_period)
                true_anomaly = geo.true_anomaly(time*geo.orbital_period)
                assert phase.to_value(u.deg) == pytest.approx(
                    (true_anomaly+phase_of_periasteron).to_value(u.deg) % 360, rel=1e-6)


def test_sub_planet():
    """
    Run tests for `SystemGeometry.sub_planet()`
    """
    geo = SystemGeometry()
    time = 3*u.day
    coords1 = geo.sub_planet(time)
    coords2 = geo.sub_planet(time, phase=geo.phase(time))
    assert coords1['lat'].to_value(u.deg) == pytest.approx(
        coords2['lat'].to_value(u.deg), rel=1e-6)
    assert coords1['lon'].to_value(u.deg) == pytest.approx(
        coords2['lon'].to_value(u.deg), rel=1e-6)

    geo = SystemGeometry(phase_of_periasteron=180*u.deg,
                         init_planet_phase=180*u.deg,
                         init_stellar_lon=0*u.deg, inclination=90*u.deg)
    sub_obs = geo.sub_obs(0*u.s)
    sub_pl = geo.sub_planet(0*u.s)
    assert sub_pl['lon'].to_value(u.deg) == pytest.approx(
        sub_obs['lon'].to_value(u.deg), abs=1e-6)
    assert sub_pl['lat'].to_value(u.deg) == pytest.approx(
        sub_obs['lat'].to_value(u.deg), abs=1e-6)

    geo = SystemGeometry(phase_of_periasteron=180*u.deg,
                         init_planet_phase=180*u.deg,
                         init_stellar_lon=0*u.deg, inclination=90*u.deg,
                         stellar_offset_amp=45*u.deg, stellar_offset_phase=0*u.deg)
    sub_obs = geo.sub_obs(0*u.s)
    sub_pl = geo.sub_planet(0*u.s)
    assert sub_pl['lon'].to_value(u.deg) == pytest.approx(
        sub_obs['lon'].to_value(u.deg), abs=1e-6)
    assert sub_pl['lat'].to_value(u.deg) == pytest.approx(
        sub_obs['lat'].to_value(u.deg), abs=1e-6)

    geo = SystemGeometry(phase_of_periasteron=180*u.deg,
                         init_planet_phase=180*u.deg,
                         init_stellar_lon=0*u.deg, inclination=90*u.deg,
                         stellar_offset_amp=45*u.deg, stellar_offset_phase=90*u.deg)
    sub_obs = geo.sub_obs(0*u.s)
    sub_pl = geo.sub_planet(0*u.s)
    assert sub_pl['lon'].to_value(u.deg) == pytest.approx(
        sub_obs['lon'].to_value(u.deg), abs=1e-6)
    assert sub_pl['lat'].to_value(u.deg) == pytest.approx(
        sub_obs['lat'].to_value(u.deg), abs=1e-6)


def test_get_time_since_periasteron():
    """
    Run tests for `SystemGeometry.get_time_since_periasteron()`
    """
    geo = SystemGeometry(phase_of_periasteron=0*u.deg,
                         init_planet_phase=0*u.deg)
    assert geo.get_time_since_periasteron(0*u.deg).to_value(u.s) == pytest.approx(0, abs=10)
    assert geo.get_time_since_periasteron(180*u.deg).to_value(u.s) == pytest.approx((0.5*geo.orbital_period).to_value(u.s), abs=10)

    geo = SystemGeometry(phase_of_periasteron=90*u.deg,
                         init_planet_phase=90*u.deg)
    assert geo.get_time_since_periasteron(0*u.deg).to_value(u.s) == pytest.approx((0.75*geo.orbital_period).to_value(u.s), abs=10)
    assert geo.get_time_since_periasteron(180*u.deg).to_value(u.s) == pytest.approx((0.25*geo.orbital_period).to_value(u.s), abs=10)


def test_get_substellar_lon_at_periasteron():
    """
    Run tests for `SystemGeometry.get_time_since_periasteron()`
    """
    geo = SystemGeometry(phase_of_periasteron=0*u.deg,
                         init_planet_phase=0*u.deg,
                         planetary_init_substellar_lon=0*u.deg,
                         orbital_period=10*u.day,
                         planetary_rot_period=10*u.day)
    assert geo.get_substellar_lon_at_periasteron().to_value(u.deg) == pytest.approx(0, abs=1e-6)

    geo = SystemGeometry(phase_of_periasteron=90*u.deg,
                         init_planet_phase=270*u.deg,
                         planetary_init_substellar_lon=0*u.deg,
                         orbital_period=10*u.day,
                         planetary_rot_period=10*u.day)
    assert geo.get_substellar_lon_at_periasteron().to_value(u.deg) == pytest.approx(0, abs=1e-6)

    geo = SystemGeometry(phase_of_periasteron=90*u.deg,
                         init_planet_phase=270*u.deg,
                         planetary_init_substellar_lon=0*u.deg,
                         orbital_period=10*u.day,
                         planetary_rot_period=5*u.day)
    assert geo.get_substellar_lon_at_periasteron().to_value(u.deg) == pytest.approx(180, abs=1e-6)

    geo = SystemGeometry(phase_of_periasteron=0*u.deg,
                         init_planet_phase=50*u.deg,
                         planetary_init_substellar_lon=90*u.deg,
                         orbital_period=10*u.day,
                         planetary_rot_period=10*u.day)
    assert geo.get_substellar_lon_at_periasteron().to_value(u.deg) == pytest.approx(90, abs=1e-6)


def test_get_substellar_lon():
    """
    Run tests for `SystemGeometry.get_substellar_lon()`
    """
    geo = SystemGeometry(phase_of_periasteron=0*u.deg,
                         init_planet_phase=0*u.deg,
                         planetary_init_substellar_lon=0*u.deg,
                         orbital_period=10*u.day,
                         planetary_rot_period=10*u.day)
    for time in np.linspace(0, 1, 4):
        lon = geo.get_substellar_lon(time*geo.orbital_period)
        pred = geo.planetary_init_substellar_lon
        assert compare_angles(lon, pred, abs=1e-6*u.deg)

    geo = SystemGeometry(phase_of_periasteron=70*u.deg,
                         init_planet_phase=70*u.deg,
                         planetary_init_substellar_lon=330*u.deg,
                         orbital_period=10*u.day,
                         planetary_rot_period=10*u.day)
    for time in np.linspace(0, 1, 4):
        lon = geo.get_substellar_lon(time*geo.orbital_period)
        pred = geo.planetary_init_substellar_lon
        assert compare_angles(lon, pred, abs=1e-6*u.deg)
    geo = SystemGeometry(obliquity=10*u.deg)
    with pytest.raises(NotImplementedError):
        geo.get_substellar_lon(0*u.s)


def test_get_substellar_lat():
    """
    Run tests for `SystemGeometry.get_substellar_lat()`
    """
    geo = SystemGeometry(phase_of_periasteron=0*u.deg,
                         init_planet_phase=0*u.deg,
                         planetary_init_substellar_lon=0*u.deg,
                         orbital_period=10*u.day,
                         planetary_rot_period=10*u.day)
    for phase in np.linspace(0, 360, 4)*u.deg:
        lon = geo.get_substellar_lat(phase)
        pred = 0*u.deg
        assert compare_angles(lon, pred, abs=1e-6*u.deg)
    geo = SystemGeometry(obliquity=10*u.deg)
    with pytest.raises(NotImplementedError):
        geo.get_substellar_lat(0*u.deg)


def test_get_pl_sub_obs_lon():
    """
    Run tests for `SystemGeometry.get_pl_sub_obs_lon()`
    """
    geo = SystemGeometry(phase_of_periasteron=0*u.deg,
                         init_planet_phase=0*u.deg,
                         planetary_init_substellar_lon=0*u.deg,
                         orbital_period=10*u.day,
                         planetary_rot_period=10*u.day)
    lon = geo.get_pl_sub_obs_lon(0*u.s, 0*u.deg)
    assert compare_angles(
        lon, geo.planetary_init_substellar_lon, abs=1e-6*u.deg)
    lon = geo.get_pl_sub_obs_lon(geo.orbital_period*0.5, 180*u.deg)
    assert compare_angles(
        lon, geo.planetary_init_substellar_lon+180*u.deg, abs=1e-6*u.deg)

    geo = SystemGeometry(phase_of_periasteron=0*u.deg,
                         init_planet_phase=90*u.deg,
                         planetary_init_substellar_lon=0*u.deg,
                         orbital_period=10*u.day,
                         planetary_rot_period=10*u.day)
    lon = geo.get_pl_sub_obs_lon(0*u.s, 0*u.deg)
    assert compare_angles(
        lon, geo.planetary_init_substellar_lon, abs=1e-6*u.deg)
    lon = geo.get_pl_sub_obs_lon(geo.orbital_period*0.5, 180*u.deg)
    assert compare_angles(
        lon, geo.planetary_init_substellar_lon+180*u.deg, abs=1e-6*u.deg)

    geo = SystemGeometry(phase_of_periasteron=90*u.deg,
                         init_planet_phase=0*u.deg,
                         planetary_init_substellar_lon=0*u.deg,
                         orbital_period=10*u.day,
                         planetary_rot_period=10*u.day)
    lon = geo.get_pl_sub_obs_lon(geo.orbital_period*0.75, 0*u.deg)
    assert compare_angles(
        lon, geo.planetary_init_substellar_lon, abs=1e-6*u.deg)
    lon = geo.get_pl_sub_obs_lon(geo.orbital_period*0.25, 180*u.deg)
    assert compare_angles(
        lon, geo.planetary_init_substellar_lon+180*u.deg, abs=1e-6*u.deg)


def test_get_pl_sub_obs_lat():
    """
    Run tests for `SystemGeometry.get_pl_sub_obs_lat()`
    """
    geo = SystemGeometry(inclination=90*u.deg)
    for phase in np.linspace(0, 360, 4)*u.deg:
        lat = geo.get_pl_sub_obs_lat(phase)
        assert compare_angles(lat, 0*u.deg, 1e-6*u.deg)

    geo = SystemGeometry(inclination=0*u.deg)
    for phase in np.linspace(0, 360, 4)*u.deg:
        lat = geo.get_pl_sub_obs_lat(phase)
        assert compare_angles(lat, -90*u.deg, 1e-6*u.deg)

    geo = SystemGeometry(inclination=180*u.deg)
    for phase in np.linspace(0, 360, 4)*u.deg:
        lat = geo.get_pl_sub_obs_lat(phase)
        assert compare_angles(lat, 90*u.deg, 1e-6*u.deg)
    geo = SystemGeometry(inclination=0*u.deg, obliquity=10*u.deg)
    with pytest.raises(NotImplementedError):
        geo.get_pl_sub_obs_lat(0*u.deg)


def test_get_radius_coefficient():
    """
    Run tests for `SystemGeometry.get_radius_coefficient()`
    """
    geo = SystemGeometry(eccentricity=0)
    for phase in np.linspace(0, 360, 4)*u.deg:
        assert geo.get_radius_coeff(phase) == pytest.approx(1, rel=1e-6)
    geo = SystemGeometry(eccentricity=0.1, phase_of_periasteron=0*u.deg)
    assert geo.get_radius_coeff(geo.phase_of_periasteron) < geo.get_radius_coeff(
        geo.phase_of_periasteron+180*u.deg)

def test_get_observation_plan():
    """
    Run tests for `SystemGeometry.get_observation_plan()`
    """
    geo = SystemGeometry()
    start_times = np.linspace(0, 10, 10)*u.day
    plan = geo.get_observation_plan(start_times=start_times)
    for col in plan.colnames:
        assert len(plan[col])==10

def cartopy_installed()->bool:
    try:
        import cartopy
        return True
    except ImportError:
        return False

@pytest.mark.skipif(not cartopy_installed(),reason='Cartopy is required to run plot method.')
def test_plot():
    """
    Run tests for `SystemGeometry.plot()`
    """
    geo=SystemGeometry()
    plot = geo.plot(0*u.deg)
    assert isinstance(plot,Figure)



if __name__ in '__main__':
    pytest.main(args=[Path(__file__)])
