#!/usr/bin/env python

"""
Tests for `VSPEC.geometry` module
"""

import astropy.units as u
import pytest
import numpy as np
from matplotlib.figure import Figure

from VSPEC.helpers import to_float
from VSPEC.geometry import SystemGeometry


def compare_angles(angle1, angle2, abs=None):
    """
    Use pytest to compare two angles, return `True` if they are equal.
    """
    delta_angle = (np.abs(angle1-angle2)+180*u.deg) % (360*u.deg) - 180*u.deg
    return to_float(delta_angle, u.deg) == pytest.approx(0, abs=to_float(abs, u.deg))


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
    assert geo.sub_obs(time)['lon'] == pytest.approx(init_lon, rel=1e-6)
    assert to_float(geo.sub_obs(time)[
                    'lat'], u.deg) == pytest.approx(-1*to_float(90*u.deg-inclination, u.deg), rel=1e-6)
    time = 0.1*geo.stellar_period
    coords = geo.sub_obs(time)
    assert to_float(coords['lon'], u.deg) == pytest.approx(
        ((init_lon-0.1*360*u.deg) % (360*u.deg)).value, rel=1e-6)
    assert to_float(coords['lat'], u.deg) == pytest.approx(-1 *
                                                           to_float(90*u.deg-inclination, u.deg), rel=1e-6)
    time = 0.5*geo.stellar_period
    coords = geo.sub_obs(time)
    assert to_float(coords['lon'], u.deg) == pytest.approx(
        ((init_lon-0.5*360*u.deg) % (360*u.deg)).value, rel=1e-6)
    assert to_float(coords['lat'], u.deg) == pytest.approx(-1 *
                                                           to_float(90*u.deg-inclination, u.deg), rel=1e-6)
    time = geo.stellar_period
    coords = geo.sub_obs(time)
    assert to_float(coords['lon'], u.deg) == pytest.approx(init_lon, rel=1e-6)
    assert to_float(coords['lat'], u.deg) == pytest.approx(-1 *
                                                           to_float(90*u.deg-inclination, u.deg), rel=1e-6)


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
    assert to_float(coords['lat'], u.deg) == pytest.approx(90, rel=1e-6)
    time = 0.4*geo.stellar_period
    coords = geo.sub_obs(time)
    assert to_float(coords['lat'], u.deg) == pytest.approx(90, rel=1e-6)

    geo = SystemGeometry(stellar_offset_amp=45*u.deg,
                         stellar_offset_phase=90*u.deg, inclination=90*u.deg)
    time = 0*u.s
    coords = geo.sub_obs(time)
    assert to_float(coords['lat'], u.deg) == pytest.approx(0, rel=1e-6)
    time = 0.4*geo.stellar_period
    coords = geo.sub_obs(time)
    assert to_float(coords['lat'], u.deg) == pytest.approx(0, rel=1e-6)

    geo = SystemGeometry(stellar_offset_amp=90*u.deg,
                         stellar_offset_phase=180*u.deg, inclination=90*u.deg)
    time = 0*u.s
    coords = geo.sub_obs(time)
    assert to_float(coords['lat'], u.deg) == pytest.approx(-90, rel=1e-6)
    time = 0.4*geo.stellar_period
    coords = geo.sub_obs(time)
    assert to_float(coords['lat'], u.deg) == pytest.approx(-90, rel=1e-6)


def test_mean_motion():
    """
    Run tests for `SystemGeometry.mean_motion()`
    """
    geo = SystemGeometry()
    assert to_float(geo.mean_motion(), u.deg/u.day) == pytest.approx(
        to_float(360*u.deg/geo.orbital_period, u.deg/u.day), rel=1e-6)


def test_mean_anomaly():
    """
    Run tests for `SystemGeometry.mean_anomaly()`
    """
    geo = SystemGeometry(init_planet_phase=0*u.deg,
                         phase_of_periasteron=0*u.deg)
    for time in np.linspace(0, 2, 11):
        assert to_float(geo.mean_anomaly(time*geo.orbital_period),
                        u.deg) == pytest.approx(360 * (time % 1), rel=1e-6)


def test_eccentric_anomaly():
    """
    Run tests for `SystemGeometry.eccentric_anomaly()`
    """
    geo = SystemGeometry(init_planet_phase=0*u.deg,
                         phase_of_periasteron=0*u.deg, eccentricity=0)
    for time in np.linspace(0, 2, 11):
        assert to_float(geo.eccentric_anomaly(time*geo.orbital_period), u.deg) == pytest.approx(
            to_float(geo.mean_anomaly(time*geo.orbital_period), u.deg), rel=1e-6)
    eccentricity = 0.5
    geo = SystemGeometry(init_planet_phase=0*u.deg,
                         phase_of_periasteron=0*u.deg, eccentricity=eccentricity)
    for time in np.linspace(0, 2, 11):
        mean_anom = geo.mean_anomaly(time*geo.orbital_period)
        eccentric_anom = geo.eccentric_anomaly(time*geo.orbital_period)
        lhs = to_float(eccentric_anom, u.rad) - eccentricity * \
            to_float(np.sin(eccentric_anom), u.dimensionless_unscaled)
        rhs = to_float(mean_anom, u.rad)
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
        assert to_float(true_anomaly, u.deg) == pytest.approx(
            to_float(mean_anomaly, u.deg), rel=1e-6)
    eccentricity = 0.5
    geo = SystemGeometry(init_planet_phase=0*u.deg,
                         phase_of_periasteron=0*u.deg, eccentricity=eccentricity)
    for time in np.linspace(0, 2, 11):
        true_anomaly = geo.true_anomaly(time*geo.orbital_period)
        eccentric_anomaly = geo.eccentric_anomaly(time*geo.orbital_period)
        lhs = to_float(np.tan(eccentric_anomaly), u.dimensionless_unscaled)
        numerator = np.sqrt(1-eccentricity**2) * np.sin(true_anomaly)
        denominator = eccentricity + np.cos(true_anomaly)
        rhs = to_float(numerator/denominator, u.dimensionless_unscaled)
        assert lhs == pytest.approx(rhs, rel=1e-6)


def test_phase():
    """
    Run tests for `SystemGeometry.true_anomaly()`
    """
    geo = SystemGeometry(init_planet_phase=0*u.deg,
                         phase_of_periasteron=0*u.deg, eccentricity=0)
    for time in np.linspace(0, 2, 11):
        phase = geo.phase(time*geo.orbital_period)
        assert to_float(phase, u.deg) == pytest.approx(360*(time % 1), 1e-6)

    geo = SystemGeometry(init_planet_phase=0*u.deg,
                         phase_of_periasteron=0*u.deg, eccentricity=0)
    for time in np.linspace(0, 2, 11):
        phase = geo.phase(time*geo.orbital_period)
        true_anomaly = geo.true_anomaly(time*geo.orbital_period)
        assert to_float(phase, u.deg) == pytest.approx(
            to_float(true_anomaly, u.deg))
    for init_planet_phase in np.linspace(0, 360, 4)*u.deg:
        for phase_of_periasteron in np.linspace(0, 360, 7)*u.deg:
            geo = SystemGeometry(init_planet_phase=init_planet_phase,
                                 phase_of_periasteron=phase_of_periasteron, eccentricity=0)
            for time in np.linspace(0, 2, 11):
                phase = geo.phase(time*geo.orbital_period)
                true_anomaly = geo.true_anomaly(time*geo.orbital_period)
                assert to_float(phase, u.deg) == pytest.approx(
                    to_float(true_anomaly+phase_of_periasteron, u.deg) % 360, rel=1e-6)


def test_sub_planet():
    """
    Run tests for `SystemGeometry.sub_planet()`
    """
    geo = SystemGeometry()
    time = 3*u.day
    coords1 = geo.sub_planet(time)
    coords2 = geo.sub_planet(time, phase=geo.phase(time))
    assert to_float(coords1['lat'], u.deg) == pytest.approx(
        to_float(coords2['lat'], u.deg), rel=1e-6)
    assert to_float(coords1['lon'], u.deg) == pytest.approx(
        to_float(coords2['lon'], u.deg), rel=1e-6)

    geo = SystemGeometry(phase_of_periasteron=180*u.deg,
                         init_planet_phase=180*u.deg,
                         init_stellar_lon=0*u.deg, inclination=90*u.deg)
    sub_obs = geo.sub_obs(0*u.s)
    sub_pl = geo.sub_planet(0*u.s)
    assert to_float(sub_pl['lon'], u.deg) == pytest.approx(
        to_float(sub_obs['lon'], u.deg), abs=1e-6)
    assert to_float(sub_pl['lat'], u.deg) == pytest.approx(
        to_float(sub_obs['lat'], u.deg), abs=1e-6)

    geo = SystemGeometry(phase_of_periasteron=180*u.deg,
                         init_planet_phase=180*u.deg,
                         init_stellar_lon=0*u.deg, inclination=90*u.deg,
                         stellar_offset_amp=45*u.deg, stellar_offset_phase=0*u.deg)
    sub_obs = geo.sub_obs(0*u.s)
    sub_pl = geo.sub_planet(0*u.s)
    assert to_float(sub_pl['lon'], u.deg) == pytest.approx(
        to_float(sub_obs['lon'], u.deg), abs=1e-6)
    assert to_float(sub_pl['lat'], u.deg) == pytest.approx(
        to_float(sub_obs['lat'], u.deg), abs=1e-6)

    geo = SystemGeometry(phase_of_periasteron=180*u.deg,
                         init_planet_phase=180*u.deg,
                         init_stellar_lon=0*u.deg, inclination=90*u.deg,
                         stellar_offset_amp=45*u.deg, stellar_offset_phase=90*u.deg)
    sub_obs = geo.sub_obs(0*u.s)
    sub_pl = geo.sub_planet(0*u.s)
    assert to_float(sub_pl['lon'], u.deg) == pytest.approx(
        to_float(sub_obs['lon'], u.deg), abs=1e-6)
    assert to_float(sub_pl['lat'], u.deg) == pytest.approx(
        to_float(sub_obs['lat'], u.deg), abs=1e-6)


def test_get_time_since_periasteron():
    """
    Run tests for `SystemGeometry.get_time_since_periasteron()`
    """
    geo = SystemGeometry(phase_of_periasteron=0*u.deg,
                         init_planet_phase=0*u.deg)
    assert to_float(geo.get_time_since_periasteron(
        0*u.deg), u.s) == pytest.approx(0, abs=10)
    assert to_float(geo.get_time_since_periasteron(
        180*u.deg), u.s) == pytest.approx(to_float(0.5*geo.orbital_period, u.s), abs=10)

    geo = SystemGeometry(phase_of_periasteron=90*u.deg,
                         init_planet_phase=90*u.deg)
    assert to_float(geo.get_time_since_periasteron(
        0*u.deg), u.s) == pytest.approx(to_float(0.75*geo.orbital_period, u.s), abs=10)
    assert to_float(geo.get_time_since_periasteron(
        180*u.deg), u.s) == pytest.approx(to_float(0.25*geo.orbital_period, u.s), abs=10)


def test_get_substellar_lon_at_periasteron():
    """
    Run tests for `SystemGeometry.get_time_since_periasteron()`
    """
    geo = SystemGeometry(phase_of_periasteron=0*u.deg,
                         init_planet_phase=0*u.deg,
                         planetary_init_substellar_lon=0*u.deg,
                         orbital_period=10*u.day,
                         planetary_rot_period=10*u.day)
    assert to_float(geo.get_substellar_lon_at_periasteron(),
                    u.deg) == pytest.approx(0, abs=1e-6)

    geo = SystemGeometry(phase_of_periasteron=90*u.deg,
                         init_planet_phase=270*u.deg,
                         planetary_init_substellar_lon=0*u.deg,
                         orbital_period=10*u.day,
                         planetary_rot_period=10*u.day)
    assert to_float(geo.get_substellar_lon_at_periasteron(),
                    u.deg) == pytest.approx(0, abs=1e-6)

    geo = SystemGeometry(phase_of_periasteron=90*u.deg,
                         init_planet_phase=270*u.deg,
                         planetary_init_substellar_lon=0*u.deg,
                         orbital_period=10*u.day,
                         planetary_rot_period=5*u.day)
    assert to_float(geo.get_substellar_lon_at_periasteron(),
                    u.deg) == pytest.approx(180, abs=1e-6)

    geo = SystemGeometry(phase_of_periasteron=0*u.deg,
                         init_planet_phase=50*u.deg,
                         planetary_init_substellar_lon=90*u.deg,
                         orbital_period=10*u.day,
                         planetary_rot_period=10*u.day)
    assert to_float(geo.get_substellar_lon_at_periasteron(),
                    u.deg) == pytest.approx(90, abs=1e-6)


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
    plan = geo.get_observation_plan(0*u.deg,10*u.day,N_obs=10)
    for value in plan.values():
        assert len(value)==10
    plan = geo.get_observation_plan(0*u.deg,10*u.day,time_step=0.5*u.day,N_obs=10)
    for value in plan.values():
        assert len(value)==20

def test_plot():
    """
    Run tests for `SystemGeometry.plot()`
    """
    geo=SystemGeometry()
    plot = geo.plot(0*u.deg)
    assert isinstance(plot,Figure)



if __name__ in '__main__':
    test_compare_angles()
    test_default_init()
    test_custon_init()
    test_sub_obs()
    test_mean_motion()
    test_mean_anomaly()
    test_eccentric_anomaly()
    test_true_anomaly()
    test_phase()
    test_sub_planet()
    test_get_time_since_periasteron()
    test_get_substellar_lon_at_periasteron()
    test_get_substellar_lon()
    test_get_substellar_lat()
    test_get_pl_sub_obs_lon()
    test_get_pl_sub_obs_lat()
    test_get_radius_coefficient()
    test_get_observation_plan()
    test_plot()
