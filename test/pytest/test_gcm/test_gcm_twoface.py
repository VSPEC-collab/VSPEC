"""
Tests for the two-face planet model.
"""
import pytest
import numpy as np
from numpy import pi
from astropy import units as u
import libpypsg as psg
from VSPEC.gcm import twoface


def test_clean_interp():
    """
    Test function for inperpolation data cleaner
    """
    phis = np.linspace(0, pi, 10)
    thetas = np.linspace(0, pi, 20)
    p0d = 1 * u.bar
    q0d = 100 * u.K
    pressure_unit, quant_unit, angle_unit, phi_arr, theta_arr = twoface._clean_interp(
        phis, thetas, p0d, q0d)
    assert pressure_unit == u.bar
    assert quant_unit == u.K
    assert angle_unit == u.rad
    assert phi_arr.shape == (10, 20)
    assert theta_arr.shape == (10, 20)

    phis = np.linspace(0, 360, 100) * u.deg
    thetas = np.linspace(0, 180, 200) * u.deg
    p0d = 1 * u.bar
    q0d = 100 * u.K
    pressure_unit, quant_unit, angle_unit, phi_arr, theta_arr = twoface._clean_interp(
        phis, thetas, p0d, q0d)
    assert pressure_unit == u.bar
    assert quant_unit == u.K
    assert angle_unit == u.deg
    assert phi_arr.shape == (100, 200)
    assert theta_arr.shape == (100, 200)


def test_gen_planet():
    dayside_pressure = 1 * u.bar
    nightside_pressure = 2 * u.bar
    elbow_pressure = 0.01 * u.bar
    dayside_temperature = 800 * u.K
    nightside_temperature = 500 * u.K
    elbow_temperature = 400 * u.K
    top_pressure = 1e-5 * u.bar
    n_linear = 30
    n_const = 20
    nphi = 90
    ntheta = 45
    scheme = 'cos2'
    dayside_h2o = 1e-2 * u.dimensionless_unscaled
    nightside_h2o = 1e-1 * u.dimensionless_unscaled
    elbow_h2o = 1e-3 * u.dimensionless_unscaled
    co2 = 1e-3 * u.dimensionless_unscaled
    o3 = 1e-3 * u.dimensionless_unscaled
    no2 = 1e-3 * u.dimensionless_unscaled
    albedo = 0.1
    planet: psg.globes.PyGCM = twoface.gen_planet(
        dayside_pressure, elbow_pressure,
        dayside_temperature, elbow_temperature,
        nightside_pressure, elbow_pressure,
        nightside_temperature, elbow_temperature,
        n_linear, n_const, top_pressure, nphi, ntheta,
        scheme,
        dayside_h2o, elbow_h2o,
        nightside_h2o, elbow_h2o,
        co2, o3, no2, albedo
    )

    assert planet.pressure.dat.shape == (n_linear+n_const, nphi, ntheta), \
        f'Expected pressure shape to be {(n_linear+n_const, nphi, ntheta)}, \
        got {planet.pressure.dat.shape}'
    assert planet.temperature.dat.shape == (n_linear+n_const, nphi, ntheta), \
        f'Expected temperature shape to be {(n_linear+n_const, nphi, ntheta)}, \
        got {planet.temperature.dat.shape}'
    assert np.all(np.isclose(planet.pressure.dat[-1, :, :], top_pressure, rtol=1e-15)), \
        f'Expected top pressure to be {top_pressure}, \
            got values between {np.min(planet.pressure.dat[-1,:,:])} \
            and {np.max(planet.pressure.dat[-1,:,:])}'
    assert np.isclose(np.min(planet.pressure.dat), top_pressure, rtol=1e-15), \
        f'Expected minimum pressure to be {top_pressure}, got {np.min(planet.pressure.dat)}'
    assert np.max(planet.pressure.dat) == max(dayside_pressure, nightside_pressure), \
        f'Expected maximum pressure to be {max(dayside_pressure, nightside_pressure)}, \
            got {np.max(planet.pressure.dat)}'
    assert np.all(np.isclose(planet.temperature.dat[n_linear:, :, :], elbow_temperature, rtol=1e-15)), \
        f'Expected elbow temperature to be {elbow_temperature}, \
            got values between {np.min(planet.temperature.dat[n_linear:, :, :])} \
            and {np.max(planet.temperature.dat[n_linear:, :, :])}'
    assert np.all(
        planet.temperature.dat[0, :, :] >= min(dayside_temperature, nightside_temperature)), \
        f'Expected surface pressures greater than {min(dayside_temperature, nightside_temperature)}, \
        got values between {np.min(planet.temperature.dat[0,:,:])} and {np.max(planet.temperature.dat[0,:,:])}'
    assert np.all(planet.temperature.dat[0, :, :] <= max(dayside_temperature, nightside_temperature)), \
        f'Expected surface pressures less than {max(dayside_temperature, nightside_temperature)}, \
            got values between {np.min(planet.temperature.dat[0,:,:])} and {np.max(planet.temperature.dat[0,:,:])}'



if __name__ in '__main__':
    pytest.main(args=[__file__])
