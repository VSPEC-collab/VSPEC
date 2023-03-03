#!/usr/bin/env python

"""
Tests for `VSPEC.helpers` module
"""

from astropy import units as u
import numpy as np
import pytest

from VSPEC import helpers


def test_to_float():
    """
    Test `to_float()`

    Run tests for `VSPEC.helpers.to_float()`
    This function converts `astropy.Quantity` objects to `float`
    given a target unit
    """
    quant = 1*u.um
    unit = u.AA
    # Angstrom = 1e-10 m
    # micron = 1e-6 m
    # 1e4 angstrom / micron
    expected = 1e4
    observed = helpers.to_float(quant, unit)
    message = f'Units converted unsuccessfully. Expected: {expected}, Observed: {observed}'
    assert observed == pytest.approx(
        expected, rel=1e-9), 'test failed: ' + message

    quant = 1*u.um
    unit = u.Unit('')  # dimensionless
    try:
        helpers.to_float(quant, unit)
        assert False, 'test failed: Bad unit conversion did not raise error'
    except u.UnitConversionError:
        pass


def test_isclose():
    """
    Test `is_close()`

    Run tests for `VSPEC.helpers.isclose()`
    This function extends the `numpy.isclose()` function to
    support `astropy.units.Quantity` objects
    """
    param1 = np.linspace(0, 4, 5)*u.m
    param2 = np.linspace(0, 4, 5)*u.m
    tol = 0*u.m
    assert np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5)*u.m
    param2 = (np.linspace(0, 4, 5) + 1e-5)*u.m
    tol = 1e-4*u.m
    assert np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5)*u.m
    param2 = (np.linspace(0, 4, 5) + 1e-5)*u.m
    tol = 1e-6*u.m
    assert not np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5)*u.m
    param2 = (np.linspace(0, 4, 5) + 1e-5)*u.m
    tol = 1e-5*u.m  # boundary case
    assert np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5)*u.m
    param2 = (np.linspace(0, 400, 5))*u.cm
    tol = 1e-6*u.m
    assert np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5)*u.m
    param2 = (np.linspace(0, 4, 5))*u.cm
    tol = 1e-6*u.m
    assert not np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5)*u.m
    param2 = (np.linspace(0, 4, 5))*u.day
    tol = 1e-6*u.m
    try:
        np.all(helpers.isclose(param1, param2, tol))
        assert False, 'test failed'
    except u.UnitConversionError:
        pass

    param1 = np.linspace(0, 4, 5)*u.m
    param2 = (np.linspace(0, 4, 5))*u.m
    tol = 1e-6*u.s
    try:
        np.all(helpers.isclose(param1, param2, tol))
        assert False, 'test failed'
    except u.UnitConversionError:
        pass

    param1 = np.linspace(0, 4, 5)*u.m
    param2 = (np.linspace(0, 4, 5))*u.m
    tol = 1e-6*u.cm
    try:
        np.all(helpers.isclose(param1, param2, tol))
    except u.UnitConversionError:
        assert False, 'test failed'


def test_get_transit_radius():
    """
    Test `get_transit_radius()`

    Run tests for VSPEC.helpers.get_transit_radius()
    This function calculates the minimum radius from mid-transit
    (i.e. phase=180, i=90 deg) that a planet must be to have no
    overlap between the planetary disk and stellar disk.
    """
    # Use parameters for GJ 1214 b
    # Taken from NExSci Archive on
    # 2022-03-03 by Ted Johnson
    system_distance = 14.6427*u.pc
    stellar_radius = 0.215*u.R_sun
    semimajor_axis = 0.01490*u.AU
    planet_radius = 2.742*u.R_earth
    transit_duration = 0.8788*u.hr
    orbital_period = 1.58040433*u.day
    true_radius = np.pi*u.rad * \
        helpers.to_float(transit_duration/orbital_period,
                         u.dimensionless_unscaled)
    predicted_radius = helpers.get_transit_radius(system_distance,
                                                  stellar_radius,
                                                  semimajor_axis,
                                                  planet_radius)
    true_radius = helpers.to_float(true_radius, u.deg)
    predicted_radius = helpers.to_float(predicted_radius, u.deg)
    message = f'True: {true_radius:.2f}, Calculated: {predicted_radius:.2f}'
    assert predicted_radius == pytest.approx(
        true_radius, rel=0.1), 'test failed: ' + message


if __name__ in '__main__':
    test_to_float()
    test_isclose()
    test_get_transit_radius()
