"""
Tests for VSPEC.helpers.astropy_units
"""
from astropy import units as u
import numpy as np
import pytest
from VSPEC import helpers


def test_isclose():
    """
    Test `isclose()`

    Run tests for `VSPEC.helpers.astropy_units.isclose()`.
    This function extends the `numpy.isclose()` function to
    support `astropy.units.Quantity` objects.
    """
    param1 = np.linspace(0, 4, 5) * u.m
    param2 = np.linspace(0, 4, 5) * u.m
    tol = 0 * u.m
    assert np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5) * u.m
    param2 = (np.linspace(0, 4, 5) + 1e-5) * u.m
    tol = 1e-4 * u.m
    assert np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5) * u.m
    param2 = (np.linspace(0, 4, 5) + 1e-5) * u.m
    tol = 1e-6 * u.m
    assert not np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5) * u.m
    param2 = (np.linspace(0, 4, 5) + 1e-5) * u.m
    tol = 1e-5 * u.m  # boundary case
    assert np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5) * u.m
    param2 = (np.linspace(0, 400, 5)) * u.cm
    tol = 1e-6 * u.m
    assert np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5) * u.m
    param2 = (np.linspace(0, 4, 5)) * u.cm
    tol = 1e-6 * u.m
    assert not np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5) * u.m
    param2 = (np.linspace(0, 4, 5)) * u.day
    tol = 1e-6 * u.m
    with pytest.raises(u.UnitConversionError):
        helpers.isclose(param1, param2, tol)

    param1 = np.linspace(0, 4, 5) * u.m
    param2 = (np.linspace(0, 4, 5)) * u.m
    tol = 1e-6 * u.s
    with pytest.raises(u.UnitConversionError):
        helpers.isclose(param1, param2, tol)

    param1 = np.linspace(0, 4, 5) * u.m
    param2 = (np.linspace(0, 4, 5)) * u.m
    tol = 1e-6 * u.cm
    assert np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    # Additional edge cases
    param1 = np.array([1.0, 2.0, 3.0]) * u.m
    param2 = np.array([1.0, 2.0, 3.0]) * u.cm
    tol = 0.01 * u.m
    assert not np.any(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.array([1.0, 2.0, 3.0]) * u.m / u.s
    param2 = np.array([1.0, 2.0, 3.0]) * u.cm / u.ms
    tol = 0.01 * u.m / u.s
    assert not np.any(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.array([1.0, 2.0, 3.0]) * u.m / u.s
    param2 = np.array([1.0, 2.0, 3.0]) * u.cm / u.ms
    tol = 0.01 * u.m
    with pytest.raises(u.UnitConversionError):
        helpers.isclose(param1, param2, tol)
