

import astropy.units as u
import numpy as np
import pytest

from VSPEC import helpers


def test_arrange_teff():
    """
    Test `VSPEC.helpers.arrange_teff`
    """
    teff1 = 3010*u.K
    teff2 = 3090*u.K
    assert np.all(helpers.arrange_teff(teff1, teff2) == [3000, 3100]*u.K)

    teff1 = 3000*u.K
    teff2 = 3100*u.K
    assert np.all(helpers.arrange_teff(teff1, teff2) == [3000, 3100]*u.K)

    teff1 = 2750*u.K
    teff2 = 3300*u.K
    assert np.all(helpers.arrange_teff(teff1, teff2) == [
                  27, 28, 29, 30, 31, 32, 33]*(100*u.K))


def test_get_surrounding_teffs():
    """
    Test `VSPEC.helpers.arrange_teff`
    """
    Teff = 3050*u.K
    low, high = helpers.get_surrounding_teffs(Teff)
    assert low == 3000*u.K
    assert high == 3100*u.K

    with pytest.raises(ValueError):
        Teff = 3000*u.K
        helpers.get_surrounding_teffs(Teff)


def test_round_teff():
    """
    Test `VSPEC.helpers.round_teff`
    """
    teff = 100.3*u.K
    assert helpers.round_teff(teff) == 100*u.K


def test_clip_teff():
    """
    Test `VSPEC.helpers.clip_teff`
    """
    low_bound = 2300 * u.K
    high_bound = 3900 * u.K

    teff = 2000 * u.K
    clipped_teff = helpers.clip_teff(teff)
    assert clipped_teff == low_bound

    teff = 6500 * u.K
    clipped_teff = helpers.clip_teff(teff)
    assert clipped_teff == high_bound

    teff = 3000 * u.K
    clipped_teff = helpers.clip_teff(teff)
    assert clipped_teff == teff
