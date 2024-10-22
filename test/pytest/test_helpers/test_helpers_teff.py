

import astropy.units as u
import numpy as np

from VSPEC import helpers


def test_arrange_teff():
    """
    Test `VSPEC.helpers.arrange_teff`
    """
    teff1 = 3010*u.K
    teff2 = 3090*u.K
    assert np.all(helpers.arrange_teff(teff1, teff2) == np.array([3000, 3100]))

    teff1 = 3000*u.K
    teff2 = 3100*u.K
    assert np.all(helpers.arrange_teff(teff1, teff2) == np.array([3000, 3100]))

    teff1 = 2750*u.K
    teff2 = 3300*u.K
    assert np.all(helpers.arrange_teff(teff1, teff2) == np.array([
                  27, 28, 29, 30, 31, 32, 33])*100)
