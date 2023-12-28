"""
Misc helpers
"""

from astropy import units as u
import numpy as np


def get_planet_indicies(
    planet_times: u.Quantity,
    tindex: u.Quantity
    ) -> tuple[int, int]:
    """
    Get the indices of the planet spectra to interpolate over.

    This helper function enables interpolation of planet spectra by determining
    the appropriate indices in the `planet_times` array. By running PSG once for
    multiple "integrations" and interpolating between the spectra, computational
    efficiency is improved.


    Parameters
    ----------
    planet_times : astropy.units.Quantity
        The times (cast to since periasteron) at which the planet spectrum was taken.
    tindex : astropy.units.Quantity
        The epoch of the current observation. The goal is to place this between
        two elements of `planet_times`

    Returns
    -------
    int
        The index of `planet_times` before `tindex`
    int
        The index of `planet_times` after `tindex`

    Raises
    ------
    ValueError
        If multiple elements of 'planet_times' are equal to 'tindex'.
    """
    after = planet_times > tindex
    equal = planet_times == tindex
    if equal.sum() == 1:
        N1 = np.argwhere(equal)[0][0]
        N2 = np.argwhere(equal)[0][0]
    elif equal.sum() > 1:
        raise ValueError('There must be a duplicate time')
    elif equal.sum() == 0:
        N2 = np.argwhere(after)[0][0]
        N1 = N2 - 1
    return N1, N2