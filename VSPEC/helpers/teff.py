"""
Helpers for Teff related calculations
"""

from astropy import units as u
import numpy as np

def arrange_teff(minteff: u.Quantity, maxteff: u.Quantity):
    """
    Generate a list of effective temperature (Teff) values with steps of 100 K that fully
    encompass the specified range.

    This function is useful for obtaining a list of Teff values to be used for binning spectra later on.

    Parameters
    ----------
    minteff : astropy.units.Quantity
        The minimum Teff value required.
    maxteff : astropy.units.Quantity
        The maximum Teff value required.

    Returns
    -------
    teffs : list of int
        An array of Teff values, with steps of 100 K.

    Notes
    -----
    - The function calculates the Teff values that fully encompass the specified range by rounding down the minimum value to the nearest multiple of 100 K and rounding up the maximum value to the nearest multiple of 100 K.
    - The `np.arange` function is then used to generate a sequence of Teff values with steps of 100 K, covering the entire range from the rounded-down minimum value to the rounded-up maximum value.

    Examples
    --------
    >>> minteff = 5000 * u.K
    >>> maxteff = 6000 * u.K
    >>> arrange_teff(minteff, maxteff)
    [5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000]
    """

    step = 100*u.K
    if (minteff % step) == 0*u.K:
        low = minteff
    else:
        low = minteff - (minteff % step)
    if (maxteff % step) == 0*u.K:
        high = maxteff
    else:
        high = maxteff - (maxteff % step) + step
    number_of_steps = ((high-low)/step).to_value(u.dimensionless_unscaled)
    number_of_steps = int(round(number_of_steps))
    teffs = low + np.arange(number_of_steps+1)*step
    return np.array([int(round(teff.to_value(u.K))) for teff in teffs],dtype=int)