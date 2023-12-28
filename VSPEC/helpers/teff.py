"""
Helpers for Teff related calculations
"""

from astropy import units as u
import numpy as np
import warnings

from VSPEC import config


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


def get_surrounding_teffs(Teff: u.Quantity):
    """
    Get the effective temperatures (Teffs) of the two spectra to interpolate between
    in order to obtain a spectrum with the target Teff.

    This function is useful for determining the Teffs of the two spectra that surround
    a given target Teff value, which are necessary for performing interpolation to
    obtain a spectrum with the desired Teff.

    Parameters
    ----------
    Teff : astropy.units.Quantity
        The target Teff for the interpolated spectrum.

    Returns
    -------
    low_teff : astropy.units.Quantity
        The Teff of the spectrum below the target Teff.
    high_teff : astropy.units.Quantity
        The Teff of the spectrum above the target Teff.

    Raises
    ------
    ValueError
        If the target Teff is a multiple of 100 K, which would cause problems with ``scipy`` interpolation.

    Notes
    -----
    - The function checks if the target Teff is a multiple of 100 K. If it is, a `ValueError` is raised because this would lead to issues with scipy interpolation.
    - If the target Teff is not a multiple of 100 K, the function determines the Teff of the spectrum
        below the target Teff by rounding down to the nearest multiple of 100 K, and the Teff of the
        spectrum above the target Teff is obtained by adding 100 K to the low Teff.

    Examples
    --------
    >>> Teff = 5500 * u.K
    >>> get_surrounding_teffs(Teff)
    (<Quantity 5500. K>, <Quantity 5600. K>)
    """

    step = 100*u.K
    if (Teff % step) == 0*u.K:
        raise ValueError(
            f'Teff of {Teff} is a multiple of {100*u.K}. This will cause problems with scipy.')
    else:
        low_teff = Teff - (Teff % step)
        high_teff = low_teff + step
    return low_teff, high_teff


def round_teff(teff):
    """
    Round the effective temperature to the nearest integer.
    The goal is to reduce the number of unique effective temperatures
    while not affecting the accuracy of the model.

    Parameters
    ----------
    teff : astropy.units.Quantity
        The temperature to round.

    Returns
    -------
    astropy.units.Quantity
        The rounded temperature.

    Notes
    -----
    This function rounds the given effective temperature to the nearest integer value. It is designed to decrease the number of unique effective temperatures while maintaining the accuracy of the model.

    Examples
    --------
    >>> teff = 1234.56 * u.K
    >>> rounded_teff = round_teff(teff)
    >>> print(rounded_teff)
    1235 K

    >>> teff = 2000.4 * u.K
    >>> rounded_teff = round_teff(teff)
    >>> print(rounded_teff)
    2000 K

    """

    val = teff.value
    unit = teff.unit
    return int(round(val)) * unit


def clip_teff(teff: u.Quantity):
    """
    Clip an effective temperature value to ensure it is within the bounds of
    available models.

    Parameters
    ----------
    teff : astropy.units.Quantity
        The effecitve temperature to clip

    Returns
    -------
    astropy.units.Quantity
        The clipped effective temperature.
    """
    low, high = config.grid_teff_bounds
    if teff > high:
        warnings.warn(
            f'Teff of {teff:.1f} too high, clipped to {high:.1f}', RuntimeWarning)
        return high
    elif teff < low:
        warnings.warn(
            f'Teff of {teff:.1f} too low, clipped to {low:.1f}', RuntimeWarning)
        return low
    else:
        return teff
