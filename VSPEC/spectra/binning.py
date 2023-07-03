"""
Spectra binning functions
"""

import numpy as np


def get_wavelengths(resolving_power: int, lam1: float, lam2: float) -> np.ndarray:
    """
    Get wavelengths

    Get wavelength points given a resolving power and a desired spectral range.
    Provides one more point than PSG, which is alows us to set a red bound on the last pixel.

    Parameters
    ----------
    resolving_power : int
        Resolving power.
    lam1 : float
        Initial wavelength.
    lam2 : float
        Final wavelength.

    Returns
    -------
    numpy.ndarray
        Wavelength points.
    """
    lam = lam1
    lams = [lam]
    while lam < lam2:
        dlam = lam / resolving_power
        lam = lam + dlam
        lams.append(lam)
    lams = np.array(lams)
    return lams


def bin_spectra(wl_old: np.array, fl_old: np.array, wl_new: np.array):
    """
    Bin spectra

    This is a generic binning funciton.

    Parameters
    ----------
    wl_old : np.ndarray
        The original wavelength values.
    fl_old : np.ndarray
        The original flux values.
    wl_new : np.ndarray
        The new wavelength values.

    Returns
    -------
    fl_new : np.ndarray
        The new flux values.
    """
    binned_flux = []
    for i in range(len(wl_new) - 1):
        lam_cen = wl_new[i]
        upper = 0.5*(lam_cen + wl_new[i+1])
        if i == 0:
            # dl = upper - lam_cen # uncomment to sample blue of first pixel
            lower = lam_cen  # - dl
        else:
            lower = 0.5*(lam_cen + wl_new[i-1])
        if lower >= upper:
            raise ValueError('Somehow lower is greater than upper!')
        reg = (wl_old >= lower) & (wl_old < upper)
        if not np.any(reg):
            raise ValueError(
                f'Some pixels must be selected!\nlower={lower}, upper={upper}')
        binned_flux.append(fl_old[reg].mean())
    binned_flux = np.array(binned_flux)
    return binned_flux
