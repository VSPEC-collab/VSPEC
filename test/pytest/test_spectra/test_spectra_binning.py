import numpy as np
import pytest
from VSPEC.spectra.binning import get_wavelengths, bin_spectra


def test_get_wavelengths():
    """
    Test for `get_wavelengths()` function
    """
    resolving_power = 1000
    lam1 = 400
    lam2 = 800
    wavelengths = get_wavelengths(resolving_power, lam1, lam2)

    assert isinstance(wavelengths, np.ndarray)
    assert len(wavelengths) > 0
    assert wavelengths[0] == lam1
    # the last pixel gets thrown away after binning
    assert wavelengths[-2] <= lam2
    assert np.all(np.diff(np.diff(wavelengths)) > 0)


@pytest.mark.parametrize(
    "resolving_power, lam1, lam2",
    [
        (1000, 400, 800),
        (500, 350, 600),
        (2000, 600, 1000),
    ],
)
def test_get_wavelengths_parametrized(resolving_power, lam1, lam2):
    """
    Parametrized test for `get_wavelengths()` function
    """
    wavelengths = get_wavelengths(resolving_power, lam1, lam2)

    assert isinstance(wavelengths, np.ndarray)
    assert len(wavelengths) > 0
    assert wavelengths[0] == lam1
    assert wavelengths[-2] <= lam2
    assert np.all(np.diff(np.diff(wavelengths)) > 0)


@pytest.mark.parametrize(
    "wl_old, fl_old, wl_new, expected",
    [
        (
            np.array([400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500]),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([405, 425, 435, 455, 465, 485]),
            np.array([1, 1, 1, 1, 1]),
        ),
        # Add more test cases if needed
    ],
)
def test_bin_spectra_parametrized(wl_old, fl_old, wl_new, expected):
    """
    Parametrized test for `bin_spectra()` function
    """
    binned_flux = bin_spectra(wl_old, fl_old, wl_new)

    assert isinstance(binned_flux, np.ndarray)
    assert len(binned_flux) == len(wl_new) - 1
    assert np.all(binned_flux == expected)
