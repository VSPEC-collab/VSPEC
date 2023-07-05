import numpy as np
from astropy import units as u

from VSPEC.spectra.grid import GridSpectra
from VSPEC.helpers import isclose


def test_grid_spectra_1():
    wl = np.linspace(3000, 3200, 3) * u.Angstrom
    spectra = [np.random.rand(3), np.random.rand(3), np.random.rand(3)]
    params = (np.array([3000, 3100, 3200]),)
    grid = GridSpectra(wl, spectra, *params)

    # Evaluate the grid at specific parameter values
    evaluated = grid.evaluate(wl, 3100)

    # Assertions
    assert evaluated.shape == (3,)
    assert np.all(np.isclose(evaluated, spectra[1], atol=1e-6))


def test_grid_spectra_from_vspec():
    w1 = 3 * u.um
    w2 = 4 * u.um
    R = 500
    teffs = np.linspace(3000, 3200, 3) * u.K
    grid = GridSpectra.from_vspec(w1, w2, R, teffs)

    # Evaluate the grid at specific parameter values
    wl = np.linspace(3000, 3200, 3) * u.um
    evaluated = grid.evaluate(wl, 3100)

    # Assertions
    assert evaluated.shape == (3,)
    # assert evaluated.unit == grid._evaluate([3100])[0].unit
