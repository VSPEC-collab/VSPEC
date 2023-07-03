
import numpy as np
from jax import numpy as jnp
from jax import jit
from jax.scipy.interpolate import RegularGridInterpolator
from astropy import units as u
from tqdm.auto import tqdm

from VSPEC.spectra import read_phoenix
from VSPEC.helpers import isclose
from VSPEC import config


class GridSpectra:
    """
    Store, recall, and interpolate a grid of spectra

    Parameters
    ----------
    wl : astropy.units.Quantity
        The wavelength axis of the spectra.
    spectra : list of numpy.ndarray
        The flux values to place in the grid.
    params : tuple of numpy.ndarray
        The other axes of the grid.

    Examples
    --------
    >>> spectra = [spec1,spec2,spec3]
    >>> params = (3000,3100,3200)
    >>> GridSpectra(wl,spectra,params)

    >>> spectra = [
            [spec11,spec12],
            [spec21,spec22],
            [spec31,spec32]
        ]
    >>> params = [
            (3000,3100,3200), # teff
            (-1,1)          # metalicity
        ]
    >>> GridSpectra(wl,spectra,*params)

    """

    def __init__(self, wl: u.Quantity, spectra: list, *params):
        params = params + (wl.to_value(config.wl_unit),)
        spectra = np.array(spectra)
        assert np.shape(spectra) == tuple([len(param) for param in params])
        self._evaluate = jit(RegularGridInterpolator(params, spectra))

    def evaluate(self, wl: u.Quantity, *args):
        """
        Evaluate the grid. `args` has the same order as `params` in the `__init__` method.

        Parameters
        ----------
        wl : astropy.units.Quantity
            The wavelength coordinates to evaluate at.
        args : list of float
            The points on the other axes to evaluate the grid.

        Returns
        -------
        numpy.ndarray
            The flux of the grid at the evaluated points.

        """
        X = jnp.array([jnp.ones_like(wl)*arg for arg in args] +
                      [wl.to_value(config.wl_unit)]).T
        return self._evaluate(X)

    @classmethod
    def from_vspec(
        cls,
        w1: u.Quantity,
        w2: u.Quantity,
        R: int,
        teffs: u.Quantity
    ):
        """
        Load the default VSPEC PHOENIX grid.

        Parameters
        ----------
        w1 : astropy.units.Quantity
            The blue wavelength limit.
        w2 : astropy.units.Quantity
            The red wavelength limit.
        R : int
            The resolving power to use.
        teffs : astropy.units.Quantity
            The temperature coordinates to load.

        """
        specs = []
        wl = None
        for teff in tqdm(teffs, desc='Loading Spectra', total=len(teffs)):
            wave, flux = read_phoenix(teff, R, w1, w2)
            specs.append(flux.to_value(config.flux_unit))
            if wl is None:
                wl = wave
            else:
                if not np.all(isclose(wl, wave, 1e-6*u.um)):
                    raise ValueError('Wavelength values are different!')
        return cls(wl, np.array(specs), np.array([teff.to_value(config.teff_unit) for teff in teffs]))
