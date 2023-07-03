
import numpy as np
from astropy import units as u, constants as c
from VSPEC import config


class ForwardSpectra:
    """
    Spectra from a forward model.

    Parameters
    ----------
    func : callable
        The function that performs the evaluation of the spectra.

    """

    def __init__(self, func):
        """
        pass a function that does the real work
        """
        self.evaluate = func

    @classmethod
    def blackbody(cls):
        """
        Create a ForwardSpectra object for blackbody spectra.

        Returns
        -------
        ForwardSpectra
            A ForwardSpectra object for blackbody spectra.

        """

        def func(wl: u.Quantity, teff: u.Quantity):
            """
            Evaluate the blackbody spectra at the given wavelength and temperature.

            Parameters
            ----------
            wl : jax.numpy.ndarray
                The array of wavelength values in micrometers.
            teff : float
                The temperature in Kelvin.

            Returns
            -------
            jax.numpy.ndarray
                The flux values of the blackbody spectra at the given wavelength and temperature.

            """
            if not wl.unit.physical_type == u.um.physical_type:
                raise u.UnitTypeError('wl is wrong physical type')
            if not teff.unit.physical_type == u.K.physical_type:
                raise u.UnitTypeError('teff is wrong physical type')
            A = 2 * c.h * c.c**2/wl**5
            B = np.exp(((c.h*c.c)/(wl*c.k_B*teff)
                        ).to(u.dimensionless_unscaled)) - 1
            return (A/B).to(config.flux_unit)
        return cls(func)
