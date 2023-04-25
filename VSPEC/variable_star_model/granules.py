"""
Granulation model for VSPEC
"""
import jax
import jax.numpy as jnp
from tinygp import kernels, GaussianProcess
from astropy import units as u
import numpy as np

time_unit = u.hr


class GranulationKernel(kernels.Kernel):
    """
    GP Kernel to describe Granulation
    """

    def __init__(self, scale, period):
        self.scale = jnp.atleast_1d(scale)
        self.freq = 1/jnp.atleast_1d(period)

    def evaluate(self, X1, X2):
        tau = jnp.atleast_1d(jnp.abs(X1 - X2))[..., None]

        return jnp.sum(self.scale*self.freq*jnp.exp(
            -self.freq*tau/jnp.sqrt(2)
        )
            * jnp.cos((self.freq*tau/jnp.sqrt(2))-(jnp.pi/4)))


def build_gp(params, X):
    kernel = GranulationKernel(params['scale'], params['period'])
    return GaussianProcess(
        kernel,
        X,
        diag=0,
        mean=params['mean']
    )


def save_cast_coverage(coverage: np.ndarray):
    """
    Safely cast coverage array so that each value is on [0,1]

    Parameters
    ----------
    coverage : np.ndarray
        The surface coverage.
    """
    coverage[coverage < 0] = 0
    coverage[coverage > 1] = 1
    return coverage


class Granulation:
    """
    Class to contain the granulation model

    Parameters
    ----------
    mean_coverage : float
        The mean surface coverage of the low-Teff regions.
    amplitude : float
        The amplitude of the low-Teff coverage oscillations.
    period : astropy.units.Quantity
        The period of low-Teff coverage oscillations.
    dteff : astropy.units.Quantity
        The temperature difference between the nominal photosphere and the low-Teff region.
    
    Attributes
    ----------
    params : dict
        Parameters for the gaussian process
    dteff : astropy.units.Quantity
        The temperature difference between the nominal photosphere and the low-Teff region.
    """

    def __init__(
            self,
            mean_coverage: float,
            amplitude: float,
            period: u.Quantity,
            dteff: u.Quantity
    ):
        self.params = dict(
            mean=mean_coverage,
            scale=amplitude,
            period=period.to_value(time_unit)
        )
        self.dteff = dteff

    def _build_gp(self, X: np.ndarray) -> GaussianProcess:
        return build_gp(self.params, X)

    def get_coverage(self, time: u.Quantity) -> np.ndarray:
        """
        Get low-Teff region coverage at each point in time.

        Parameters
        ----------
        time : astropy.units.Quantity
            The time axis to generate coverages of.

        Returns
        -------
        np.ndarray
            The coverage corresponding to each point in `time`.    
        """
        key = jax.random.PRNGKey(np.random.randint(
            1, None))  # generate a random key. There must be a better way to do this.
        gp = self._build_gp(time.to_value(time_unit))
        coverage = gp.sample(key)
        return save_cast_coverage(np.array(coverage))
