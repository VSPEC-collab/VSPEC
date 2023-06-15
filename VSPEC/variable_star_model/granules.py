"""
Granulation model for VSPEC
"""
import jax
import jax.numpy as jnp
from tinygp import kernels, GaussianProcess
from astropy import units as u
import numpy as np

from VSPEC.params import GranulationParameters

time_unit = u.hr


class GranulationKernel(kernels.Kernel):
    """
    GP Kernel to describe Granulation

    Parameters
    ----------
    scale : float
        A scalar coefficient for the kernel function.
    period : float
        The period of the kernel in hours.

    Notes
    -----
    This kernel is based on [1]_.

    References
    ----------
    .. [1] Gordon, T. A., Agol, E., & Foreman-Mackey, D. 2020, AJ, 160, 240
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


def build_gp(params: dict, X: np.ndarray) -> GaussianProcess:
    """
    Build Gaussian Process

    Parameters
    ----------
    params : dict
        GP parameters for this instance.
    X : np.ndarray
        The time coordinates for this GP instance.

    """
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
            dteff: u.Quantity,
            seed:int = np.random.randint(1, None)
    ):
        self.params = dict(
            mean=mean_coverage,
            scale=amplitude,
            period=period.to_value(time_unit)
        )
        self.dteff = dteff
        self.seed = seed
    @classmethod
    def from_params(cls,granulation_params:GranulationParameters,seed:int):
        """
        Create an instance of ``Granulation`` from ``GranulationParameters``.

        Parameters
        ----------
        granulation_params : GranulationParameters
            The parameters to construct from.
        """
        return cls(
            mean_coverage=granulation_params.mean,
            amplitude=granulation_params.amp,
            period=granulation_params.period,
            dteff=granulation_params.dteff,
            seed=seed
        )
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
        key = jax.random.PRNGKey(seed=self.seed)
        gp = self._build_gp(time.to_value(time_unit))
        coverage = gp.sample(key)
        return save_cast_coverage(np.array(coverage))
