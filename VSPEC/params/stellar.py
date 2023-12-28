"""
Stellar Parameters
"""
from typing import Union, Tuple
from astropy import units as u
import numpy as np
import yaml

from vspec_vsm import (
    SpotGenerator,
    FaculaGenerator,
    FlareGenerator,
    Granulation,
    Star,
    SpotCollection,
    FaculaCollection,
    CoordinateGrid
)
from vspec_vsm import config as vsm_config


from VSPEC.config import MSH, PRESET_PATH
from VSPEC.params.base import BaseParameters


class LimbDarkeningParameters(BaseParameters):
    """
    Limb Darkening Parameters for the Quadratic
    Limb Darkening Law

    Parameters
    ----------
    u1 : float
        The limb darkening parameter u1
    u2 : float
        The limb darkening parameter u2

    Attributes
    ----------
    u1 : float
        The limb darkening parameter u1
    u2 : float
        The limb darkening parameter u2

    Notes
    -----
    Quadratic Law (Kopal, 1950, cited in :cite:t:`2022AJ....163..228P`)

    .. math::

        \\frac{I(\\mu)}{I(1)} = 1 - u_1 (1-\\mu) - u_2 (1-\\mu)^2
    
    Examples
    --------
    >>> params_dict = {'preset': 'solar'}
    >>> params = LimbDarkeningParameters.from_dict(params_dict)

    In the example above, the 'solar' preset configuration is used to create an instance
    of LimbDarkeningParameters.

    >>> params_dict = {'u1': 0.3, 'u2': 0.1}
    >>> params = LimbDarkeningParameters.from_dict(params_dict)

    In the example above, custom values for 'u1' and 'u2' are provided to create an instance
    of LimbDarkeningParameters.

    """

    def __init__(self, u1: float, u2: float):
        self.u1 = u1
        self.u2 = u2

    @classmethod
    def _from_dict(cls, d):
        return cls(
            float(d['u1']),
            float(d['u2'])
        )

    @classmethod
    def solar(cls):
        """
        From :cite:t:`2013A&A...552A..16C`
        S4 bandpass
        """
        return cls(0.0473, 0.0841)

    @classmethod
    def proxima(cls):
        """
        From :cite:t:`2012A&A...546A..14C`
        S4 bandpass
        """
        return cls(0.0551, 0.1075)

    @classmethod
    def trappist(cls):
        """
        From :cite:t:`2012A&A...546A..14C`
        S4 bandpass
        """
        return cls(0.0778, 0.1619)

    @classmethod
    def lambertian(cls):
        """
        No limb darkening.
        """
        return cls(0., 0.)


class SpotParameters(BaseParameters):
    """
    Parameters controling variability from star spots.

    Parameters
    ----------
    distribution : str
        The distribution function to be used for the spot positions.
        Either ``iso`` for an isotropic distribution or ``solar`` for two bands at :math:`\\pm 15^\\circ` latitude.
    initial_coverage : float
        The spot coverage created initially by generating spots at random stages of life.
    area_mean : astropy.units.Quantity
        The mean area of a spot on the star's surface is MSH.
    area_logsigma : float
        The standard deviation of the spot areas. This is a lognormal
        distribution, so the units of this value are dex.
    teff_umbra : astropy.units.Quantity
        The effective temperature of the spot umbrae.
    teff_penumbra : astropy.units.Quantity
        The effective temperature of the spot penumbrae.
    equillibrium_coverage : float
        The fractional coverage of the star's surface by spots. This is the value
        at growth-decay equillibrium.
    burn_in : astropy.units.Quantity
        The duration of the burn-in period, during which the spot coverage approaches
        equillibrium.
    growth_rate : astropy.units.Quantity
        The rate at which new spots grow.
    decay_rate : astropy.units.Quantity
        The rate at which existing spots decay.
    initial_area : astropy.units.Quantity
        The initial area of newly created spots.

    Attributes
    ----------
    distribution : str
        The distribution function to be used for the spot positions. 'iso' or 'solar'.
    initial_coverage : float
        The coverage for a 'hot start'.
    area_logmean : float
        The mean area of a spot on the star's surface is MSH.
    area_logsigma : float
        The standard deviation of the spot areas. This is a lognormal
        distribution, so the units of this value are dex.
    teff_umbra : astropy.units.Quantity
        The effective temperature of the spot umbrae.
    teff_penumbra : astropy.units.Quantity
        The effective temperature of the spot penumbrae.
    equillibrium_coverage : float
        The fractional coverage of the star's surface by spots. This is the value
        at growth-decay equillibrium, and different from the 'hot start' value given
        by `initial_coverage`.
    burn_in : astropy.units.Quantity
        The duration of the burn-in period, during which the spot coverage approaches
        equillibrium.
    growth_rate : astropy.units.Quantity
        The rate at which new spots grow.
    decay_rate : astropy.units.Quantity
        The rate at which existing spots decay.
    initial_area : astropy.units.Quantity
        The initial area of newly created spots.
    
    """
    _PRESET_PATH = PRESET_PATH / 'spots.yaml'
    """
    The path to the preset file.
    """
    def __init__(
        self,
        distribution: str,
        initial_coverage: float,
        area_mean: u.Quantity,
        area_logsigma: float,
        teff_umbra: u.Quantity,
        teff_penumbra: u.Quantity,
        equillibrium_coverage: float,
        burn_in: u.Quantity,
        growth_rate: u.Quantity,
        decay_rate: u.Quantity,
        initial_area: u.Quantity
    ):
        self.distribution = distribution
        self.initial_coverage = initial_coverage
        self.area_mean = area_mean
        self.area_logsigma = area_logsigma
        self.teff_umbra = teff_umbra
        self.teff_penumbra = teff_penumbra
        self.equillibrium_coverage = equillibrium_coverage
        self.burn_in = burn_in
        self.growth_rate = growth_rate
        self.decay_rate = decay_rate
        self.initial_area = initial_area
        self._validate()

    def _validate(self):
        """
        Validate class instance
        """
        if self.distribution not in ['solar', 'iso']:
            raise ValueError('`distribution` must either be `solar` or `iso`')
        if self.initial_coverage > 1 or self.initial_coverage < 0:
            raise ValueError('`initial_coverage` must be between 0 and 1.')
        if self.equillibrium_coverage > 1 or self.equillibrium_coverage < 0:
            raise ValueError(
                '`equillibrium_coverage` must be between 0 and 1.')

    @classmethod
    def _from_dict(cls, d):
        u.add_enabled_units([MSH])
        return cls(
            distribution=str(d['distribution']),
            initial_coverage=float(d['initial_coverage']),
            area_mean=u.Quantity(d['area_mean']),
            area_logsigma=float(d['area_logsigma']),
            teff_umbra=u.Quantity(d['teff_umbra']),
            teff_penumbra=u.Quantity(d['teff_penumbra']),
            equillibrium_coverage=float(d['equillibrium_coverage']),
            burn_in=u.Quantity(d['burn_in']),
            growth_rate=u.Quantity(d['growth_rate']),
            decay_rate=u.Quantity(d['decay_rate']),
            initial_area=u.Quantity(d['initial_area'])
        )
    @classmethod
    def from_preset(cls,name):
        """
        Load a ``SpotParameters`` instance from a preset file.

        Parameters
        ----------
        name : str
            The name of the preset to load.
        
        Returns
        -------
        SpotParameters
            The class instance loaded from a preset.
        """
        with open(cls._PRESET_PATH, 'r',encoding='UTF-8') as file:
            data = yaml.safe_load(file)
            return cls.from_dict(data[name])

    @classmethod
    def none(cls):
        """
        No spots
        """
        return cls.from_preset('none')

    @classmethod
    def mdwarf(cls):
        """
        Static Spots
        """
        return cls.from_preset('mdwarf')

    @classmethod
    def solar(cls):
        """
        Solar-style spots
        """
        return cls.from_preset('solar')
    def to_generator(
        self,
        grid_params: Union[int,Tuple[int, int]] = (vsm_config.NLAT, vsm_config.NLON),
        gridmaker: CoordinateGrid = None,
        rng: np.random.Generator = np.random.default_rng()
    )-> SpotGenerator:
        """
        Create a `vspec_vsm.SpotGenerator` instance from the class instance.
        
        Parameters
        ----------
        grid_params : Union[int, Tuple[int, int]], optional
            If tuple, the number of grid points in the latitude and longitude directions.
            If int, the number of total grid points for a sprial grid.
            Defaults to (vsm_config.NLAT, vsm_config.NLON).
        gridmaker : CoordinateGrid, optional
            An instance of `vspec_vsm.CoordinateGrid` to use for the grid.
            Defaults to None.
        rng : np.random.Generator, optional
            The random number generator to use. Defaults to np.random.default_rng().
        
        Returns
        -------
        vspec_vsm.SpotGenerator
            The `vspec_vsm.SpotGenerator` instance.
        """
        return SpotGenerator(
            dist_area_mean=self.area_mean,
            dist_area_logsigma=self.area_logsigma,
            umbra_teff=self.teff_umbra,
            penumbra_teff=self.teff_penumbra,
            growth_rate=self.growth_rate,
            decay_rate=self.decay_rate,
            init_area=vsm_config.starspot_initial_area,
            distribution=self.distribution,
            coverage=self.equillibrium_coverage,
            grid_params=grid_params,
            gridmaker=gridmaker,
            rng=rng
        )


class FaculaParameters(BaseParameters):
    """
    Facula Parameters

    Parameters
    ----------
    distribution : str
        The distribution used to generate the faculae on the star.
        Currently only 'iso' is supported.
    equillibrium_coverage : float
        The fraction of the star's surface covered by the faculae at growth-decay equilibrium.
    burn_in : astropy.units.Quantity
        The duration of the burn-in period, during which the facula coverage approaches
        equillibrium.
    mean_radius : astropy.units.Quantity
        The mean radius of the faculae.
    logsigma_radius : float
        The standard deviation of :math:`\\log{r_{0}}.
    depth : astropy.units.Quantity
        The depth of the facula depression.
    mean_timescale : astropy.units.Quantity
        The mean faculae lifetime.
    logsigma_timescale : float
        The standard deviation of :math:`\\log{\\tau}`
    floor_teff_slope : astropy.units.Quantity
        The slope of the radius-Teff relationship
    floor_teff_min_rad : astropy.units.Quantity
        The minimum radius at which the floor is visible. Otherwise the facula
        is a bright point -- even near disk center.
    floor_teff_base_dteff : astropy.units.Quantity
        The Teff of the floor at the minimum radius.
    wall_teff_slope : astropy.units.Quantity
        The slope of the radius-Teff relationship
    wall_teff_intercept : astropy.units.Quantity
        The Teff of the wall when :math:`R = 0`.

    Attributes
    ----------
    distribution : str
        The distribution used to generate the faculae on the star.
    equillibrium_coverage : float
        The fraction of the star's surface covered by the faculae at growth-decay equilibrium.
    burn_in : astropy.units.Quantity
        The duration of the burn-in period, during which the facula coverage approaches
        equillibrium.
    mean_radius : astropy.units.Quantity
        The mean radius of the faculae.
    logsigma_radius : float
        The standard deviation of :math:`\\log{r_{0}}.
    depth : astropy.units.Quantity
        The depth of the facula depression.
    mean_timescale : astropy.units.Quantity
        The mean faculae lifetime.
    logsigma_timescale : float
        The standard deviation of :math:`\\log{\\tau}`
    floor_teff_slope : astropy.units.Quantity
        The slope of the radius-Teff relationship
    floor_teff_min_rad : astropy.units.Quantity
        The minimum radius at which the floor is visible. Otherwise the facula
        is a bright point -- even near disk center.
    floor_teff_base_dteff : astropy.units.Quantity
        The Teff of the floor at the minimum radius.
    wall_teff_slope : astropy.units.Quantity
        The slope of the radius-Teff relationship
    wall_teff_intercept : astropy.units.Quantity
        The Teff of the wall when :math:`R = 0`.
    """
    _PRESET_PATH = PRESET_PATH / 'faculae.yaml'
    def __init__(
        self,
        distribution: str,
        equillibrium_coverage: float,
        burn_in: u.Quantity,
        mean_radius: u.Quantity,
        logsigma_radius: float,
        depth: u.Quantity,
        mean_timescale: u.Quantity,
        logsigma_timescale:float,
        floor_teff_slope: u.Quantity,
        floor_teff_min_rad: u.Quantity,
        floor_teff_base_dteff: u.Quantity,
        wall_teff_slope: u.Quantity,
        wall_teff_intercept: u.Quantity,
    ):
        self.distribution = distribution
        self.equillibrium_coverage = equillibrium_coverage
        self.burn_in = burn_in
        self.mean_radius = mean_radius
        self.logsigma_radius = logsigma_radius
        self.depth = depth
        self.mean_timescale = mean_timescale
        self.logsigma_timescale = logsigma_timescale
        self.floor_teff_slope = floor_teff_slope
        self.floor_teff_min_rad = floor_teff_min_rad
        self.floor_teff_base_dteff = floor_teff_base_dteff
        self.wall_teff_slope = wall_teff_slope
        self.wall_teff_intercept = wall_teff_intercept
        self._validate()

    def _validate(self):
        """
        Validate class instance
        """
        if self.distribution not in ['iso']:
            raise ValueError('`distribution` must be `iso`')
        if self.equillibrium_coverage > 1 or self.equillibrium_coverage < 0:
            raise ValueError(
                '`equillibrium_coverage` must be between 0 and 1.')

    @classmethod
    def _from_dict(cls, d):
        return cls(
            distribution=str(d['distribution']),
            equillibrium_coverage=float(d['equillibrium_coverage']),
            burn_in=u.Quantity(d['burn_in']),
            mean_radius=u.Quantity(d['mean_radius']),
            logsigma_radius=float(d['logsigma_radius']),
            mean_timescale=u.Quantity(d['mean_timescale']),
            logsigma_timescale=float(d['logsigma_timescale']),
            depth=u.Quantity(d['depth']),
            floor_teff_slope=u.Quantity(d['floor_teff_slope']),
            floor_teff_min_rad=u.Quantity(d['floor_teff_min_rad']),
            floor_teff_base_dteff=u.Quantity(d['floor_teff_base_dteff']),
            wall_teff_slope=u.Quantity(d['wall_teff_slope']),
            wall_teff_intercept=u.Quantity(d['wall_teff_intercept'])
        )
    @classmethod
    def from_preset(cls, name):
        """
        Load a ``FaculaParameters`` instance from a preset file.

        Parameters
        ----------
        name : str
            The name of the preset to load.
        
        Returns
        -------
        FaculaParameters
            The class instance loaded from a preset.
        """
        return super().from_preset(name)
    @classmethod
    def none(cls):
        """
        Load a parameter set without faculae.
        """
        return cls.from_preset('none')

    @classmethod
    def std(cls):
        """
        Load a parameter preset with simple standard faculae
        for testing.
        """
        return cls.from_preset('std')
    def to_generator(
        self,
        grid_params: Union[int,Tuple[int, int]] = (vsm_config.NLAT, vsm_config.NLON),
        gridmaker: CoordinateGrid = None,
        rng: np.random.Generator = np.random.default_rng()
    )->FaculaGenerator:
        """
        Construct a `vspec_vsm.FaculaGenerator` instance from the class.
        
        Parameters
        ----------
        grid_params : Union[int,Tuple[int, int]]
            If tuple, the number of grid points in the latitude and longitude
            directions. If int, the number of total grid points for a sprial
            grid. Defaults to (vsm_config.NLAT, vsm_config.NLON).
        gridmaker : CoordinateGrid
            An instance of `vspec_vsm.CoordinateGrid` to use for the grid.
            Defaults to None.
        rng : np.random.Generator
            Random number generator. Defaults to np.random.default_rng().
        
        Returns
        -------
        vspec_vsm.FaculaGenerator
            The `vspec_vsm.FaculaGenerator` instance.
        """
        return FaculaGenerator(
            dist_r_peak=self.mean_radius,
            dist_r_logsigma=self.logsigma_radius,
            depth=self.depth,
            dist_life_peak=self.mean_timescale,
            dist_life_logsigma=self.logsigma_timescale,
            floor_teff_slope=self.floor_teff_slope,
            floor_teff_min_rad=self.floor_teff_min_rad,
            floor_teff_base_dteff=self.floor_teff_base_dteff,
            wall_teff_slope=self.wall_teff_slope,
            wall_teff_intercept=self.wall_teff_intercept,
            coverage=self.equillibrium_coverage,
            dist=self.distribution,
            grid_params=grid_params,
            gridmaker=gridmaker,
            rng=rng
        )


class FlareParameters(BaseParameters):
    """
    Class to store stellar flare parameters

    Parameters
    ----------
    dist_teff_mean : astropy.units.Quantity
        The mean of the temperature distribution.
    dist_teff_sigma : astropy.units.Quantity
        The standard deviation of the temperature distribution.
    dist_fwhm_mean : astropy.units.Quantity
        The mean of the FWHM distribution.
    dist_fwhm_logsigma : float
        The standard deviation of the logorithm of the FWHM distribution in dex.
    alpha : float
        The slope of the log frequency - log energy relationship.
    beta : float
        The y-intercept of the log frequency - log energy relationship.
    min_energy : astropy.units.Quantity
        The minimum energy to consider. Set to ``np.inf*u.erg`` to disable flares.
    cluster_size : int
        The typical size of flare clusters.
    
    Attributes
    ----------
    dist_teff_mean : astropy.units.Quantity
        The mean of the temperature distribution.
    dist_teff_sigma : astropy.units.Quantity
        The standard deviation of the temperature distribution.
    dist_fwhm_mean : astropy.units.Quantity
        The mean of the FWHM distribution.
    dist_fwhm_logsigma : float
        The standard deviation of the logorithm of the FWHM distribution in dex.
    alpha : float
        The slope of the log frequency - log energy relationship.
    beta : float
        The y-intercept of the log frequency - log energy relationship.
    min_energy : astropy.units.Quantity
        The minimum energy to consider. Set to ``np.inf*u.erg`` to disable flares.
    cluster_size : int
        The typical size of flare clusters.
    """
    _PRESET_PATH = PRESET_PATH / 'flares.yaml'
    def __init__(
        self,
        dist_teff_mean: u.Quantity,
        dist_teff_sigma: u.Quantity,
        dist_fwhm_mean: u.Quantity,
        dist_fwhm_logsigma: float,
        alpha: float,
        beta: float,
        min_energy: u.Quantity,
        cluster_size: int
    ):
        self.dist_teff_mean = dist_teff_mean
        self.dist_teff_sigma = dist_teff_sigma
        self.dist_fwhm_mean = dist_fwhm_mean
        self.dist_fwhm_logsigma = dist_fwhm_logsigma
        self.alpha = alpha
        self.beta = beta
        self.min_energy = min_energy
        self.cluster_size = cluster_size
    @classmethod
    def from_preset(cls, name):
        """
        Load a ``FlareParameters`` configuration from a preset.

        Parameters
        ----------
        name : str
            The name of the preset to load.
        
        Returns
        -------
        FlareParameters
            The class instance loaded from a preset.
        """
        return super().from_preset(name)
    @classmethod
    def _from_dict(cls, d):
        return cls(
            dist_teff_mean=u.Quantity(d['dist_teff_mean']),
            dist_teff_sigma=u.Quantity(d['dist_teff_sigma']),
            dist_fwhm_mean=u.Quantity(d['dist_fwhm_mean']),
            dist_fwhm_logsigma=float(d['dist_fwhm_logsigma']),
            alpha=float(d['alpha']),
            beta=float(d['beta']),
            min_energy=u.Quantity(d['min_energy']),
            cluster_size=int(d['cluster_size'])
        )

    @classmethod
    def none(cls):
        """
        A configuration with no flares.
        """
        return cls.from_preset('none')

    @classmethod
    def std(cls):
        """
        A standard flare configuration for testing.
        """
        return cls.from_preset('std')
    def to_generator(
        self,
        rng: np.random.Generator = np.random.default_rng()
    )->FlareGenerator:
        """
        Create a `vspec_vsm.FlareGenerator` instance from the class instance.
        
        Parameters
        ----------
        rng : np.random.Generator, optional
            The random number generator to use. Defaults to np.random.default_rng().
        
        Returns
        -------
        vspec_vsm.FlareGenerator
            The `vspec_vsm.FlareGenerator` instance.
        """
        return FlareGenerator(
            dist_teff_mean=self.dist_teff_mean,
            dist_teff_sigma=self.dist_teff_sigma,
            dist_fwhm_mean=self.dist_fwhm_mean,
            dist_fwhm_logsigma=self.dist_fwhm_logsigma,
            alpha=self.alpha,
            beta=self.beta,
            min_energy=self.min_energy,
            cluster_size=self.cluster_size,
            rng=rng
        )


class GranulationParameters(BaseParameters):
    """
    Granulation Parameters

    Parameters
    ----------
    mean : float
        The mean coverage of low-teff granulation.
    amp : float
        The amplitude of granulation oscillations.
    period : astropy.units.Quantity
        The period of granulation oscillations.
    dteff : astropy.units.Quantity
        The difference between the quiet photosphere and the low-teff granulation region.

    Attributes
    ----------
    mean : float
        The mean coverage of low-teff granulation.
    amp : float
        The amplitude of granulation oscillations.
    period : astropy.units.Quantity
        The period of granulation oscillations.
    dteff : astropy.units.Quantity
        The difference between the quiet photosphere and the low-teff granulation region.
    """

    def __init__(
        self,
        mean: float,
        amp: float,
        period: u.Quantity,
        dteff: u.Quantity
    ):
        self.mean = mean
        self.amp = amp
        self.period = period
        self.dteff = dteff
        self.validate()

    def validate(self):
        if self.mean > 1 or self.mean < 0:
            raise ValueError('`mean` must be between 0 and 1.')
        if self.amp > 1 or self.amp < 0:
            raise ValueError('`amp` must be between 0 and 1.')

    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            mean=float(d['mean']),
            amp=float(d['amp']),
            period=u.Quantity(d['period']),
            dteff=u.Quantity(d['dteff'])
        )

    @classmethod
    def std(cls):
        return cls(
            0.2, 0.01, 5*u.day, 200*u.K
        )

    @classmethod
    def none(cls):
        return cls(
            0.0, 0.00, 5*u.day, 200*u.K
        )
    def to_generator(
        self,
        seed: int=0
    )->Granulation:
        """
        Create a `vspec_vsm.Granulation` instance from the class instance.
        
        Parameters
        ----------
        seed : int, optional
            The seed for the random number generator. Defaults to 0.
        
        Returns
        -------
        vspec_vsm.Granulation
            The `vspec_vsm.Granulation` instance.
        """
        return Granulation(
            mean_coverage=self.mean,
            amplitude=self.amp,
            period=self.period,
            dteff=self.dteff,
            seed=seed
        )


class StarParameters(BaseParameters):
    """
    Parameters describing the ``VSPEC`` stellar model.

    Parameters
    ----------
    psg_star_template : str
        The template used for the stellar model.
    teff : astropy.units.Quantity
        The effective temperature of the star.
    mass : astropy.units.Quantity
        The mass of the star.
    radius : astropy.units.Quantity
        The radius of the star.
    period : astropy.units.Quantity
        The rotational period of the star.
    misalignment : astropy.units.Quantity
        The misalignment between the stellar rotation axis and the orbital axis.
    misalignment_dir : astropy.units.Quantity
        The direction of stellar rotation axis misalignment.
    ld : LimbDarkeningParameters
        The limb darkening parameters of the star.
    spots : SpotParameters
        The parameters of the spots on the star.
    faculae : FaculaParameters
        The parameters of the faculae on the star.
    flares : FlareParameters
        The parameters of the flares on the star.
    granulation : GranulationParameters
        The parameters of the granulation on the star.
    Nlat : int
        Number of latitudes in the stellar surface.
    Nlon : int
        Number of longitudes in the stellar surface.
    
    Attributes
    ----------
    psg_star_template : str
        The template used for the stellar model.
    teff : astropy.units.Quantity
        The effective temperature of the star.
    mass : astropy.units.Quantity
        The mass of the star.
    radius : astropy.units.Quantity
        The radius of the star.
    period : astropy.units.Quantity
        The rotational period of the star.
    misalignment : astropy.units.Quantity
        The misalignment between the stellar rotation axis and the orbital axis.
    misalignment_dir : astropy.units.Quantity
        The direction of stellar rotation axis misalignment.
    ld : LimbDarkeningParameters
        The limb darkening parameters of the star.
    spots : SpotParameters
        The parameters of the spots on the star.
    faculae : FaculaParameters
        The parameters of the faculae on the star.
    flares : FlareParameters
        The parameters of the flares on the star.
    granulation : GranulationParameters
        The parameters of the granulation on the star.
    Nlat : int
        Number of latitudes in the stellar surface.
    Nlon : int
        Number of longitudes in the stellar surface.
    """

    def __init__(
        self,
        psg_star_template: str,
        teff: u.Quantity,
        mass: u.Quantity,
        radius: u.Quantity,
        period: u.Quantity,
        misalignment: u.Quantity,
        misalignment_dir: u.Quantity,
        ld: LimbDarkeningParameters,
        spots: SpotParameters,
        faculae: FaculaParameters,
        flares: FlareParameters,
        granulation: GranulationParameters,
        grid_params: tuple,
    ):
        self.psg_star_template = psg_star_template
        self.teff = teff
        self.mass = mass
        self.radius = radius
        self.period = period
        self.misalignment = misalignment
        self.misalignment_dir = misalignment_dir
        self.ld = ld
        self.spots = spots
        self.faculae = faculae
        self.flares = flares
        self.granulation = granulation
        self.grid_params = grid_params
    @classmethod
    def from_dict(cls, d: dict):
        """
        Construct a ``StarParameters`` object from a dictionary.

        Parameters
        ----------
        d : dict
            The dictionary representing the ``StarParameters`` object.
        
        Notes
        -----
        Available presets include ``static_proxima``, ``spotted_proxima``, ``flaring_proxima``, and ``proxima``.
        """
        return super().from_dict(d)
    @classmethod
    def _from_dict(cls, d: dict):
        try:
            grid_params = tuple(d['grid_params'])
        except TypeError:
            grid_params = int(d['grid_params'])
        return cls(
            psg_star_template=str(d['psg_star_template']),
            teff=u.Quantity(d['teff']),
            mass=u.Quantity(d['mass']),
            radius=u.Quantity(d['radius']),
            period=u.Quantity(d['period']),
            misalignment=u.Quantity(d['misalignment']),
            misalignment_dir=u.Quantity(d['misalignment_dir']),
            ld=LimbDarkeningParameters.from_dict(d['ld']),
            spots=SpotParameters.from_dict(d['spots']),
            faculae=FaculaParameters.from_dict(d['faculae']),
            flares=FlareParameters.from_dict(d['flares']),
            granulation=GranulationParameters.from_dict(d['granulation']),
            grid_params=grid_params,
        )
    def to_psg(self)->dict:
        """
        Write a dictionary containing PSG config options.

        Returns
        -------
        dict
            Configurations to send to PSG.
        """
        return {
            'OBJECT-STAR-TYPE': self.psg_star_template,
            'OBJECT-STAR-TEMPERATURE': f'{self.teff.to_value(u.K):.1f}',
            'OBJECT-STAR-RADIUS': f'{self.radius.to_value(u.R_sun):.4f}',
            'GENERATOR-CONT-STELLAR': 'Y'
        }

    @classmethod
    def static_proxima(cls):
        """
        A Proxima Centauri-like star that has no variability.
        """
        return cls(
            psg_star_template='M',
            teff=3300*u.K,
            mass=0.12*u.M_sun,
            radius=0.154*u.R_sun,
            period=40*u.day,
            misalignment=0*u.deg,
            misalignment_dir=0*u.deg,
            ld=LimbDarkeningParameters.proxima(),
            spots=SpotParameters.none(),
            faculae=FaculaParameters.none(),
            flares=FlareParameters.none(),
            granulation=GranulationParameters.none(),
            grid_params=(500, 1000),
        )

    @classmethod
    def spotted_proxima(cls):
        """
        A Proxima Centauri-like star that has spots.
        """
        return cls(
            psg_star_template='M',
            teff=3300*u.K,
            mass=0.12*u.M_sun,
            radius=0.154*u.R_sun,
            period=40*u.day,
            misalignment=0*u.deg,
            misalignment_dir=0*u.deg,
            ld=LimbDarkeningParameters.proxima(),
            spots=SpotParameters.mdwarf(),
            faculae=FaculaParameters.none(),
            flares=FlareParameters.none(),
            granulation=GranulationParameters.none(),
            grid_params=(500, 1000),
        )

    @classmethod
    def flaring_proxima(cls):
        """
        A Proxima Centauri-like star that flares.
        """
        return cls(
            psg_star_template='M',
            teff=3300*u.K,
            mass=0.12*u.M_sun,
            radius=0.154*u.R_sun,
            period=40*u.day,
            misalignment=0*u.deg,
            misalignment_dir=0*u.deg,
            ld=LimbDarkeningParameters.proxima(),
            spots=SpotParameters.none(),
            faculae=FaculaParameters.none(),
            flares=FlareParameters.std(),
            granulation=GranulationParameters.none(),
            grid_params=(500, 1000),
        )

    @classmethod
    def proxima(cls):
        """
        A Proxima Centauri-like star that has spots and flares.
        """
        return cls(
            psg_star_template='M',
            teff=3300*u.K,
            mass=0.12*u.M_sun,
            radius=0.154*u.R_sun,
            period=40*u.day,
            misalignment=0*u.deg,
            misalignment_dir=0*u.deg,
            ld=LimbDarkeningParameters.proxima(),
            spots=SpotParameters.mdwarf(),
            faculae=FaculaParameters.none(),
            flares=FlareParameters.std(),
            granulation=GranulationParameters.std(),
            grid_params=(500, 1000),
        )
    def to_star(
        self,
        rng: np.random.Generator = np.random.default_rng(),
        seed: int = 0
    )->Star:
        """
        Create a `vspec_vsm.Star` instance from the class instance.
        
        Parameters
        ----------
        rng : np.random.Generator, optional
            The random number generator to use. Defaults to np.random.default_rng().
        seed : int, optional
            The seed for the random number generator. Defaults to 0.
        """
        return Star(
            radius=self.radius,
            period=self.period,
            teff=self.teff,
            spots=SpotCollection(grid_params=self.grid_params),
            faculae=FaculaCollection(grid_params=self.grid_params),
            grid_params=self.grid_params,
            flare_generator=self.flares.to_generator(rng=rng),
            spot_generator=self.spots.to_generator(
                grid_params=self.grid_params,
                gridmaker=None,
                rng=rng
            ),
            fac_generator=self.faculae.to_generator(
                grid_params=self.grid_params,
                gridmaker=None,
                rng=rng
            ),
            granulation=self.granulation.to_generator(seed=seed),
            u1=self.ld.u1,
            u2=self.ld.u2,
            rng=rng
        )