"""
Stellar Parameters
"""
from astropy import units as u
import numpy as np
from VSPEC.config import stellar_area_unit

from VSPEC.helpers import MSH


class LimbDarkeningParameters:
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
    Quadratic Law (Kopal, 1950)
    .. math::

        \frac{I(\mu)}{I(1)} = 1 - u_1 (1-mu) - u_2 (1-mu)^2

    We use values taken from 
    """

    def __init__(self, u1: float, u2: float):
        self.u1 = u1
        self.u2 = u2

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
        return cls(1., 0.)


class SpotParameters:
    """
    Spot Parameters

    Parameters
    -----------
    distribution : str
        The distribution function to be used for the spot positions. 'iso' or 'solar'.
    initial_coverage : float
        The coverage for a 'hot start'.
    equillibrium_coverage : float
        The fractional coverage of the star's surface by spots. This is the value
        at growth-decay equillibrium, and different from the 'hot start' value given
        by `initial_coverage`.
    warmup : astropy.units.Quantity
        The duration of the warmup period, during which the spot coverage approaches
        equillibrium.
    area_mean : astropy.units.quantity
        The mean area of a spot on the star's surface is MSH.
    area_logsigma : float
        The standard deviation of the spot areas. This is a lognormal
        distribution, so the units of this value are dex.
    teff_umbra : astropy.units.Quantity
        The effective temperature of the spot umbrae.
    teff_penumbra : astropy.units.Quantity
        The effective temperature of the spot penumbrae.
    growth_rate : astropy.units.Quantity
        The rate at which new spots grow.
    decay_rate : astropy.units.Quantity
        The rate at which existing spots decay.
    initial_area : astropy.units.Quantity
        The initial area of newly created spots.

    Attrributes
    -----------
    distribution : str
        The distribution function to be used for the spot positions. 'iso' or 'solar'.
    initial_coverage : float
        The coverage for a 'hot start'.
    equillibrium_coverage : float
        The fractional coverage of the star's surface by spots. This is the value
        at growth-decay equillibrium, and different from the 'hot start' value given
        by `initial_coverage`.
    warmup : astropy.units.Quantity
        The duration of the warmup period, during which the spot coverage approaches
        equillibrium.
    area_logmean : float
        The mean area of a spot on the star's surface is MSH.
    area_logsigma : float
        The standard deviation of the spot areas. This is a lognormal
        distribution, so the units of this value are dex.
    teff_umbra : astropy.units.Quantity
        The effective temperature of the spot umbrae.
    teff_penumbra : astropy.units.Quantity
        The effective temperature of the spot penumbrae.
    growth_rate : astropy.units.Quantity
        The rate at which new spots grow.
    decay_rate : astropy.units.Quantity
        The rate at which existing spots decay.
    initial_area : astropy.units.Quantity
        The initial area of newly created spots.
    """

    def __init__(
        self,
        distribution: str,
        initial_coverage: float,
        equillibrium_coverage: float,
        warmup: u.Quantity,
        area_mean: u.Quantity,
        area_logsigma: float,
        teff_umbra: u.Quantity,
        teff_penumbra: u.Quantity,
        growth_rate: u.Quantity,
        decay_rate: u.Quantity,
        initial_area: u.Quantity
    ):
        self.distribution = distribution
        self.initial_coverage = initial_coverage
        self.equillibrium_coverage = equillibrium_coverage
        self.warmup = warmup
        self.area_logmean = np.log10(area_mean/stellar_area_unit)
        self.area_logsigma = area_logsigma
        self.teff_umbra = teff_umbra
        self.teff_penumbra = teff_penumbra
        self.growth_rate = growth_rate
        self.decay_rate = decay_rate
        self.initial_area = initial_area
        self.validate()

    def validate(self):
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
    def none(cls):
        """
        No spots
        """
        return cls(
            'iso', 0., 0., 0.,
            u.LogQuantity(500*MSH), 0.2,
            100*u.K, 100*u.K,
            0./u.day, 0*MSH/u.day,
            10*u.MSH
        )

    @classmethod
    def mdwarf(cls):
        """
        Static Spots
        """
        return cls(
            'iso', 0.2, 0., 0.,
            u.LogQuantity(500*MSH), 0.2,
            2500*u.K, 2700*u.K,
            0./u.day, 0*MSH/u.day,
            10*u.MSH
        )

    @classmethod
    def solar(cls):
        """
        Solar-style spots
        """
        return cls(
            'solar', 0.1, 0.1, 30*u.day,
            u.LogQuantity(500*MSH), 0.2,
            2500*u.K, 2700*u.K,
            0.52/u.day, 10.8*MSH/u.day,
            10*u.MSH
        )


class FaculaParameters:
    """
    Facula Parameters

    Parameters
    ----------
    distribution : str
        The distribution used to generate the faculae on the star. Currently only 'iso' is supported.
    equillibrium_coverage : float
        The fraction of the star's surface covered by the faculae at growth-decay equilibrium.
    warmup : astropy.units.Quantity [time]
        The warmup time for the faculae on the star to reach equilibrium.
    mean_radius : astropy.units.quantity.Quantity [distance]
        The mean radius of the faculae.
    hwhm_radius : astropy.units.quantity.Quantity [distance]
        The half-width at half-maximum radius of the faculae. It is the difference between the peak of the radius
        distribution and the half maximum in the positive direction.
    mean_timescale : astropy.units.quantity.Quantity [time]
        The mean faculae lifetime.
    hwhm_timescale : astropy.units.quantity.Quantity [time]
        The facula timescale distribution half-width-half-maximum in hours. It is the difference between the peak of
        the timescale distribution and the half maximum in the positive direction.

    Attributes
    ----------
    distribution : str
        The distribution used to generate the faculae on the star.
    equillibrium_coverage : float
        The fraction of the star's surface covered by the faculae at growth-decay equilibrium.
    warmup : astropy.units.Quantity
        The warmup time for the faculae on the star to reach equilibrium.
    mean_radius : astropy.units.quantity.Quantity
        The mean radius of the faculae.
    hwhm_radius : astropy.units.quantity.Quantity
        The half-width at half-maximum radius of the faculae.
    mean_timescale : astropy.units.quantity.Quantity
        The mean faculae lifetime.
    hwhm_timescale : astropy.units.quantity.Quantity
        The facula timescale distribution half-width-half-maximum in hours.
    """

    def __init__(
        self,
        distribution: str,
        equillibrium_coverage: float,
        warmup: u.Quantity,
        mean_radius: u.Quantity,
        hwhm_radius: u.Quantity,
        mean_timescale: u.Quantity,
        hwhm_timescale: u.Quantity,
    ):
        self.distribution = distribution
        self.equillibrium_coverage = equillibrium_coverage
        self.warmup = warmup
        self.mean_radius = mean_radius
        self.hwhm_radius = hwhm_radius
        self.mean_timescale = mean_timescale
        self.hwhm_timescale = hwhm_timescale
        self.validate()

    def validate(self):
        """
        Validate class instance
        """
        if self.distribution not in ['iso']:
            raise ValueError('`distribution` must be `iso`')
        if self.equillibrium_coverage > 1 or self.equillibrium_coverage < 0:
            raise ValueError(
                '`equillibrium_coverage` must be between 0 and 1.')

    @classmethod
    def none(cls):
        return cls(
            'iso', 0.000, 0*u.s,
            500*u.km, 200*u.km,
            10*u.hr, 4*u.hr,
        )

    @classmethod
    def std(cls):
        return cls(
            'iso', 0.001, 30*u.hr,
            500*u.km, 200*u.km,
            10*u.hr, 4*u.hr
        )


class FlareParameters:
    """
    Parameters
    ----------
    group_probability : float
        The probability that a given flare will be closely followed by another flare.
    teff_mean : astropy.units.quantity.Quantity [temperature]
        The mean temperature of the flare blackbody.
    teff_sigma : astropy.units.quantity.Quantity [temperature]
        The standard deviation of the generated flare temperature.
    fwhm_mean : astropy.units.LogQuantity
        The mean logarithm of the full width at half maximum (FWHM) of the flare in days.
    fwhm_sigma : float
        The standard deviation of the logarithm of the FWHM of the flare in days.
    E_min : astropy.units.LogQuantity
        Log of the minimum energy flares to be considered in ergs.
    E_max : astropy.units.LogQuantity
        Log of the maximum energy flares to be considered in ergs.
    E_steps : int
        The number of flare energy steps to consider.

    Attributes
    ----------
    group_probability : float
        The probability that a given flare will be closely followed by another flare.
    teff_mean : astropy.units.quantity.Quantity [temperature]
        The mean temperature of the flare blackbody.
    teff_sigma : astropy.units.quantity.Quantity [temperature]
        The standard deviation of the generated flare temperature.
    fwhm_mean : astropy.units.LogQuantity
        The mean logarithm of the full width at half maximum (FWHM) of the flare in days.
    fwhm_sigma : float
        The standard deviation of the logarithm of the FWHM of the flare in days.
    E_min : astropy.units.LogQuantity
        Log of the minimum energy flares to be considered in ergs.
    E_max : astropy.units.LogQuantity
        Log of the maximum energy flares to be considered in ergs.
    E_steps : int
        The number of flare energy steps to consider.
    """

    def __init__(
        self,
        group_probability: float,
        teff_mean: u.Quantity,
        teff_sigma: u.Quantity,
        fwhm_mean: u.LogQuantity,
        fwhm_sigma: float,
        E_min: u.LogQuantity,
        E_max: u.LogQuantity,
        E_steps: int
    ):
        self.group_probability = group_probability
        self.teff_mean = teff_mean
        self.teff_sigma = teff_sigma
        self.fwhm_mean = fwhm_mean
        self.fwhm_sigma = fwhm_sigma
        self.E_min = E_min
        self.E_max = E_max
        self.E_steps = E_steps

    @classmethod
    def none(cls):
        return cls(
            0.5, 9000*u.K, 500*u.K,
            u.LogQuantity(0.14*u.day), 0.3,
            u.LogQuantity(10**32.5*u.erg), u.LogQuantity(10**34.5*u.erg),
            0
        )

    @classmethod
    def std(cls):
        return cls(
            0.5, 9000*u.K, 500*u.K,
            u.LogQuantity(0.14*u.day), 0.3,
            u.LogQuantity(10**32.5*u.erg), u.LogQuantity(10**34.5*u.erg),
            100
        )


class GranulationParameters:
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
    def std(cls):
        return cls(
            0.2, 0.01, 5*u.day, 200*u.K
        )

    @classmethod
    def none(cls):
        return cls(
            0.0, 0.00, 5*u.day, 200*u.K
        )


class StarParameters:
    """
    Class to store stellar model parameters.

    Parameters
    ----------
    template : str
        The template used for the stellar model.
    teff : astropy.units.quantity.Quantity
        The effective temperature of the star.
    mass : astropy.units.quantity.Quantity
        The mass of the star.
    radius : astropy.units.quantity.Quantity
        The radius of the star.
    period : astropy.units.quantity.Quantity
        The rotational period of the star.
    offset_magnitude : astropy.units.quantity.Quantity
        The magnitude of the stellar rotation axis offset.
    offset_direction : astropy.units.quantity.Quantity
        The direction offset of the stellar rotation axis offset.
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

    Attributes
    ----------
    template : str
        The template used for the stellar model.
    teff : astropy.units.quantity.Quantity
        The effective temperature of the star.
    mass : astropy.units.quantity.Quantity
        The mass of the star.
    radius : astropy.units.quantity.Quantity
        The radius of the star.
    period : astropy.units.quantity.Quantity
        The rotational period of the star.
    offset_magnitude : astropy.units.quantity.Quantity
        The magnitude of the stellar rotation axis offset.
    offset_direction : astropy.units.quantity.Quantity
        The direction offset of the stellar rotation axis offset.
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
    """

    def __init__(
        self,
        template: str,
        teff: u.Quantity,
        mass: u.Quantity,
        radius: u.Quantity,
        period: u.Quantity,
        offset_magnitude: u.Quantity,
        offset_direction: u.Quantity,
        ld: LimbDarkeningParameters,
        spots: SpotParameters,
        faculae: FaculaParameters,
        flares: FlareParameters,
        granulation: GranulationParameters
    ):
        self.template = template
        self.teff = teff
        self.mass = mass
        self.radius = radius
        self.period = period
        self.offset_magnitude = offset_magnitude
        self.offset_direction = offset_direction
        self.ld = ld
        self.spots = spots
        self.faculae = faculae
        self.flares = flares
        self.granulation = granulation

    @classmethod
    def static_proxima(cls):
        return cls(
            template='M',
            teff=3300*u.K,
            mass=0.12*u.M_sun,
            radius=0.154*u.R_sun,
            period=40*u.day,
            offset_magnitude=0*u.deg,
            offset_direction=0*u.deg,
            ld=LimbDarkeningParameters.proxima(),
            spots=SpotParameters.none(),
            faculae=FaculaParameters.none(),
            flares=FlareParameters.none(),
            granulation=GranulationParameters.none()
        )

    @classmethod
    def spotted_proxima(cls):
        return cls(
            template='M',
            teff=3300*u.K,
            mass=0.12*u.M_sun,
            radius=0.154*u.R_sun,
            period=40*u.day,
            offset_magnitude=0*u.deg,
            offset_direction=0*u.deg,
            ld=LimbDarkeningParameters.proxima(),
            spots=SpotParameters.mdwarf(),
            faculae=FaculaParameters.none(),
            flares=FlareParameters.none(),
            granulation=GranulationParameters.none()
        )

    @classmethod
    def flaring_proxima(cls):
        return cls(
            template='M',
            teff=3300*u.K,
            mass=0.12*u.M_sun,
            radius=0.154*u.R_sun,
            period=40*u.day,
            offset_magnitude=0*u.deg,
            offset_direction=0*u.deg,
            ld=LimbDarkeningParameters.proxima(),
            spots=SpotParameters.none(),
            faculae=FaculaParameters.none(),
            flares=FlareParameters.std(),
            granulation=GranulationParameters.none()
        )

    @classmethod
    def proxima(cls):
        return cls(
            template='M',
            teff=3300*u.K,
            mass=0.12*u.M_sun,
            radius=0.154*u.R_sun,
            period=40*u.day,
            offset_magnitude=0*u.deg,
            offset_direction=0*u.deg,
            ld=LimbDarkeningParameters.proxima(),
            spots=SpotParameters.mdwarf(),
            faculae=FaculaParameters.none(),
            flares=FlareParameters.std(),
            granulation=GranulationParameters.std()
        )
