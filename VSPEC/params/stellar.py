"""
Stellar Parameters
"""



from pathlib import Path
from astropy import units as u
from typing import Union

from VSPEC.helpers import MSH

class LimbDarkeningParameters:
    """
    Limb Darkening Parameters for the Quadratic
    Limb Darkening Law

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
    """
    def __init__(self,u1:float,u2:float):
        self.u1 = u1
        self.u2 = u2
    @classmethod
    def solar(cls):
        """
        From 2012A&A...546A..14C
        S4 bandpass
        """
        return cls(0.0473,0.0841)
    @classmethod
    def proxima(cls):
        """
        From 2012A&A...546A..14C
        S4 bandpass
        """
        return cls(0.0551,0.1075)
    @classmethod
    def trappist(cls):
        """
        From 2012A&A...546A..14C
        S4 bandpass
        """
        return cls(0.0778,0.1619)
    @classmethod
    def lambertian(cls):
        """
        No limb darkening.
        """
        return cls(1.,0.)

class SpotParameters:
    """
    Spot Parameters
    """
    def __init__(
        self,
        distribution:str,
        initial_coverage:float,
        equillibrium_coverage:float,
        warmup:u.Quantity,
        area_logmean:u.LogQuantity,
        area_logsigma:float,
        teff_umbra:u.Quantity,
        teff_penumbra:u.Quantity,
        growth_rate:u.Quantity,
        decay_rate:u.Quantity,
        initial_area:u.Quantity
        ):
        self.distribution = distribution
        self.initial_coverage = initial_coverage
        self.equillibrium_coverage = equillibrium_coverage
        self.warmup = warmup
        self.area_logmean = area_logmean
        self.area_logsigma = area_logsigma
        self.teff_umbra = teff_umbra
        self.teff_penumbra = teff_penumbra
        self.growth_rate = growth_rate
        self.decay_rate = decay_rate
        self.initial_area = initial_area
        self.validate()
    def validate(self):
        if self.distribution not in ['solar','iso']:
            raise ValueError('`distribution` must either be `solar` or `iso`')
        if self.initial_coverage > 1 or self.initial_coverage < 0:
            raise ValueError('`initial_coverage` must be between 0 and 1.')
        if self.equillibrium_coverage > 1 or self.equillibrium_coverage < 0:
            raise ValueError('`equillibrium_coverage` must be between 0 and 1.')
    @classmethod
    def none(cls):
        """
        No spots
        """
        return cls(
            'iso',0.,0.,0.,
            u.LogQuantity(500*MSH),0.2,
            100*u.K,100*u.K,
            0./u.day,0*MSH/u.day,
            10*u.MSH
        )
    @classmethod
    def mdwarf(cls):
        """
        Static Spots
        """
        return cls(
            'iso',0.2,0.,0.,
            u.LogQuantity(500*MSH),0.2,
            2500*u.K,2700*u.K,
            0./u.day,0*MSH/u.day,
            10*u.MSH
        )
    @classmethod
    def solar(cls):
        """
        Solar-style spots
        """
        return cls(
            'solar',0.1,0.1,30*u.day,
            u.LogQuantity(500*MSH),0.2,
            2500*u.K,2700*u.K,
            0.52/u.day,10.8*MSH/u.day,
            10*u.MSH
        )
class FaculaParameters:
    """
    Facula Parameters
    """
    def __init__(
        self,
        distribution:str,
        equillibrium_coverage:float,
        warmup:u.Quantity,
        mean_radius:u.Quantity,
        hwhm_radius:u.Quantity,
        mean_timescale:u.Quantity,
        hwhm_timescale:u.Quantity,
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
        if self.distribution not in ['iso']:
            raise ValueError('`distribution` must be `iso`')
        if self.equillibrium_coverage > 1 or self.equillibrium_coverage < 0:
            raise ValueError('`equillibrium_coverage` must be between 0 and 1.')
    @classmethod
    def none(cls):
        return cls(
            'iso',0.000,0*u.s,
            500*u.km,200*u.km,
            10*u.hr,4*u.hr,
        )
    @classmethod
    def std(cls):
        return cls(
            'iso',0.001,30*u.hr,
            500*u.km,200*u.km,
            10*u.hr,4*u.hr
        )

class FlareParameters:
    def __init__(
        self,
        group_probability:float,
        teff_mean:u.Quantity,
        teff_sigma:u.Quantity,
        fwhm_mean:u.LogQuantity,
        fwhm_sigma:float,
        E_min:u.LogQuantity,
        E_max:u.LogQuantity,
        E_steps:int
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
            0.5,9000*u.K,500*u.K,
            u.LogQuantity(0.14*u.day),0.3,
            u.LogQuantity(10**32.5*u.erg),u.LogQuantity(10**34.5*u.erg),
            0
        )
    @classmethod
    def std(cls):
        return cls(
            0.5,9000*u.K,500*u.K,
            u.LogQuantity(0.14*u.day),0.3,
            u.LogQuantity(10**32.5*u.erg),u.LogQuantity(10**34.5*u.erg),
            100
        )

class GranulationParameters:
    """
    Granulation Parameters
    """
    def __init__(
        self,
        mean:float,
        amp:float,
        period:u.Quantity,
        dteff:u.Quantity
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
            0.2,0.01,5*u.day,200*u.K
        )
    @classmethod
    def none(cls):
        return cls(
            0.0,0.00,5*u.day,200*u.K
        )

class StarParameters:
    """
    Class to store stellar model parameters.
    """
    def __init__(
        self,
        template:str,
        teff:u.Quantity,
        mass:u.Quantity,
        radius:u.Quantity,
        period:u.Quantity,
        offset_magnitude:u.Quantity,
        offset_direction:u.Quantity,
        ld:LimbDarkeningParameters,
        spots:SpotParameters,
        faculae:FaculaParameters,
        flares:FlareParameters,
        granulation:GranulationParameters
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
            teff = 3300*u.K,
            mass = 0.12*u.M_sun,
            radius = 0.154*u.R_sun,
            period = 40*u.day,
            offset_magnitude = 0*u.deg,
            offset_direction = 0*u.deg,
            ld = LimbDarkeningParameters.proxima(),
            spots = SpotParameters.none(),
            faculae = FaculaParameters.none(),
            flares = FlareParameters.none(),
            granulation = GranulationParameters.none()
        )
    @classmethod
    def spotted_proxima(cls):
        return cls(
            template='M',
            teff = 3300*u.K,
            mass = 0.12*u.M_sun,
            radius = 0.154*u.R_sun,
            period = 40*u.day,
            offset_magnitude = 0*u.deg,
            offset_direction = 0*u.deg,
            ld = LimbDarkeningParameters.proxima(),
            spots = SpotParameters.mdwarf(),
            faculae = FaculaParameters.none(),
            flares = FlareParameters.none(),
            granulation = GranulationParameters.none()
        )
    @classmethod
    def flaring_proxima(cls):
        return cls(
            template='M',
            teff = 3300*u.K,
            mass = 0.12*u.M_sun,
            radius = 0.154*u.R_sun,
            period = 40*u.day,
            offset_magnitude = 0*u.deg,
            offset_direction = 0*u.deg,
            ld = LimbDarkeningParameters.proxima(),
            spots = SpotParameters.none(),
            faculae = FaculaParameters.none(),
            flares = FlareParameters.std(),
            granulation = GranulationParameters.none()
        )
    @classmethod
    def proxima(cls):
        return cls(
            template='M',
            teff = 3300*u.K,
            mass = 0.12*u.M_sun,
            radius = 0.154*u.R_sun,
            period = 40*u.day,
            offset_magnitude = 0*u.deg,
            offset_direction = 0*u.deg,
            ld = LimbDarkeningParameters.proxima(),
            spots = SpotParameters.mdwarf(),
            faculae = FaculaParameters.none(),
            flares = FlareParameters.std(),
            granulation = GranulationParameters.std()
        )