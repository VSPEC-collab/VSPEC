"""
Planet Parameters
"""

from astropy import units as u
from VSPEC.params.base import BaseParameters
from VSPEC.config import planet_distance_unit, period_unit, planet_radius_unit


class GravityParameters(BaseParameters):
    """
    Class representing gravity parameters.

    Parameters
    ----------
    mode : str
        The mode of the gravity parameter. Valid options are 'gravity', 'density', and 'mass'.
    value : astropy.units.Quantity
        The value of the gravity parameter.

    Attributes
    ----------
    mode : str
        The mode of the gravity parameter.
    _value : astropy.units.Quantity
        The value of the gravity parameter.

    Properties
    ----------
    value : float
        The value of the gravity parameter converted to the appropriate unit based on the mode.

    Notes
    -----
    - The available modes and their corresponding units are:
        - 'g': meters per second squared (m s^-2)
        - 'rho': grams per cubic centimeter (g cm^-3)
        - 'kg': kilograms (kg)

    """

    psg_units = {
        'g': u.Unit('m s-2'),
        'rho': u.Unit('g cm-3'),
        'kg': u.kg
    }

    def __init__(
        self,
        mode: str,
        value: u.Quantity
    ):
        self.mode = mode
        self._value = value

    @property
    def value(self):
        """
        The value of the gravity parameter converted to the appropriate unit based on the mode.

        Returns
        -------
        float
            The value of the gravity parameter.
        """

        return self._value.to_value(self.psg_units[self.mode])
    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            mode = str(d['mode']),
            value = u.Quantity(d['value'])
        )
    def to_psg(self)->dict:
        """
        Convert the gravity parameter to the PSG input format.

        Returns
        -------
        dict
            A dictionary representing the gravity parameter in the PSG input format.

        """

        return {
            'OBJECT-GRAVITY': f'{self.value:.4e}',
            'OBJECT-GRAVITY-UNIT': self.mode
        }

class PlanetParameters(BaseParameters):
    """
    Class representing planet parameters.

    Parameters
    ----------
    name : str
        The name of the planet.
    radius : astropy.units.Quantity
        The radius of the planet.
    gravity : GravityParameters
        The gravity parameters of the planet.
    semimajor_axis : astropy.units.Quantity
        The semi-major axis of the planet's orbit.
    orbit_period : astropy.units.Quantity
        The period of the planet's orbit.
    rotation_period : astropy.units.Quantity
        The rotation period of the planet.
    eccentricity : float
        The eccentricity of the planet's orbit
    obliquity : astropy.units.Quantity
        The obliquity (tilt) of the planet.
    obliquity_direction : astropy.units.Quantity
        The direction of the planet's obliquity. The true anomaly
        at which the planet's north pole faces away from the star.
    init_phase : astropy.units.Quantity
        The initial phase of the planet.
    init_substellar_lon : astropy.units.Quantity
        The initial substellar longitude of the planet.

    Attributes
    ----------
    name : str
        The name of the planet.
    radius : astropy.units.Quantity
        The radius of the planet.
    gravity : GravityParameters
        The gravity parameters of the planet.
    semimajor_axis : astropy.units.Quantity
        The semi-major axis of the planet's orbit.
    orbit_period : astropy.units.Quantity
        The period of the planet's orbit.
    rotation_period : astropy.units.Quantity
        The rotation period of the planet.
    eccentricity : float
        The eccentricity of the planet's orbit
    obliquity : astropy.units.Quantity
        The obliquity (tilt) of the planet.
    obliquity_direction : astropy.units.Quantity
        The direction of the planet's obliquity. The true anomaly
        at which the planet's north pole faces away from the star.
    init_phase : astropy.units.Quantity
        The initial phase of the planet.
    init_substellar_lon : astropy.units.Quantity
        The initial substellar longitude of the planet.

    Methods
    -------
    to_psg():
        Write the parameters in the PSG config format.
    proxcenb(init_phase, init_substellar_lon)
        Initialize parameters for Proxima Centauri b.
    std(init_phase, init_substellar_lon)
        Initialize parameters for a standard exoplanet.

    Notes
    -----
    - The `proxcenb` class method initializes parameters specific to Proxima Centauri b.
    - The `std` class method initializes parameters for a standard exoplanet.

    """

    def __init__(
        self,
        name: str,
        radius: u.Quantity,
        gravity: GravityParameters,
        semimajor_axis: u.Quantity,
        orbit_period: u.Quantity,
        rotation_period: u.Quantity,
        eccentricity: float,
        obliquity: u.Quantity,
        obliquity_direction: u.Quantity,
        init_phase: u.Quantity,
        init_substellar_lon: u.Quantity
    ):
        self.name = name
        self.radius = radius
        self.gravity = gravity
        self.semimajor_axis = semimajor_axis
        self.orbit_period = orbit_period
        self.rotation_period = rotation_period
        self.eccentricity = eccentricity
        self.obliquity = obliquity
        self.obliquity_direction = obliquity_direction
        self.init_phase = init_phase
        self.init_substellar_lon = init_substellar_lon

    def to_psg(self):
        """
        Convert the parameters to the PSG config format.

        Returns
        -------
        dict
            A dictionary containing the parameters in the PSG config format.

        """
        psg_dict = {
            'OBJECT': 'Exoplanet',
            'OBJECT-NAME': self.name,
            'OBJECT-DIAMETER': f'{2*self.radius.to_value(planet_radius_unit):.4f}',
            'OBJECT-STAR-DISTANCE': f'{self.semimajor_axis.to_value(planet_distance_unit):.4f}',
            'OBJECT-PERIOD': f'{self.orbit_period.to_value(period_unit):.4f}',
            'OBJECT-ECCENTRICITY': f'{self.eccentricity:.4f}',
        }
        psg_dict.update(self.gravity.to_psg())
        return psg_dict

    @classmethod
    def _from_dict(cls,d:dict):
        return cls(
            name = str(d['name']),
            radius = u.Quantity(d['radius']),
            gravity = GravityParameters.from_dict(d['gravity']),
            semimajor_axis = u.Quantity(d['semimajor_axis']),
            orbit_period = u.Quantity(d['orbit_period']),
            rotation_period = u.Quantity(d['rotation_period']),
            eccentricity = float(d['eccentricity']),
            obliquity = u.Quantity(d['obliquity']),
            obliquity_direction = u.Quantity(d['obliquity_direction']),
            init_phase = u.Quantity(d['init_phase']),
            init_substellar_lon = u.Quantity(d['init_substellar_lon'])
        )
    @classmethod
    def proxcenb(cls, init_phase: u.Quantity, init_substellar_lon: u.Quantity):
        """
        Proxima Centauri b :cite:p:`2022A&A...658A.115F`

        Parameters
        ----------
        init_phase : astropy.units.Quantity
            The initial phase of the planet.
        init_substellar_lon : astropy.units.Quantity
            The initial subsolar longitude of the planet.

        Returns
        -------
        PlanetParameters
            The planet parameters for Proxima Centauri b.


        """
        return cls(
            name='Prox Cen b',
            radius=1.03*u.M_earth,  # Earth density
            gravity=GravityParameters('kg', 1.07*u.M_earth),
            semimajor_axis=0.04856*u.AU,
            orbit_period=11.1868*u.day,
            rotation_period=11.1868*u.day,
            eccentricity=0.,
            obliquity=0*u.deg,
            obliquity_direction=0*u.deg,
            init_phase=init_phase,
            init_substellar_lon=init_substellar_lon
        )

    @classmethod
    def std(cls, init_phase: u.Quantity, init_substellar_lon: u.Quantity):
        """
        The default VSPEC planet.

        Parameters
        ----------
        init_phase : astropy.units.Quantity
            The initial phase of the planet.
        init_substellar_lon : astropy.units.Quantity
            The initial subsolar longitude of the planet.

        Returns
        -------
        PlanetParameters
            The planet parameters for the default VSPEC exoplanet.

        """
        return cls(
            name='Exoplanet',
            radius=1.*u.M_earth,
            gravity=GravityParameters('kg', 1.0*u.M_earth),
            semimajor_axis=0.05*u.AU,
            orbit_period=10*u.day,
            rotation_period=10*u.day,
            eccentricity=0.,
            obliquity=0*u.deg,
            obliquity_direction=0*u.deg,
            init_phase=init_phase,
            init_substellar_lon=init_substellar_lon
        )


class SystemParameters:
    """
    Class representing system parameters.

    Parameters
    ----------
    distance : astropy.units.Quantity
        The distance to the system.
    inclination : astropy.units.Quantity
        The inclination angle of the system. Transit occurs at 90 degrees.
    phase_of_periasteron : astropy.units.Quantity
        The phase (as seen from the observer) of the planet when it reaches periasteron.

    Attributes
    ----------
    distance : astropy.units.Quantity
        The distance to the system.
    inclination : astropy.units.Quantity
        The inclination angle of the system. Transit occurs at 90 degrees.
    phase_of_periasteron : astropy.units.Quantity
        The phase (as seen from the observer) of the planet when it reaches periasteron.

    """

    def __init__(
        self,
        distance,
        inclination,
        phase_of_periasteron
    ):
        self.distance = distance
        self.inclination = inclination
        self.phase_of_periasteron = phase_of_periasteron
