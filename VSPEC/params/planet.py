"""
Planet Parameters
"""

from astropy import units as u

class GravityParameters:
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
        - 'gravity': meters per second squared (m s^-2)
        - 'density': grams per cubic centimeter (g cm^-3)
        - 'mass': kilograms (kg)

    """

    psg_units = {
        'gravity': u.Unit('m s-2'),
        'density': u.Unit('g cm-3'),
        'mass' : u.kg
    }
    def __init__(
        self,
        mode:str,
        value:u.Quantity
    ):
        self.mode=mode
        self._value = value
    @property
    def value(self):
        return self._value.to_value(self.psg_units[self.mode])

class PlanetParameters:
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
    obliquity : astropy.units.Quantity
        The obliquity (tilt) of the planet.
    obliquity_direction : astropy.units.Quantity
        The direction of the planet's obliquity.
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
    obliquity : astropy.units.Quantity
        The obliquity (tilt) of the planet.
    obliquity_direction : astropy.units.Quantity
        The direction of the planet's obliquity.
    init_phase : astropy.units.Quantity
        The initial phase of the planet.
    init_substellar_lon : astropy.units.Quantity
        The initial substellar longitude of the planet.

    Methods
    -------
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
        name:str,
        radius:u.Quantity,
        gravity:GravityParameters,
        semimajor_axis:u.Quantity,
        orbit_period:u.Quantity,
        rotation_period:u.Quantity,
        obliquity:u.Quantity,
        obliquity_direction:u.Quantity,
        init_phase:u.Quantity,
        init_substellar_lon:u.Quantity
    ):
        self.name = name
        self.radius = radius
        self.gravity = gravity
        self.semimajor_axis = semimajor_axis
        self.orbit_period = orbit_period
        self.rotation_period = rotation_period
        self.obliquity = obliquity
        self.obliquity_direction = obliquity_direction
        self.init_phase = init_phase
        self.init_substellar_lon = init_substellar_lon
    @classmethod
    def proxcenb(cls,init_phase:u.Quantity,init_substellar_lon:u.Quantity):
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
            radius = 1.03*u.M_earth, # Earth density
            gravity = GravityParameters('mass',1.07*u.M_earth),
            semimajor_axis = 0.04856*u.AU,
            orbit_period = 11.1868*u.day,
            rotation_period = 11.1868*u.day,
            obliquity = 0*u.deg,
            obliquity_direction = 0*u.deg,
            init_phase = init_phase,
            init_substellar_lon = init_substellar_lon
        )
    @classmethod
    def std(cls,init_phase:u.Quantity,init_substellar_lon:u.Quantity):
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
            radius = 1.*u.M_earth,
            gravity = GravityParameters('mass',1.0*u.M_earth),
            semimajor_axis = 0.05*u.AU,
            orbit_period = 10*u.day,
            rotation_period = 10*u.day,
            obliquity = 0*u.deg,
            obliquity_direction = 0*u.deg,
            init_phase = init_phase,
            init_substellar_lon = init_substellar_lon
        )

