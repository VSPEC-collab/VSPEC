"""VSPEC helpers module

This module contains functions that may be shared
throughout the rest of the package, especially
pertainting to safe-casting of astropy.units
objects.
"""
import warnings
from astropy import units as u
from numpy import isclose as np_isclose
import numpy as np
import pandas as pd
import socket
from os import system

MSH = u.def_unit('msh', 1e-6 * 0.5 * 4*np.pi*u.R_sun**2)
"""Micro-solar hemisphere

This is a standard unit in heliophysics that
equals one millionth of one half the surface area of the Sun.
"""


def to_float(quant: u.Quantity, unit: u.Unit) -> float:
    """
    Cast to float

    Cast an `astropy.Quantity` to a float given a unit.

    Parameters
    ----------
    quant : astropy.units.Quantity
        Quantity to be cast to float
    unit : astropy.units.Unit
        Unit to be used when casting `quant`

    Returns
    -------
    float
        `quant` cast to `unit`

    Warns
    -----
    RuntimeWarning
        If `quant` is of type `float` and is converted to a
        dimensionless unit
    """
    if isinstance(quant, float) and unit == u.dimensionless_unscaled:
        message = 'Value passed to `to_float() is already a float`'
        warnings.warn(message, category=RuntimeWarning)
    return (quant/unit).to(u.Unit('')).value


def isclose(quant1: u.Quantity, quant2: u.Quantity, tol: u.Quantity) -> bool:
    """
    Check if two quantities are close

    Use `numpy.isclose` on two quantity objects.
    This function safely casts them to floats first.

    Parameters
    ----------
    param1 : astropy.units.Quantity
        First object for comparison
    param2 : astropy.units.Quantity
        Second object for comparison
    tol : astropy.units.Quantity
        Error tolerance between `param1` and `param2`

    Returns
    -------
    bool
        Whether `param1` and `param2` are within `tol`
    """
    unit = tol.unit
    return np_isclose(to_float(quant1, unit), to_float(quant2, unit), atol=to_float(tol, unit))


def get_transit_radius(
    system_distance: u.Quantity[u.pc],
    stellar_radius: u.Quantity[u.R_sun],
    semimajor_axis: u.Quantity[u.AU],
    planet_radius: u.Quantity[u.R_earth]
) -> u.Quantity[u.rad]:
    """
    Get the phase radius of a planetary transit.

    Calculate the radius from mid-transit where there is
    some overlap between the planetary and stellar disk.
    This is an approximation that asumes a circular orbit
    and i=90 deg.

    Parameters
    ----------
    system_distance : astropy.units.Quantity
        Heliocentric distance to the host star.
    stellar_radius : astropy.units.Quantity
        Radius of the host star.
    semimajor-axis : astropy.units.Quantity
        Semimajor axis of the planet's orbit.
    planet_radius : astropy.units.Quantity
        Radius of the planet

    Returns
    -------
    astropy.units.Quantity
        The maximum radius from mid-transit where
        there is stellar and planetary disk overlap
    """
    radius_over_semimajor_axis = to_float(
        stellar_radius/semimajor_axis, u.Unit(''))
    radius_over_distance = to_float(stellar_radius/system_distance, u.Unit(''))
    angle_point_planet = np.arcsin(radius_over_semimajor_axis*np.cos(
        radius_over_distance)) - radius_over_distance  # float in radians
    planet_radius_angle = to_float(
        planet_radius/(2*np.pi*semimajor_axis), u.Unit(''))
    return (angle_point_planet+planet_radius_angle)*u.rad


def is_port_in_use(port: int) -> bool:
    """
    Check if a port is in use on your machine.
    This is useful to keep from calling PSG when it
    is not running.

    Parameters
    ----------
    port : int
        The port that PSG is running on. If you
        followed the online instructions, this should
        be 3000.

    Returns
    -------
    bool
        Whether or not something is running on port `port`.
    """
    socket_obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    return socket_obj.connect_ex(('localhost', port)) == 0

def set_psg_state(running:bool):
    """
    Set the local PSG state.

    Parameters
    ----------
    running : bool
        Whether the end PSG state should be running.
    """
    psg_port = 3000
    if is_port_in_use(psg_port) and not running:
        system('docker stop psg')
    elif not is_port_in_use(psg_port) and running:
        system('docker start psg')
        


def arrange_teff(minteff: u.Quantity, maxteff: u.Quantity):
    """
    Get a list of Teff values with steps of `100 K` that fully encompase
    the min and max values. This is done to get a list of
    the spectra that it is necessary to bin for later.

    Parameters
    ----------
    minteff : astropy.units.Quantity
        The lowest Teff required
    maxteff : astropy.units.Quantity
        The highest Teff required

    Returns
    -------
    teffs : astropy.units.Quantity
        An array of Teff values.
    """
    step = 100*u.K
    if (minteff % step) == 0*u.K:
        low = minteff
    else:
        low = minteff - (minteff % step)
    if (maxteff % step) == 0*u.K:
        high = maxteff
    else:
        high = maxteff - (maxteff % step) + step
    number_of_steps = to_float((high-low)/step, u.dimensionless_unscaled)
    number_of_steps = int(round(number_of_steps))
    teffs = low + np.arange(number_of_steps+1)*step
    return teffs


def get_surrounding_teffs(Teff: u.Quantity):
    """
    Get the Teffs of the two spectra to interpolate between
    to obtain a spectrum with Teff `Teff`

    Parameters
    ----------
    Teff : astropy.units.Quantity
        The target Teff

    Returns
    -------
    low_teff : astropy.units.Quantity
        The spectrum teff below `Teff`
    high_teff : astropy.units.Quantity
        The spectrum teff above `Teff`

    Raises
    ------
    ValueError
        If `Teff` is a multiple of 100 K
    """
    step = 100*u.K
    if (Teff % step) == 0*u.K:
        raise ValueError(
            f'Teff of {Teff} is a multiple of {100*u.K}. This will cause problems with scipy.')
    else:
        low_teff = Teff - (Teff % step)
        high_teff = low_teff + step
    return low_teff, high_teff


def plan_to_df(observation_plan:dict)->pd.DataFrame:
    """
    Turn an observation plan dictionary into a pandas DataFrame.

    Parameters
    ----------
    observation_plan : dict
        A dictionary that contains arrays of geometric values at each epoch.
        The keys are:\n
        ``{'time', 'phase', 'sub_obs_lat', 'sub_obs_lon',
        'sub_planet_lat', 'sub_planet_lon', 'sub_stellar_lon',
        'sub_stellar_lat', 'planet_sub_obs_lon', 'planet_sub_obs_lat',
        'orbit_radius'}``
    
    Returns
    -------
    pandas.DataFrame
        A dataframe containing the dictionary data.
    """
    obs_df = pd.DataFrame()
    for key in observation_plan.keys():
        try:
            unit = observation_plan[key].unit
            name = f'{key}[{str(unit)}]'
            obs_df[name] = observation_plan[key].value
        except AttributeError:
            unit = ''
            name = f'{key}[{str(unit)}]'
            obs_df[name] = observation_plan[key]
    return obs_df


class CoordinateGrid:
    """
    Class to standardize the creation of latitude and longitude grids.

    Parameters
    ----------
    Nlat : int, default=500
        Number of latitude points.
    Nlon : int, default=1000
        Number of longitude points.

    Attributes
    ----------
    Nlat : int, default=500
        Number of latitude points.
    Nlon : int, default=1000
        Number of longitude points.

    """

    def __init__(self, Nlat=500, Nlon=1000):
        if not isinstance(Nlat, int):
            raise TypeError('Nlat must be int')
        if not isinstance(Nlon, int):
            raise TypeError('Nlon must be int')
        self.Nlat = Nlat
        self.Nlon = Nlon

    def oned(self):
        """
        Create one dimensional arrays of latitude and longitude points.

        Returns
        -------
        lats : astropy.units.Quantity , shape=(Nlat,)
            Array of latitude points.
        lons : astropy.units.Quantity , shape=(Nlon,)
            Array of longitude points.

        """
        lats = np.linspace(-90, 90, self.Nlat)*u.deg
        lons = np.linspace(0, 360, self.Nlon)*u.deg
        return lats, lons

    def grid(self):
        """
        Create a 2 dimensional grid of latitude and longitude points.

        Returns
        -------
        lats : astropy.units.Quantity , shape=(Nlat,Nlon)
            Array of latitude points.
        lons : astropy.units.Quantity , shape=(Nlat,Nlon)
            Array of longitude points.

        """
        lats, lons = self.oned()
        return np.meshgrid(lats, lons)

    def zeros(self, dtype='float32'):
        """
        Get a grid of zeros.

        Parameters
        ----------
        dtype : str, default='float32
            Data type to pass to np.zeros.

        Returns
        -------
        arr : np.ndarray, shape=(Nlon, Nlat)
            Grid of zeros.

        """
        return np.zeros(shape=(self.Nlon, self.Nlat), dtype=dtype)

    def __eq__(self, other):
        """
        Check to see if two CoordinateGrid objects are equal.

        Parameters
        ----------
        other : CoordinateGrid
            Another CoordinateGrid object.

        Returns
        -------
        bool
            Whether the two objects have equal properties.
        
        Raises
        ------
        TypeError
            If `other` is not a CoordinateGrid object.

        """
        if not isinstance(other, CoordinateGrid):
            raise TypeError('other must be of type CoordinateGrid')
        else:
            return (self.Nlat == other.Nlat) & (self.Nlon == other.Nlon)

def round_teff(teff):
    """
    Round the effective temperature to the nearest integer.
    The goal is to reduce the number of unique effective temperatures
    while not affecting the accuracy of the model.

    Parameters
    ----------
    teff : astropy.units.Quantity 
        The temperature to round.

    Returns
    -------
    astropy.units.Quantity 
        The rounded temperature.
    """
    val = teff.value
    unit = teff.unit
    return int(round(val)) * unit


def get_angle_between(
    lat1:u.Quantity,
    lon1:u.Quantity,
    lat2:u.Quantity,
    lon2:u.Quantity
):
    """
    Get the angle between to lat/lon coordinates.
    """
    mu = (np.sin(lat1) * np.sin(lat2)
                 + np.cos(lat1) * np.cos(lat2)
                 * np.cos(lon1-lon2))
    return np.arccos(mu)

def proj_ortho(
    lat0:u.Quantity,
    lon0:u.Quantity,
    lats:u.Quantity,
    lons:u.Quantity
):
    """
    Use the spherical law of cosines to project each
    lat/lon point onto the x-y plane.

    Parameters
    ----------
    lat0 : astropy.units.quantity
        The latitude of the central point.
    lon0 : astropy.units.quantity
        The longitude of the central point.
    lats : astropy.units.quantity
        The latitude of the other points.
    lons : astropy.units.quantity
        The longitude of the other points.
    
    Returns
    -------
    x : np.ndarray
        The x coordinates of the points in the
        orthographic projection
    y : np.ndarray
        The y coordinates of the points in the
        orthographic projection
    
    Notes
    -----
    Spherical law of cosines:
    ..math:
        cos(c) = cos(a) cos(b) + sin(a) sin(b) cos(C)
    We want to find C.
    b is the colatitude of the central point.
    c is the colatitude of the other points.
    a is the distance of each point from the center (i.e. cos(mu))
    This can be difficult for special cases, but this code attempts
    to account for those.
    """
    # Cast all to radians
    lon0:float = lon0.to_value(u.rad)
    lat0:float = lat0.to_value(u.rad)
    lons:np.ndarray = lons.to_value(u.rad)
    lats:np.ndarray = lats.to_value(u.rad)
    b = np.pi/2 - lat0
    c = np.pi/2 - lats
    if b == 0: # centered on north pole
        a = c
        C = np.where(c <= np.pi/2,np.pi-lons,np.nan)
    elif b == np.pi: # centered on south pole
        a = c
        C = np.where(c >= np.pi/2,lons,np.nan)
    else: # centered elsewhere
        mu = (np.sin(lat0) * np.sin(lats)
                    + np.cos(lat0) * np.cos(lats)
                    * np.cos(lon0-lons))
        a = np.arccos(mu)
        cos_C = np.where(
            mu < 1,
            (np.cos(c) - np.cos(a)*np.cos(b)) / (np.sin(a)*np.sin(b)),
            1
        )
        cos_C = np.round(cos_C,4)
        C = np.where(
            lons==lon0,
            0
            ,np.arccos(cos_C)
        )
        C = np.where(
            (lons==lon0) & (lats<lat0),
            np.pi
            ,C
        )
        C = np.where(
            mu >= 0,
            C,
            np.nan
        )
        C = np.where(
            np.sin(lons-lon0)<0,
            -C,C
        )
    x = np.sin(C)*np.sin(a)
    y = np.cos(C)*np.sin(a)
    return x,y



def calc_circ_fraction_inside_unit_circle(x,y,r):
    """
    Calculate what fraction of a circle sits inside
    the unit circle. This is usefull for modeling
    secondary eclipse.

    Parameters
    ----------
    x : float
        The x coordinate of the circle's center.
    y : float
        The y coordinate of the circle's center.
    r : float
        The circle's radius.
    
    Returns
    -------
    float
        The fraction of the circle that lies inside the unit circle.
    
    Notes
    -----
    Math from [1].

    References
    ----------
    [1].  Weisstein, Eric W. "Circle-Circle Intersection."
        From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/Circle-CircleIntersection.html 
    """
    x0,y0,r0 = 0,0,1
    d = np.sqrt((x-x0)**2 + (y-y0)**2)
    if r>1:
        raise ValueError('R must be less than 1.')

    if d > (r0 + r):
        return 0.0
    if d <= (r0 - r):
        return 1.0
    if d==0 and r==1: # it is the unit circle
        return 1.0

    R = r0
    d1 = (d**2 - r**2 + R**2) / (2*d)
    d2 = (d**2 + r**2 - R**2) / (2*d)
    def area(_R,_d):
        return _R**2 * np.arccos(_d/_R) - _d * np.sqrt(_R**2 - _d**2)
    A1 = area(R,d1)
    A2 = area(r,d2)
    area_of_intersection = A1+A2

    area_of_circle = np.pi*r**2
    return area_of_intersection/area_of_circle
    
def get_planet_indicies(planet_times: u.Quantity, tindex: u.Quantity) -> tuple[int, int]:
    """
    Get the incicies of the planet spectra to interpolate over.
    This is a helper function that allows for interpolation of planet spectra.
    Since the planet changes over much longer timescales than the star (flares, etc),
    it makes sense to only run PSG once for multiple "integrations".

    Parameters
    ----------
    planet_times : astropy.units.Quantity [time]
        The times (cast to since periasteron) at which the planet spectrum was taken.
    tindex : astropy.units.Quantity [time]
        The epoch of the current observation. The goal is to place this between
        two elements of `planet_times`

    Returns
    -------
    int
        The index of `planet_times` before `tindex`
    int
        The index of `planet_times` after `tindex`

    Raises
    ------
    ValueError
        If multiple elements of 'planet_times' are equal to 'tindex'.
    """
    after = planet_times > tindex
    equal = planet_times == tindex
    if equal.sum() == 1:
        N1 = np.argwhere(equal)[0][0]
        N2 = np.argwhere(equal)[0][0]
    elif equal.sum() > 1:
        raise ValueError('There must be a duplicate time')
    elif equal.sum() == 0:
        N2 = np.argwhere(after)[0][0]
        N1 = N2 - 1
    return N1, N2

