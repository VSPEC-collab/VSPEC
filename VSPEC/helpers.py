"""VSPEC helpers module

This module contains functions that may be shared
throughout the rest of the package, especially
pertainting to safe-casting of astropy.units
objects.
"""
from astropy import units as u
from numpy import isclose as np_isclose
from io import StringIO
import numpy as np
import pandas as pd
import socket
from os import system

from VSPEC.config import PSG_PORT


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

    Examples
    --------
    >>> from astropy import units as u
    >>> import numpy as np

    >>> values1 = np.array([1.0, 2.0, 3.0]) * u.m
    >>> values2 = np.array([1.01, 2.02, 3.03]) * u.m
    >>> tol = 0.05 * u.m
    >>> isclose(values1, values2, tol)
    array([ True,  True, False])

    >>> temperatures1 = np.array([25.0, 30.0, 35.0]) * u.K
    >>> temperatures2 = np.array([25.5, 30.2, 34.8]) * u.K
    >>> tol = 0.3 * u.K
    >>> isclose(temperatures1, temperatures2, tol)
    array([ True,  True,  True])

    """
    unit = tol.unit
    return np_isclose(quant1.to_value(unit), quant2.to_value(unit), atol=tol.to_value(unit))


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


    .. deprecated:: 0.1
        This function is no longer used.

    .. warning::
        This math is not validated.
    """
    radius_over_semimajor_axis = (
        stellar_radius/semimajor_axis).to_value(u.dimensionless_unscaled)
    radius_over_distance = (
        stellar_radius/system_distance).to_value(u.dimensionless_unscaled)
    angle_point_planet = np.arcsin(radius_over_semimajor_axis*np.cos(
        radius_over_distance)) - radius_over_distance  # float in radians
    planet_radius_angle = (
        planet_radius/(2*np.pi*semimajor_axis)).to_value(u.dimensionless_unscaled)
    return (angle_point_planet+planet_radius_angle)*u.rad


def is_port_in_use(port: int) -> bool:
    """
    Check if a port is in use on your machine.

    This function is useful to determine if a specific port is already being used
    by another process, such as PSG (Planetary Spectrum Generator).
    It attempts to establish a connection to the specified port on the local machine.
    If the connection is successful (return value of 0), it means that something is
    already running on that port.

    Parameters
    ----------
    port : int
        The port number to check. Typically, PSG runs on port 3000.

    Returns
    -------
    bool
        Returns True if a process is already running on the specified port, and False otherwise.

    Notes
    -----
    - If you call this function immediately after changing the Docker image state,
        you may get an incorrect answer due to timing issues. It is recommended to use
        this function within a function that incorporates a timeout mechanism.
    - This function relies on the `socket` module from the Python standard library.

    Examples
    --------
    >>> is_port_in_use(3000)
    True

    >>> is_port_in_use(8080)
    False

    """
    socket_obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    return socket_obj.connect_ex(('localhost', port)) == 0


def set_psg_state(running: bool):
    """
    Set the local PSG (Planetary Spectrum Generator) state.

    This function allows you to control the state of the local
    PSG Docker container. By specifying whether the PSG should be running
    or not, you can start or stop the PSG container accordingly.

    Parameters
    ----------
    running : bool
        A boolean value indicating whether the PSG should be running.
        - If `running` is True and the PSG is not already running, the function will start the PSG container.
        - If `running` is False and the PSG is currently running, the function will stop the PSG container.

    Notes
    -----
    - This function relies on the `system` function from the `os` module to execute Docker commands.
    - The `is_port_in_use` function from the `VSPEC.helpers` module is used to check if the PSG port is in use.

    Examples
    --------
    >>> set_psg_state(True)  # Start the PSG container if not already running

    >>> set_psg_state(False)  # Stop the PSG container if currently running
    """
    if is_port_in_use(PSG_PORT) and not running:
        system('docker stop psg')
    elif not is_port_in_use(PSG_PORT) and running:
        system('docker start psg')


def arrange_teff(minteff: u.Quantity, maxteff: u.Quantity):
    """
    Generate a list of effective temperature (Teff) values with steps of 100 K that fully
    encompass the specified range.

    This function is useful for obtaining a list of Teff values to be used for binning spectra later on.

    Parameters
    ----------
    minteff : astropy.units.Quantity
        The minimum Teff value required.
    maxteff : astropy.units.Quantity
        The maximum Teff value required.

    Returns
    -------
    teffs : astropy.units.Quantity
        An array of Teff values, with steps of 100 K.

    Notes
    -----
    - The function calculates the Teff values that fully encompass the specified range by rounding down the minimum value to the nearest multiple of 100 K and rounding up the maximum value to the nearest multiple of 100 K.
    - The `np.arange` function is then used to generate a sequence of Teff values with steps of 100 K, covering the entire range from the rounded-down minimum value to the rounded-up maximum value.

    Examples
    --------
    >>> minteff = 5000 * u.K
    >>> maxteff = 6000 * u.K
    >>> arrange_teff(minteff, maxteff)
    <Quantity [5000., 5100., 5200., 5300., 5400., 5500., 5600., 5700., 5800., 5900., 6000.] K>
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
    number_of_steps = ((high-low)/step).to_value(u.dimensionless_unscaled)
    number_of_steps = int(round(number_of_steps))
    teffs = low + np.arange(number_of_steps+1)*step
    return teffs


def get_surrounding_teffs(Teff: u.Quantity):
    """
    Get the effective temperatures (Teffs) of the two spectra to interpolate between
    in order to obtain a spectrum with the target Teff.

    This function is useful for determining the Teffs of the two spectra that surround
    a given target Teff value, which are necessary for performing interpolation to
    obtain a spectrum with the desired Teff.

    Parameters
    ----------
    Teff : astropy.units.Quantity
        The target Teff for the interpolated spectrum.

    Returns
    -------
    low_teff : astropy.units.Quantity
        The Teff of the spectrum below the target Teff.
    high_teff : astropy.units.Quantity
        The Teff of the spectrum above the target Teff.

    Raises
    ------
    ValueError
        If the target Teff is a multiple of 100 K, which would cause problems with ``scipy`` interpolation.

    Notes
    -----
    - The function checks if the target Teff is a multiple of 100 K. If it is, a `ValueError` is raised because this would lead to issues with scipy interpolation.
    - If the target Teff is not a multiple of 100 K, the function determines the Teff of the spectrum
        below the target Teff by rounding down to the nearest multiple of 100 K, and the Teff of the
        spectrum above the target Teff is obtained by adding 100 K to the low Teff.

    Examples
    --------
    >>> Teff = 5500 * u.K
    >>> get_surrounding_teffs(Teff)
    (<Quantity 5500. K>, <Quantity 5600. K>)
    """

    step = 100*u.K
    if (Teff % step) == 0*u.K:
        raise ValueError(
            f'Teff of {Teff} is a multiple of {100*u.K}. This will cause problems with scipy.')
    else:
        low_teff = Teff - (Teff % step)
        high_teff = low_teff + step
    return low_teff, high_teff


def plan_to_df(observation_plan: dict) -> pd.DataFrame:
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

    This class provides a convenient way to create latitude and longitude grids of specified dimensions. It allows the creation of both one-dimensional arrays and two-dimensional grids of latitude and longitude points.

    Parameters
    ----------
    Nlat : int, optional (default=500)
        Number of latitude points.
    Nlon : int, optional (default=1000)
        Number of longitude points.

    Raises
    ------
    TypeError
        If Nlat or Nlon is not an integer.


    Attributes
    ----------
    Nlat : int
        Number of latitude points.
    Nlon : int
        Number of longitude points.

    Examples
    --------
    >>> grid = CoordinateGrid(Nlat=100, Nlon=200)
    >>> lats, lons = grid.oned()
    >>> print(lats.shape, lons.shape)
    (100,) (200,)
    >>> grid_arr = grid.grid()
    >>> print(grid_arr.shape)
    (100, 200)
    >>> zeros_arr = grid.zeros()
    >>> print(zeros_arr.shape)
    (200, 100)
    >>> other_grid = CoordinateGrid(Nlat=100, Nlon=200)
    >>> print(grid == other_grid)
    True

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
        dtype : str, default='float32'
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

    Notes
    -----
    This function rounds the given effective temperature to the nearest integer value. It is designed to decrease the number of unique effective temperatures while maintaining the accuracy of the model.

    Examples
    --------
    >>> teff = 1234.56 * u.K
    >>> rounded_teff = round_teff(teff)
    >>> print(rounded_teff)
    1235 K

    >>> teff = 2000.4 * u.K
    >>> rounded_teff = round_teff(teff)
    >>> print(rounded_teff)
    2000 K

    """

    val = teff.value
    unit = teff.unit
    return int(round(val)) * unit


def get_angle_between(
    lat1: u.Quantity,
    lon1: u.Quantity,
    lat2: u.Quantity,
    lon2: u.Quantity
):
    """
    Compute the angle between two coordintates in
    lat/lon space.

    Parameters
    ----------
    lat1 : astropy.units.Quantity
        The latitude of the first coordinate
    lon1 : astropy.units.Quantity
        The longitude of the first coordinate
    lat2 : astropy.units.Quantity
        The latitude of the second coordinate
    lon2 : astropy.units.Quantity
        The longitude of the second coordinate

    Returns
    -------
    alpha : astropy.units.Quantity
        The angle between the coordinates.

    Notes
    -----
    This function computes the angle between two coordinates specified in latitude and longitude space using the spherical law of cosines:

    .. math::
        \\mu = \\sin{\\phi_1} \\sin{\\phi_2} + \\cos{\\phi_1} \\cos{\\phi_2} \\cos{(\\lambda_1 - \\lambda_2)}

    where :math:`(\\phi_1, \\lambda_1)` and :math:`(\\phi_2, \\lambda_2)` are the latitude and longitude pairs of the two coordinates, respectively. The angle between the coordinates is then computed as:

    .. math::
        \\alpha = \\arccos{\\mu}

    The computed angle is returned as an `astropy.units.Quantity` object.

    Examples
    --------
    >>> lat1 = 30 * u.deg
    >>> lon1 = 45 * u.deg
    >>> lat2 = 40 * u.deg
    >>> lon2 = 60 * u.deg
    >>> angle = get_angle_between(lat1, lon1, lat2, lon2)
    >>> print(angle)
    0.27581902582503454 rad

    """
    mu = (np.sin(lat1) * np.sin(lat2)
          + np.cos(lat1) * np.cos(lat2)
          * np.cos(lon1-lon2))
    return np.arccos(mu)


def proj_ortho(
    lat0: u.Quantity,
    lon0: u.Quantity,
    lats: u.Quantity,
    lons: u.Quantity
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
    This function uses the spherical law of cosines to project each lat/lon point
    onto the x-y plane of an orthographic projection.

    Spherical law of cosines:

    .. math::

        \\cos{c} = \\cos{a} \\cos{b} + \\sin{a} \\sin{b} \\cos{C}

    We want to find :math:`C`.
        - :math:`b` is the colatitude of the central point.
        - :math:`c` is the colatitude of the other points.
        - :math:`a` is the distance of each point from the center (i.e. :math:`\\cos{\\mu}`)
    This can be difficult for special cases, but this code attempts
    to account for those.

    Examples
    --------
    >>> lat0 = 30 * u.deg
    >>> lon0 = 45 * u.deg
    >>> lats = [40, 50, 60] * u.deg
    >>> lons = [60, 70, 80] * u.deg
    >>> x, y = proj_ortho(lat0, lon0, lats, lons)
    >>> print(x)
    [0.19825408 0.27164737 0.28682206]
    >>> print(y)
    [0.18671294 0.37213692 0.54519419]

    """
    # Cast all to radians
    lon0: float = lon0.to_value(u.rad)
    lat0: float = lat0.to_value(u.rad)
    lons: np.ndarray = lons.to_value(u.rad)
    lats: np.ndarray = lats.to_value(u.rad)
    b = np.pi/2 - lat0
    c = np.pi/2 - lats
    if b == 0:  # centered on north pole
        a = c
        C = np.where(c <= np.pi/2, np.pi-lons, np.nan)
    elif b == np.pi:  # centered on south pole
        a = c
        C = np.where(c >= np.pi/2, lons, np.nan)
    else:  # centered elsewhere
        mu = (np.sin(lat0) * np.sin(lats)
              + np.cos(lat0) * np.cos(lats)
              * np.cos(lon0-lons))
        a = np.arccos(mu)
        cos_C = np.where(
            mu < 1,
            (np.cos(c) - np.cos(a)*np.cos(b)) / (np.sin(a)*np.sin(b)),
            1
        )
        cos_C = np.round(cos_C, 4)
        C = np.where(
            lons == lon0,
            0, np.arccos(cos_C)
        )
        C = np.where(
            (lons == lon0) & (lats < lat0),
            np.pi, C
        )
        C = np.where(
            mu >= 0,
            C,
            np.nan
        )
        C = np.where(
            np.sin(lons-lon0) < 0,
            -C, C
        )
    x = np.sin(C)*np.sin(a)
    y = np.cos(C)*np.sin(a)
    return x, y


def calc_circ_fraction_inside_unit_circle(x, y, r):
    """
    Calculate what fraction of a circle sits inside
    the unit circle.

    This is useful for modeling secondary eclipse.

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

    Raises
    ------
    ValueError
        If the circle's radius exceeds 1.


    Notes
    -----
    The calculation is based on the circle-circle intersection math presented in [1].

    References
    ----------
    [1].  Weisstein, Eric W. "Circle-Circle Intersection."
        From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/Circle-CircleIntersection.html 
    """
    x0, y0, r0 = 0, 0, 1
    d = np.sqrt((x-x0)**2 + (y-y0)**2)
    if r > 1:
        raise ValueError('R must be less than 1.')

    if d > (r0 + r):
        return 0.0
    if d <= (r0 - r):
        return 1.0
    if d == 0 and r == 1:  # it is the unit circle
        return 1.0

    R = r0
    d1 = (d**2 - r**2 + R**2) / (2*d)
    d2 = (d**2 + r**2 - R**2) / (2*d)

    def area(_R, _d):
        return _R**2 * np.arccos(_d/_R) - _d * np.sqrt(_R**2 - _d**2)
    A1 = area(R, d1)
    A2 = area(r, d2)
    area_of_intersection = A1+A2

    area_of_circle = np.pi*r**2
    return area_of_intersection/area_of_circle


def get_planet_indicies(planet_times: u.Quantity, tindex: u.Quantity) -> tuple[int, int]:
    """
    Get the indices of the planet spectra to interpolate over.

    This helper function enables interpolation of planet spectra by determining
    the appropriate indices in the `planet_times` array. By running PSG once for
    multiple "integrations" and interpolating between the spectra, computational
    efficiency is improved.


    Parameters
    ----------
    planet_times : astropy.units.Quantity
        The times (cast to since periasteron) at which the planet spectrum was taken.
    tindex : astropy.units.Quantity
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


def read_lyr(filename: str) -> pd.DataFrame:
    """
    Read a PSG layer file and convert it to a pandas DataFrame.

    This function parses a PSG ``.lyr`` file and transforms it into a pandas DataFrame,
    making it easier to work with the layer data.

    Parameters
    ----------
    filename : str
        The name of the layer file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the layer data.
    """
    lines = []
    with open(filename, 'r', encoding='UTF-8') as file:
        save = False
        for line in file:
            if 'Alt[km]' in line:
                save = True
            if save:
                if '--' in line:
                    if len(lines) > 2:
                        save = False
                    else:
                        pass
                else:
                    lines.append(line[2:-1])
    if len(lines) == 0:
        raise ValueError('No data was captured. Perhaps the format is wrong.')
    dat = StringIO('\n'.join(lines[1:]))
    names = lines[0].split()
    for i, name in enumerate(names):
        # get previous parameter (e.g 'water' for 'water_size')
        if 'size' in name:
            names[i] = names[i-1] + '_' + name
    return pd.read_csv(dat, delim_whitespace=True, names=names)
