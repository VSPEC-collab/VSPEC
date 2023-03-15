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
        The keys are {'time','phase','sub_obs_lat','sub_obs_lon',
                        'sub_planet_lat','sub_planet_lon','sub_stellar_lon',
                        'sub_stellar_lat','planet_sub_obs_lon','planet_sub_obs_lat',
                        'orbit_radius'
                     }
    
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