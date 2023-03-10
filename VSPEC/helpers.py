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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0



