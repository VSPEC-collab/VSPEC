"""VSPEC helpers module

This module contains functions that may be shared
throughout the rest of the package, especially
pertainting to safe-casting of astropy.units
objects.
"""

from astropy import units as u
from numpy import isclose as np_isclose
import numpy as np

def to_float(quant,unit):
    """
    to float

    Cast a quantity to a float given a unit

    Args:
        quant (Quantity): Quantity to be cast
        unit (Unit): Unit to be cast with
    Returns:
        (float): Cast quantity
    """
    return (quant/unit).to(u.Unit('')).value

def isclose(a:u.Quantity,b:u.Quantity,tol:u.Quantity)->bool:
    """
    Check if two quantities are close

    Use numpy.isclose on two quantity objects. This function safely casts them to floats first.
    Args:
        a (Quantity): array to be compared
        b (Quantity): array to be compared
        tol (Quantity): tolerance
    """
    unit = tol.unit
    return np_isclose(to_float(a,unit),to_float(b,unit),rtol=to_float(tol,unit))

def get_transit_radius(
        system_distance:u.Quantity[u.pc],
        stellar_radius:u.Quantity[u.R_sun],
        semimajor_axis:u.Quantity[u.AU],
        planet_radius:u.Quantity[u.R_earth]
    )->u.Quantity[u.rad]:
    """
    Calculate the radius from mid-transit where there is some
    overlap between the planetary and stellar disk

    This is an approximation that asumes a circular orbit and i=90 deg

    Returns:
        (u.Quantity[u.deg]): angle from mid-transit where overlap begins
    """
    R_a = to_float(stellar_radius/semimajor_axis,u.Unit(''))
    R_D = to_float(stellar_radius/system_distance,u.Unit(''))
    angle_point_planet = np.arcsin(R_a*np.cos(R_D)) - R_D # float in radians
    planet_radius_angle = to_float(planet_radius/(2*np.pi*semimajor_axis),u.Unit(''))
    return (angle_point_planet+planet_radius_angle)*u.rad
