"""
Misc helpers
"""

from astropy import units as u
import numpy as np


def get_transit_radius(
    system_distance: u.Quantity,
    stellar_radius: u.Quantity,
    semimajor_axis: u.Quantity,
    planet_radius: u.Quantity
) -> u.Quantity:
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


def get_planet_indicies(
    planet_times: u.Quantity,
    tindex: u.Quantity
    ) -> tuple[int, int]:
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