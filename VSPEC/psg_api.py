"""VSPEC module to communicate with the PSG API

This module communucates between `VSPEC` and
and the Planetary Spectrum Generator via the API.
"""

from astropy import units as u

import libpypsg

from VSPEC.params.read import InternalParameters

def change_psg_parameters(
    params:InternalParameters,
    phase:u.Quantity,
    orbit_radius_coeff:float,
    sub_stellar_lon:u.Quantity,
    sub_stellar_lat:u.Quantity,
    pl_sub_obs_lon:u.Quantity,
    pl_sub_obs_lat:u.Quantity,
    include_star:bool
    )->libpypsg.PyConfig:
    """
    Get the time-dependent PSG parameters

    Parameters
    ----------
    params : VSPEC.params.Parameters
        The parameters of this VSPEC simulation
    phase : astropy.units.Quantity
        The phase of the planet
    orbit_radius_coeff : float
        The planet-star distance normalized to the semimajor axis.
    sub_stellar_lon : astropy.units.Quantity
        The sub-stellar longitude of the planet.
    sub_stellar_lat : astropy.units.Quantity
        The sub-stellar latitude of the planet.
    pl_sub_obs_lon : astropy.units.Quantity
        The sub-observer longitude of the planet.
    pl_sub_obs_lat : astropy.units.Quantity
        The sub-observer latitude of the planet.
    include_star : bool
        If True, include the star in the simulation.
    
    Returns
    -------
    config : dict
        The PSG config in dictionary form.
    """
    target = libpypsg.cfg.Target(
        star_type=params.star.psg_star_template if include_star else '-',
        season=phase,
        star_distance=orbit_radius_coeff*params.planet.semimajor_axis,
        solar_longitude=sub_stellar_lon,
        solar_latitude=sub_stellar_lat,
        obs_longitude=pl_sub_obs_lon,
        obs_latitude=pl_sub_obs_lat
    )
    return libpypsg.PyConfig(target=target)