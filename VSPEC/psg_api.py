"""VSPEC module to communicate with the PSG API

This module communucates between `VSPEC` and
and the Planetary Spectrum Generator via the API.
"""

import re
import warnings
from astropy import units as u
import numpy as np
import requests
from typing import Union

import pypsg

from VSPEC.params.read import InternalParameters

warnings.simplefilter('ignore', category=u.UnitsWarning)


def call_api(
    psg_url: str = 'https://psg.gsfc.nasa.gov',
    api_key: str = None,
    output_type: str = None,
    app: str = None,
    config_data: str = None
)->bytes:
    """
    Call the PSG API.

    Parameters
    ----------
    psg_url : str, default='https://psg.gsfc.nasa.gov'
        The URL of the `PSG` API. Use 'http://localhost:3000' if running locally.
    api_key : str, default=None
        The key for the public API. Needed only if not runnning `PSG` locally.
    output_type : str, default=None
        The type of output to retrieve from `PSG`. Options include 'cfg', 'rad',
        'noi', 'lyr', 'all'.
    app : str, default=None
        The PSG app to call. For example: 'globes'
    config_data : str, default=None
        The data contained by a config file. Essentially removes the need
        to write a config to file.

    Returns
    -------
    bytes
        The content of the response from PSG.
    """
    data = {}
    data['file'] = config_data
    if api_key is not None:
        data['key'] = api_key
    if app is not None:
        data['app'] = app
    if output_type is not None:
        data['type'] = output_type
    url = f'{psg_url}/api.php'
    reply = requests.post(url, data=data, timeout=120)
    return reply.content


def call_api_from_file(config_path: str = None, psg_url: str = 'https://psg.gsfc.nasa.gov',
             api_key: str = None, output_type: str = None, app: str = None) -> Union[None,bytes]:
    """
    Call the PSG api by first reading data from a file.

    Parameters
    ----------
    config_path : str or pathlib.Path, default=None
        The path to the `PSG` config file.
    psg_url : str, default='https://psg.gsfc.nasa.gov'
        The URL of the `PSG` API. Use 'http://localhost:3000' if running locally.
    api_key : str, default=None
        The key for the public API. Needed only if not runnning `PSG` locally.
    output_type : str, default=None
        The type of output to retrieve from `PSG`. Options include 'cfg', 'rad',
        'noi', 'lyr', 'all'.
    app : str, default=None
        The PSG app to call. For example: 'globes'

    
    Returns
    -------
    bytes
       The content of the response.
    """
    with open(config_path, 'rb') as file:
        dat = file.read()
    
    content = call_api(
        psg_url=psg_url,
        api_key=api_key,
        output_type=output_type,
        app=app,
        config_data=dat
    )
    return content


def parse_full_output(output_text:bytes):
    """
    Parse PSG full output.

    Parameters
    ----------
    output_text : bytes
        The output of a PSG 'all' call.

    Returns
    -------
    dict
        The parsed, separated output files.
    """
    pattern = rb'results_([\w]+).txt'
    split_text = re.split(pattern,output_text)
    names = split_text[1::2]
    content = split_text[2::2]
    data = {}
    for name,dat in zip(names,content):
        data[name] = dat.strip()
    return data

def cfg_to_bytes(config:dict)->bytes:
    """
    Convert a PSG config dictionary into a bytes sequence.

    Parameters
    ----------
    config : dict
        The dictionary containing PSG parameters
    
    Returns
    -------
    bytes
        A bytes object containing the file content.
    """
    s = b''
    for key,value in config.items():
        s += bytes(f'<{key}>{value}\n',encoding='UTF-8')
    return s

def cfg_to_dict(config:str)->dict:
    """
    Convert a PSG config file into a dictionary.
    """
    cfg = {}
    for line in config.split('\n'):
        key,value = line.replace('<','').split('>')
        cfg.update({key:value})
    return cfg

def change_psg_parameters(
    params:InternalParameters,
    phase:u.Quantity,
    orbit_radius_coeff:float,
    sub_stellar_lon:u.Quantity,
    sub_stellar_lat:u.Quantity,
    pl_sub_obs_lon:u.Quantity,
    pl_sub_obs_lat:u.Quantity,
    include_star:bool
    )->pypsg.PyConfig:
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
    target = pypsg.cfg.Target(
        star_type=params.star.psg_star_template if include_star else '-',
        season=phase,
        star_distance=orbit_radius_coeff*params.planet.semimajor_axis,
        solar_longitude=sub_stellar_lon,
        solar_latitude=sub_stellar_lat,
        obs_longitude=pl_sub_obs_lon,
        obs_latitude=pl_sub_obs_lat
    )
    return pypsg.PyConfig(target=target)



def get_reflected(
    cmb_rad: pypsg.PyRad,
    therm_rad: pypsg.PyRad,
    planet_name: str
    ) -> u.Quantity:
    """
    Get reflected spectra.

    Parameters
    ----------
    cmb_rad : PSGrad
        A rad file from the star+planet PSG call.
    therm_rad : PSGrad
        A rad file from the planet-only PSG call.

    Returns
    -------
    astropy.units.Quantity
        The spectrum of the reflected light.

    Raises
    ------
    ValueError
        If the wavelength axes do not match.
    KeyError
        If neither object has a `'Reflected'` data array and at least one
        of them is missing the `planet_name` data array.
    """
    axis_equal = np.all(np.isclose(
        cmb_rad.wl.to_value(u.um),
        therm_rad.wl.to_value(u.um),
        atol=1e-3
    ))
    if not axis_equal:
        raise ValueError('The spectral axes must be equivalent.')
    planet_name = planet_name.replace(' ', '-')

    
    if 'Reflected' in cmb_rad.colnames:
        return cmb_rad['Reflected']
    elif 'Reflected' in therm_rad.colnames:
        return therm_rad['Reflected']
    elif (planet_name in cmb_rad.colnames) and (planet_name in therm_rad.colnames):
        if 'Transit' in cmb_rad.colnames:
            return cmb_rad[planet_name] * 0 # assume there is no refection during transit
        else:
            return cmb_rad[planet_name] - therm_rad[planet_name]
    else:
        raise KeyError(f'Data array {planet_name} not found.')
