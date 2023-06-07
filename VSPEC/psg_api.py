"""VSPEC module to communicate with the PSG API

This module communucates between `VSPEC` and
and the Planetary Spectrum Generator via the API.
"""

from io import StringIO
import re
import warnings
from astropy import units as u
import pandas as pd
import numpy as np
import requests

from VSPEC.params.read import Parameters

warnings.simplefilter('ignore', category=u.UnitsWarning)


def call_api(config_path: str = None, psg_url: str = 'https://psg.gsfc.nasa.gov',
             api_key: str = None, output_type: str = None, app: str = None,
             outfile: str = None, config_data: str = None) -> None:
    """
    Call the PSG api

    Build and execute an API query to communicate with PSG.

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
    outfile : str, default=None
        The path to write the PSG output.
    config_data : str, default=None
        The data contained by a config file. Essentially removes the need
        to write a config to file.

    Raises
    ------
    ValueError
        If `config_path` and `config_data` are both `None`
    """
    data = {}
    if config_path is not None:
        with open(config_path, 'rb') as file:
            dat = file.read()
        data['file'] = dat
    else:
        if config_data is None:
            raise ValueError(
                'A config file or the files contents must be specified '
                'using the `config_path` or `config_data` parameters'
            )
        else:
            data['file'] = config_data
    if api_key is not None:
        data['key'] = api_key
    if app is not None:
        data['app'] = app
    if output_type is not None:
        data['type'] = output_type
    url = f'{psg_url}/api.php'
    reply = requests.post(url, data=data, timeout=120)
    if outfile is not None:
        with open(outfile, 'wb') as file:
            file.write(reply.content)
        return None
    else:
        return reply.content


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
    params:Parameters,
    phase:u.Quantity,
    orbit_radius_coeff:float,
    sub_stellar_lon:u.Quantity,
    sub_stellar_lat:u.Quantity,
    pl_sub_obs_lon:u.Quantity,
    pl_sub_obs_lat:u.Quantity,
    include_star:bool
    )->dict:
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
    config = {}
    config['OBJECT-STAR-TYPE'] = params.star.template if include_star else '-'
    config['OBJECT-SEASON'] = f'{phase.to_value(u.deg):.4f}'
    config['OBJECT-STAR-DISTANCE'] = f'{(orbit_radius_coeff*params.planet.semimajor_axis).to_value(u.AU):.4f}'
    config['OBJECT-SOLAR-LONGITUDE'] = f'{sub_stellar_lon.to_value(u.deg)}'
    config['OBJECT-SOLAR-LATITUDE'] = f'{sub_stellar_lat.to_value(u.deg)}'
    config['OBJECT-OBS-LONGITUDE'] = f'{pl_sub_obs_lon.to_value(u.deg)}'
    config['OBJECT-OBS-LATITUDE'] = f'{pl_sub_obs_lat.to_value(u.deg)}'
    return config


class PSGrad:
    """
    Container for PSG rad files

    Parameters
    ----------
    header : dict
        Dictionary containing header information. This includes the date, any warnings,
        and the units of the rad file.
    data : dict
        Dictionary containing the spectral data. They keys are the column names and the
        values are astropy.units.Quantity arrays.

    Attributes
    ----------
    header : dict
        Dictionary containing header information. This includes the date, any warnings,
        and the units of the rad file.
    data : dict
        Dictionary containing the spectral data. They keys are the column names and the
        values are astropy.units.Quantity arrays.
    """

    def __init__(self, header, data):
        self.header = header
        self.data = data

    @classmethod
    def from_rad(cls, filename):
        """
        Create a `PSGrad` object from a file. This is designed to load in
        the raw `.rad` output from PSG
        """
        raw_header = []
        raw_data = []
        with open(filename, 'r', encoding='UTF-8') as file:
            for line in file:
                if line[0] == '#':
                    raw_header.append(line.replace('\n', ''))
                else:
                    raw_data.append(line.replace('\n', ''))
        header = {
            'warnings': [],
            'errors': [],
            'binning': -1,
            'author': '',
            'date': '',
            'velocities': {},
            'spectral_unit': u.dimensionless_unscaled,
            'radiance_unit': u.dimensionless_unscaled,

        }
        for _, item in enumerate(raw_header):
            if 'WARNING' in item:
                _, kind, message = item.split('|')
                header['warnings'].append(dict(kind=kind, message=message))
            elif 'ERROR' in item:
                _, kind, message = item.split('|')
                header['errors'].append(dict(kind=kind, message=message))
            elif '3D spectroscopic simulation' in item:
                header['binning'] = int(re.findall(r'of ([\d]+) \(', item)[0])
            elif 'Planetary Spectrum Generator' in item:
                header['author'] = item[1:].strip()
            elif 'Synthesized' in item:
                header['date'] = item[1:].strip()
            elif 'Doppler velocities' in item:
                unit = u.Unit(re.findall(r'\[([\w\d/]+)\]', item)[0])
                keys = re.findall(r'\(([\w\d, \+]+)\)', item)[0].split(',')
                values = item.split(':')[1].split(',')
                for key, value in zip(keys, values):
                    header['velocities'][key] = float(value)*unit
            elif 'Spectral unit' in item:
                header['spectral_unit'] = u.Unit(
                    re.findall(r'\[([\w\d/]+)\]', item)[0])
            elif 'Radiance unit' in item:
                header['radiance_unit'] = u.Unit(
                    re.findall(r'\[([\w\d/]+)\]', item)[0])
        columns = raw_header[-1][1:].strip().split()
        dat = StringIO('\n'.join(raw_data))
        df = pd.read_csv(dat, names=columns, delim_whitespace=True)
        if len(df) == 0:
            raise ValueError(
                'It looks like there might not be any data in this rad file.')
        data = {}
        if not columns[0] == 'Wave/freq':
            raise ValueError('.rad format is incorrect')
        data[columns[0]] = df[columns[0]].values * header['spectral_unit']
        for col in columns[1:]:
            data[col] = df[col].values * header['radiance_unit']
        return cls(header, data)


def get_reflected(cmb_rad: PSGrad, therm_rad: PSGrad, planet_name: str) -> u.Quantity:
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
    axis_equal = np.isclose(
        cmb_rad.data['Wave/freq'].to_value(u.um),
        therm_rad.data['Wave/freq'].to_value(u.um),
        atol=1e-3
    )
    if not axis_equal:
        raise ValueError('The spectral axes must be equivalent.')

    if 'Reflected' in cmb_rad.data.keys():
        return cmb_rad.data['Reflected']
    elif 'Reflected' in therm_rad.data.keys():
        return therm_rad.data['Reflected']
    elif (planet_name in cmb_rad.data.keys()) and (planet_name in therm_rad.data.keys()):
        try:
            return cmb_rad.data[planet_name] - therm_rad.data[planet_name] - cmb_rad.data['Transit']
        except KeyError:
            return cmb_rad.data[planet_name] - therm_rad.data[planet_name]
    else:
        raise KeyError(f'Data array {planet_name} not found.')
