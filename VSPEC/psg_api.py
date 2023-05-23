"""VSPEC module to communicate with the PSG API

This module communucates between `VSPEC` and
and the Planetary Spectrum Generator via the API.
"""

from io import StringIO
import re
from pathlib import Path
import warnings
from astropy import units as u
import pandas as pd
import numpy as np
import requests

from VSPEC.read_info import ParamModel
from VSPEC.helpers import to_float, isclose

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
        with open(outfile, 'w', encoding='UTF-8') as file:
            file.write(reply.text)
        return None
    else:
        return reply.text


def parse_full_output(output_text:str):
    pattern = r'results_([\w]+).txt'
    split_text = re.split(pattern,output_text)
    names = split_text[1::2]
    content = split_text[2::2]
    data = {}
    for name,dat in zip(names,content):
        data[name] = dat
    return data



def get_static_psg_parameters(params: ParamModel)->dict:
    """
    Get the static (i.e. not phase or time dependent) parameters
    for PSG.

    Parameters
    ----------
    params : VSPEC.ParamModel
        The parameters to fill the config with.
    
    Returns
    -------
    config : dict
        The parameters translated into the PSG format.
    """
    bool_to_str = {True: 'Y', False: 'N'}
    config = {}
    config['OBJECT'] = 'Exoplanet'
    config['OBJECT-NAME'] = params.planet_name
    config['OBJECT-DIAMETER'] = to_float(2*params.planet_radius, u.km)
    config['OBJECT-GRAVITY'] = params.planet_grav
    config['OBJECT-GRAVITY-UNIT'] = params.planet_grav_mode
    config['OBJECT-STAR-TYPE'] = params.psg_star_template
    config['OBJECT-STAR-DISTANCE'] = to_float(params.planet_semimajor_axis, u.AU)
    config['OBJECT-PERIOD'] = to_float(params.planet_orbital_period, u.day)
    config['OBJECT-ECCENTRICITY'] = params.planet_eccentricity
    config['OBJECT-PERIAPSIS'] = to_float(params.system_phase_of_periasteron, u.deg)
    config['OBJECT-STAR-TEMPERATURE'] = to_float(params.star_teff, u.K)
    config['OBJECT-STAR-RADIUS'] = to_float(params.star_radius, u.R_sun)
    config['GEOMETRY'] = 'Observatory'
    config['GEOMETRY-OBS-ALTITUDE'] = to_float(params.system_distance, u.pc)
    config['GEOMETRY-ALTITUDE-UNIT'] = 'pc'
    config['GENERATOR-RANGE1'] = to_float(params.lambda_min, params.target_wavelength_unit)
    config['GENERATOR-RANGE2'] = to_float(params.lambda_max, params.target_wavelength_unit)
    config['GENERATOR-RANGEUNIT'] = params.target_wavelength_unit
    config['GENERATOR-RESOLUTION'] = params.resolving_power
    config['GENERATOR-RESOLUTIONUNIT'] = 'RP'
    config['GENERATOR-BEAM'] = params.beamValue
    config['GENERATOR-BEAM-UNIT'] = params.beamUnit
    config['GENERATOR-CONT-STELLAR'] = 'Y'
    config['OBJECT-INCLINATION'] = to_float(params.system_inclination, u.deg)
    config['OBJECT-SOLAR-LATITUDE'] = '0.0'
    config['OBJECT-OBS-LATITUDE'] = '0.0'
    config['GENERATOR-RADUNITS'] = params.psg_rad_unit
    config['GENERATOR-GCM-BINNING'] = params.gcm_binning
    config['GENERATOR-GAS-MODEL'] = bool_to_str[params.use_molec_signatures]
    config['GENERATOR-NOISE'] = params.detector_type
    config['GENERATOR-NOISEOTEMP'] = params.detector_temperature
    config['GENERATOR-NOISEOEFF'] = f"{params.detector_throughput:.1f}"
    config['GENERATOR-NOISEOEMIS'] = f"{params.detector_emissivity:.1f}"
    config['GENERATOR-NOISEFRAMES'] = params.detector_number_of_integrations
    config['GENERATOR-NOISEPIXELS'] = params.detector_pixel_sampling
    config['GENERATOR-NOISE1'] = params.detector_read_noise
    config['GENERATOR-DIAMTELE'] = f"{params.telescope_diameter:.1f}"
    config['GENERATOR-TELESCOPE'] = 'SINGLE'
    config['GENERATOR-TELESCOPE1'] = '1'
    config['GENERATOR-TELESCOPE2'] = '1.0'
    config['GENERATOR-TELESCOPE3'] = '1.0'
    return config

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


def write_static_config(path: Path, params: ParamModel, file_mode: str = 'w') -> None:
    """
    Write the initial PSG configuration file for a phase curve simulation.
    This occurs after the GCM is uploaded and before we start iterating through
    phase.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to write the config to.
    params : VSPEC.ParamModel
        The parameters to fill the config with.
    file_mode : str, default='r'
        Flag telling `open` which file method to use. In the case
        that GlobES is off, this should be ``'a'`` for append.
    """
    config = get_static_psg_parameters(params)
    content = cfg_to_bytes(config)
    with open(path,f'{file_mode}b') as file:
        file.write(content)

def change_psg_parameters(
    params:ParamModel,
    phase:u.Quantity,
    orbit_radius_coeff:float,
    sub_stellar_lon:u.Quantity,
    sub_stellar_lat:u.Quantity,
    pl_sub_obs_lon:u.Quantity,
    pl_sub_obs_lat:u.Quantity,
    include_star:bool
    ):
    """
    
    """
    config = {}
    config['OBJECT-STAR-TYPE'] = params.psg_star_template if include_star else '-'
    config['OBJECT-SEASON'] = f'{phase.to_value(u.deg):.4f}'
    config['OBJECT-STAR-DISTANCE'] = f'{(orbit_radius_coeff*params.planet_semimajor_axis).to_value(u.AU):.4f}'
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
        for i, item in enumerate(raw_header):
            if 'WARNING' in item:
                warning, kind, message = item.split('|')
                header['warnings'].append(dict(kind=kind, message=message))
            elif 'ERROR' in item:
                error, kind, message = item.split('|')
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
    if not np.all(isclose(cmb_rad.data['Wave/freq'], therm_rad.data['Wave/freq'], 1e-3*u.um)):
        raise ValueError('The spectral axes must be equivalent.')

    if 'Reflected' in cmb_rad.data.keys():
        return cmb_rad.data['Reflected']
    elif 'Reflected' in therm_rad.data.keys():
        return therm_rad.data['Reflected']
    elif (planet_name in cmb_rad.data.keys()) and (planet_name in therm_rad.data.keys()):
        return cmb_rad.data[planet_name] - therm_rad.data[planet_name]
    else:
        raise KeyError(f'Data array {planet_name} not found.')
