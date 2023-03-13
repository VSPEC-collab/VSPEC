"""VSPEC module to communicate with the PSG API

This module communucates between `VSPEC` and
and the Planetary Spectrum Generator via the API.
"""

import os
from io import StringIO
import re
from pathlib import Path
from astropy import units as u
import pandas as pd
import numpy as np

from VSPEC.read_info import ParamModel
from VSPEC.helpers import to_float, isclose


def call_api(config_path: str, psg_url: str = 'https://psg.gsfc.nasa.gov',
             api_key: str = None, output_type: str = None, app: str = None,
             outfile: str = None, verbose: bool = False) -> None:
    """
    Call the PSG api

    Build and execute an API query to communicate with PSG.

    Parameters
    ----------
    config_path : str or pathlib.Path
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
    """
    if verbose:
        cmd = 'curl'
    else:
        cmd = 'curl -s'
    if api_key is not None:
        cmd = cmd + f' -d key={api_key}'
    if app is not None:
        cmd = cmd + f' -d app={app}'
    if output_type is not None:
        cmd = cmd + f' -d type={output_type}'
    cmd = cmd + f' --data-urlencode file@{config_path}'
    cmd = cmd + f' {psg_url}/api.php'
    if outfile is not None:
        cmd = cmd + f' > {outfile}'
    if verbose:
        print(cmd)
    os.system(cmd)

def write_static_config(path:Path,params:ParamModel,file_mode:str='w')->None:
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
    with open(path, file_mode, encoding="ascii") as file:
        bool_to_str = {True: 'Y', False: 'N'}
        file.write('<OBJECT>Exoplanet\n')
        file.write(f'<OBJECT-NAME>{params.planet_name}\n')
        file.write('<OBJECT-DIAMETER>%f\n' %
                    to_float(2*params.planet_radius, u.km))
        file.write('<OBJECT-GRAVITY>%f\n' % params.planet_grav)
        file.write(f'<OBJECT-GRAVITY-UNIT>{params.planet_grav_mode}\n')
        file.write('<OBJECT-STAR-TYPE>%s\n' % params.psg_star_template)
        file.write('<OBJECT-STAR-DISTANCE>%f\n' %
                    to_float(params.planet_semimajor_axis, u.AU))
        file.write('<OBJECT-PERIOD>%f\n' %
                    to_float(params.planet_orbital_period, u.day))
        file.write('<OBJECT-ECCENTRICITY>%f\n' %
                    params.planet_eccentricity)
        file.write('<OBJECT-PERIAPSIS>%f\n' %
                    to_float(params.system_phase_of_periasteron, u.deg))
        file.write('<OBJECT-STAR-TEMPERATURE>%f\n' %
                    to_float(params.star_teff, u.K))
        file.write('<OBJECT-STAR-RADIUS>%f\n' %
                    to_float(params.star_radius, u.R_sun))
        file.write('<GEOMETRY>Observatory\n')
        file.write('<GEOMETRY-OBS-ALTITUDE>%f\n' %
                    to_float(params.system_distance, u.pc))
        file.write('<GEOMETRY-ALTITUDE-UNIT>pc\n')
        file.write('<GENERATOR-RANGE1>%f\n' %
                    to_float(params.lambda_min, params.target_wavelength_unit))
        file.write('<GENERATOR-RANGE2>%f\n' %
                    to_float(params.lambda_max, params.target_wavelength_unit))
        file.write(
            f'<GENERATOR-RANGEUNIT>{params.target_wavelength_unit}\n')
        file.write('<GENERATOR-RESOLUTION>%f\n' %
                    params.resolving_power)
        file.write('<GENERATOR-RESOLUTIONUNIT>RP\n')
        file.write('<GENERATOR-BEAM>%d\n' % params.beamValue)
        file.write('<GENERATOR-BEAM-UNIT>%s\n' % params.beamUnit)
        file.write('<GENERATOR-CONT-STELLAR>Y\n')
        file.write('<OBJECT-INCLINATION>%s\n' %
                    to_float(params.system_inclination, u.deg))
        file.write('<OBJECT-SOLAR-LATITUDE>0.0\n')
        file.write('<OBJECT-OBS-LATITUDE>0.0\n')
        file.write('<GENERATOR-RADUNITS>%s\n' % params.psg_rad_unit)
        file.write('<GENERATOR-GCM-BINNING>%d\n' % params.gcm_binning)
        file.write(
            f'<GENERATOR-GAS-MODEL>{bool_to_str[params.use_molec_signatures]}\n')
        file.write(f'<GENERATOR-NOISE>{params.detector_type}\n')
        file.write(
            f'<GENERATOR-NOISE2>{params.detector_dark_current}\n')
        file.write(
            f'<GENERATOR-NOISETIME>{params.detector_integration_time}\n')
        file.write(
            f'<GENERATOR-NOISEOTEMP>{params.detector_temperature}\n')
        file.write(
            f'<GENERATOR-NOISEOEFF>{params.detector_throughput:.1f}\n')
        file.write(
            f'<GENERATOR-NOISEOEMIS>{params.detector_emissivity:.1f}\n')
        file.write(
            f'<GENERATOR-NOISEFRAMES>{params.detector_number_of_integrations}\n')
        file.write(
            f'<GENERATOR-NOISEPIXELS>{params.detector_pixel_sampling}\n')
        file.write(f'<GENERATOR-NOISE1>{params.detector_read_noise}\n')
        file.write(
            f'<GENERATOR-DIAMTELE>{params.telescope_diameter:.1f}\n')
        file.write('<GENERATOR-TELESCOPE>SINGLE\n')
        file.write('<GENERATOR-TELESCOPE1>1\n')
        file.write('<GENERATOR-TELESCOPE2>1.0\n')
        file.write('<GENERATOR-TELESCOPE3>1.0\n')

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
    def __init__(self,header,data):
        self.header = header
        self.data = data
    
    @classmethod
    def from_rad(cls,filename):
        """
        Create a `PSGrad` object from a file. This is designed to load in
        the raw `.rad` output from PSG
        """
        raw_header = []
        raw_data = []
        with open(filename,'r',encoding='UTF-8') as file:
            for line in file:
                if line[0] == '#':
                    raw_header.append(line.replace('\n',''))
                else:
                    raw_data.append(line.replace('\n',''))
        header = {
            'warnings' : [],
            'errors' : [],
            'binning' : -1,
            'author' : '',
            'date' : '',
            'velocities' : {},
            'spectral_unit' : u.dimensionless_unscaled,
            'radiance_unit' : u.dimensionless_unscaled,

        }
        for i, item in enumerate(raw_header):
            if 'WARNING' in item:
                warning, kind, message = item.split('|')
                header['warnings'].append(dict(kind=kind,message=message))
            elif 'ERROR' in item:
                error, kind, message = item.split('|')
                header['errors'].append(dict(kind=kind,message=message))
            elif '3D spectroscopic simulation' in item:
                header['binning'] = int(re.findall(r'of ([\d]+) \(',item)[0])
            elif 'Planetary Spectrum Generator' in item:
                header['author'] = item[1:].strip()
            elif 'Synthesized' in item:
                header['date'] = item[1:].strip()
            elif 'Doppler velocities' in item:
                unit = u.Unit(re.findall(r'\[([\w\d/]+)\]',item)[0])
                keys = re.findall(r'\(([\w\d, \+]+)\)',item)[0].split(',')
                values = item.split(':')[1].split(',')
                for key, value in zip(keys, values):
                    header['velocities'][key] = float(value)*unit
            elif 'Spectral unit' in item:
                header['spectral_unit'] = u.Unit(re.findall(r'\[([\w\d/]+)\]',item)[0])
            elif 'Radiance unit' in item:
                header['radiance_unit'] = u.Unit(re.findall(r'\[([\w\d/]+)\]',item)[0])
        columns = raw_header[-1][1:].strip().split()
        dat = StringIO('\n'.join(raw_data))
        df = pd.read_csv(dat,names = columns,delim_whitespace=True)
        if len(df) == 0:
            raise ValueError('It looks like there might not be any data in this rad file.')
        data = {}
        if not columns[0] == 'Wave/freq':
            raise ValueError('.rad format is incorrect')
        data[columns[0]] = df[columns[0]].values * header['spectral_unit']
        for col in columns[1:]:
            data[col] = df[col].values * header['radiance_unit']
        return cls(header,data)
                
def get_reflected(cmb_rad:PSGrad,therm_rad:PSGrad,planet_name:str) -> u.Quantity:
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
    if not np.all(isclose(cmb_rad.data['Wave/freq'],therm_rad.data['Wave/freq'],1e-3*u.um)):
        raise ValueError('The spectral axes must be equivalent.')
    
    if 'Reflected' in cmb_rad.data.keys():
        return cmb_rad.data['Reflected']
    elif 'Reflected' in therm_rad.data.keys():
        return therm_rad.data['Reflected']
    elif (planet_name in cmb_rad.data.keys()) and (planet_name in therm_rad.data.keys()):
        return cmb_rad.data[planet_name] - therm_rad.data[planet_name]
    else:
        raise KeyError(f'Data array {planet_name} not found.')
        
                

            

        
