"""VSPEC module to communicate with the PSG API

This module communucates between `VSPEC` and
and the Planetary Spectrum Generator via the API.
"""

import os
from pathlib import Path
from astropy import units as u

from VSPEC.read_info import ParamModel
from VSPEC.helpers import to_float


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
        file.write('<OBJECT-NAME>Planet\n')
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
