"""VSPEC module to communicate with the PSG API

This module communucates between `VSPEC` and
and the Planetary Spectrum Generator via the API.
"""
##### GERONIMO'S CODE #####
# ---------------------------------------------------------------
# Script to compute phase curves with PSG/GlobES
# Villanueva, Suissa - NASA Goddard Space Flight Center
# February 2021
# ---------------------------------------------------------------
# Adapted by Ted Johnson to run as a package, Nov 2022

import os


def call_api(config_path: str, psg_url: str = 'https://psg.gsfc.nasa.gov',
             api_key: str = None, output_type: str = None, app: str = None,
             outfile: str = None, verbose: bool = False) -> None:
    """
    Call the PSG api

    Build and execute an API query to communicate with PSG.

    Parameters
    ----------
    config_path : str or `~pathlib.Path`
        The path to the `PSG` config file.
    psg_url : str, default='https://psg.gsfc.nasa.gov'
        The URL of the `PSG` API. Use `http://localhost:3000` if running locally.
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
