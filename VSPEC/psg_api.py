
##### GERONIMO'S CODE #####
# ---------------------------------------------------------------
# Script to compute phase curves with PSG/GlobES
# Villanueva, Suissa - NASA Goddard Space Flight Center
# February 2021
# ---------------------------------------------------------------
# Adapted by Ted Johnson to run as a package, Nov 2022

import numpy as np
import os
from pathlib import Path
import VSPEC.read_info as read_info
from astropy import units as u
from VSPEC.geometry import SystemGeometry
from VSPEC.helpers import to_float


def call_api(config_path,psg_url='https://psg.gsfc.nasa.gov',
            api_key=None,type=None,app=None,outfile=None):
    """call api
    Call the PSG api

    Args:
        config_path (str or pathlib.Path): path to config file
        psg_url (str): url of psg api
        api_key (str): if not using local version of psg, ask Geronimo for one of these
        type (str): type of output to ask for
        app (str): which PSG app to call
        outfile (str): target to write out response
    
    Returns:
        None
    """
    cmd = 'curl -s'
    if api_key:
        cmd = cmd + f' -d key={api_key}'
    if app:
        cmd = cmd + f' -d app={app}'
    if type:
        cmd = cmd + f' -d type={type}'
    cmd = cmd + f' --data-urlencode file@{config_path}'
    cmd = cmd + f' {psg_url}/api.php'
    if outfile:
        cmd = cmd + f' > {outfile}'
    os.system(cmd)