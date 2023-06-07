"""
VSPEC configurations
"""
from astropy import units as u
import numpy as np
from pathlib import Path

MSH = u.def_unit('msh', 1e-6 * 0.5 * 4*np.pi*u.R_sun**2)
"""Micro-solar hemisphere

This is a standard unit in heliophysics that
equals one millionth of one half the surface area of the Sun.
"""

stellar_area_unit = MSH
"""
The standard stellar surface area unit.
"""
flux_unit = u.Unit('W m-2 um-1')
"""
The standard unit of flux.
"""

planet_distance_unit = u.AU
"""
The standard unit of planetary semimajor axis.

Determined by PSG.
"""

planet_radius_unit = u.km
"""
The standard unit of planet radius.

Determined by PSG.
"""

period_unit = u.day
"""
The standard unit of planet orbital period.

Determined by PSG
"""

psg_encoding = 'UTF-8'
"""
Default encoding for files from PSG.
"""

PSG_PORT = 3000
"""
Default port to run PSG locally.
"""

PSG_EXT_URL = 'https://psg.gsfc.nasa.gov'
"""
The URL of the external PSG version.
"""

PSG_CFG_MAX_LINES = 1500
"""
The maximum number of lines to allow in
the PSG config file. Do not set this over 2000,
as PSG will stop updating.

Notes
-----
For security reasons, PSG only allows ``.cfg`` files to be
up to 2000 lines long. This means that, every so often,
we have to send a ``set`` command, which resets everything.
This command is sent after the ``.cfg`` returned by PSG reaches
`PSG_CFG_MAX_LINES` lines long.
"""


psg_pressure_unit = u.bar
"""
PSG atmospheric pressure unit.
"""
psg_aerosol_size_unit = u.m
"""
PSG aerosol size unit.
"""

atmosphere_type_dict = {'H2':45,'He':0,'H2O':1,'CO2':2,'O3':3,'N2O':4,'CO':5,'CH4':6,'O2':7,
                        'NO':8,'SO2':9,'NO2':10,'N2':22,'HNO3':12,'HO2NO2':'SEC[26404-66-0] Peroxynitric acid',
                        'N2O5':'XSEC[10102-03-1] Dinitrogen pentoxide','O':'KZ[08] Oxygen',
                        'OH':'EXO[OH]'}
"""
A dictionary mapping molecular species to the default
database to use to create opacities. These are all
internal to PSG, but must be set by ``VSPEC``.
"""

aerosol_name_dict = {
    'Water':{
        'name':'CLDLIQ',
        'size':'REL'
    },
    'WaterIce':{
        'name':'CLDICE',
        'size':'REI'
    }
}
"""
A dictionary mapping aerosol species from their PSG name
to their name in the WACCM NetCDF format.
"""

aerosol_type_dict = {
    'Water': 'AFCRL_Water_HRI',
    'WaterIce': 'Warren_ice_HRI'
}
"""
A dictionary mapping aerosol species to the default
database to use. These are all
internal to PSG, but must be set by ``VSPEC``.
"""
############################
# Paths and file conventions
############################

N_ZFILL = 5
"""
int
    `__width` argument for filename `str.zfill()` calls. When writing 
    and reading files in the `VSPEC` output, this variable specifies
    the number of leading zeros to use in the filename.
"""

RAW_PHOENIX_PATH = Path(__file__).parent / 'data' / 'NextGenModels' / 'RawData'
"""
pathlib.Path
    The path to the raw PHOENIX stellar models.
"""

BINNED_PHOENIX_PATH = Path(__file__).parent / 'data' / 'NextGenModels' / 'binned'
"""
pathlib.Path
    The path to the binned PHOENIX stellar models.
"""

EXAMPLE_GCM_PATH = Path(__file__).parent / 'data' / 'GCMs'
"""
pathlib.Path
    The path to example GCMs.
"""

MOLEC_DATA_PATH = Path(__file__).parent / 'data' / 'molec.json'
"""
pathlib.Path
    The path to the file containing molecular data for analysis.
"""
PRESET_PATH = Path(__file__).parent / 'presets'
"""
pathlib.Path
    The path to parameter presets.
"""
