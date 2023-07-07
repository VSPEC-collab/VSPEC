"""
VSPEC configurations

This module contains global configurations used in the VSPEC code.
"""
from astropy import units as u
import numpy as np
from pathlib import Path

MSH = u.def_unit('msh', 1e-6 * 0.5 * 4*np.pi*u.R_sun**2)
"""
Micro-solar hemisphere

This is a standard unit in heliophysics that
equals one millionth of one half the surface area of the Sun.

:type: astropy.units.Unit
"""

stellar_area_unit = MSH
"""
The standard stellar surface area unit.

This unit is used to represent the surface area of stars in VSPEC.
The micro-solar hemisphere is chosen because most Sun Spot literature uses
this unit.

:type: astropy.units.Unit
"""

starspot_initial_area = 10*MSH
"""
Initial ``StarSpot`` area.

Because spots grow exponentially, they can't start at 0 area.
When they are born they are given this small area.

:type: astropy.units.Quantity

.. todo::
    This should optionaly be set by the user. So that smaller
    star spot area regimes are accessible.
"""

flux_unit = u.Unit('W m-2 um-1')
"""
The standard unit of flux.

This unit is used to standardize the flux in VSPEC calculations.
:math:`W m^{-2} \\mu m^{-1}` is chosen because it is the standard
spectral irrandience unit in PSG.

:type: astropy.units.Unit
"""

teff_unit = u.K
"""
The standard unit of temperature.

This selection standardizes units across the package.

:type: astropy.units.Unit
"""

wl_unit = u.um
"""
The standard unit of wavelength.

The output wavelength can still be changed by the user, but internally
we want units to be consistent.

:type: astropy.units.Unit
"""

nlat = 500
"""
The default latitude resolution for the stellar model. This should
be set by finding a balance between noticing small changes in spots/faculae
and computation time.

:type: int
"""

nlon = 1000
"""
The default longitude resolution for the stellar model. This should
be set by finding a balance between noticing small changes in spots/faculae
and computation time.

:type: int
"""

grid_teff_bounds = (2300*u.K, 3900*u.K)
"""
The limits on the effective temperature allowed by the grid.

:type: tuple of astropy.units.Quantity
"""

planet_distance_unit = u.AU
"""
The standard unit of planetary semimajor axis.

This unit is determined by PSG and used to standardize
the semimajor axis of planets in VSPEC.

:type: astropy.units.Unit
"""

planet_radius_unit = u.km
"""
The standard unit of planet radius.

This unit is determined by PSG and used to standardize
the radius of planets in VSPEC.

:type: astropy.units.Unit
"""

period_unit = u.day
"""
The standard unit of planet orbital period.

This unit is determined by PSG and used to standardize
the orbital and rotational periods of planets in VSPEC.

:type: astropy.units.Unit
"""

psg_encoding = 'UTF-8'
"""
Default encoding for files from PSG.

:type: str

.. depricated::
    This may not be needed now that we recieve data from PSG as bytes.
    I should check.
"""

PSG_PORT = 3000
"""
Default port to run PSG locally.

This port number is used to access the PSG API locally.

:type: int
"""

PSG_EXT_URL = 'https://psg.gsfc.nasa.gov'
"""
The URL of the external PSG (Planetary Spectrum Generator) version.

This URL is used to access the PSG web interface externally.

.. warning::
    The external PSG server only allows 100 anonymous API
    calls per day. Please either use an API key or a local
    PSG installation.

:type: str
"""

PSG_CFG_MAX_LINES = 1500
"""
The maximum number of lines to allow in
the PSG config file. Do not set this over 2000,
as PSG will stop updating.

:type: int

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

This unit is determined by PSG and used to standardize
the atmospheric pressure of planets in VSPEC.

:type: astropy.units.Unit
"""
psg_aerosol_size_unit = u.m
"""
PSG aerosol size unit.

This unit is determined by PSG and used to
standardize aerosol size in VSPEC.

:type: astropy.units.Unit
"""

atmosphere_type_dict = {'H2':45,'He':0,'H2O':1,'CO2':2,'O3':3,'N2O':4,'CO':5,'CH4':6,'O2':7,
                        'NO':8,'SO2':9,'NO2':10,'N2':22,'HNO3':12,'HO2NO2':'SEC[26404-66-0] Peroxynitric acid',
                        'N2O5':'XSEC[10102-03-1] Dinitrogen pentoxide','O':'KZ[08] Oxygen',
                        'OH':'EXO[OH]'}
"""
A dictionary mapping molecular species to the default
database to use to create opacities. These are all
internal to PSG, but must be set by ``VSPEC``.

Integers mean that we want to use data from the HITRAN database, which for a number ``N``
is represented in PSG by ``HIT[N]``. Strings are sent straight to PSG as is.

:type: dict
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

:type: dict
"""

aerosol_type_dict = {
    'Water': 'AFCRL_Water_HRI',
    'WaterIce': 'Warren_ice_HRI'
}
"""
A dictionary mapping aerosol species to the default
database to use. These are all
internal to PSG, but must be set by ``VSPEC``.

:type: dict
"""
############################
# Paths and file conventions
############################

N_ZFILL = 5
"""
``__width`` argument for filename ``str.zfill()`` calls. When writing
and reading files in the `VSPEC` output, this variable specifies
the number of leading zeros to use in the filename.

:type: int
"""

RAW_PHOENIX_PATH = Path(__file__).parent / 'data' / 'NextGenModels' / 'RawData'
"""
The path to the raw PHOENIX stellar models.

:type: pathlib.Path
"""

BINNED_PHOENIX_PATH = Path(__file__).parent / 'data' / 'NextGenModels' / 'binned'
"""
The path to the binned PHOENIX stellar models.

:type: pathlib.Path
"""

EXAMPLE_GCM_PATH = Path(__file__).parent / 'data' / 'GCMs'
"""
The path to example GCMs.

:type: pathlib.Path
"""

MOLEC_DATA_PATH = Path(__file__).parent / 'data' / 'molec.json'
"""
The path to the file containing molecular data for analysis.

:type: pathlib.Path
"""
PRESET_PATH = Path(__file__).parent / 'presets'
"""
The path to parameter presets.

:type: pathlib.Path
"""

VSPEC_PARENT_PATH = Path('.vspec')