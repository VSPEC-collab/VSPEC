"""
VSPEC configurations
"""
from astropy import units as u
from VSPEC.helpers import MSH
stellar_area_unit = MSH
flux_unit = u.Unit('W m-2 um-1')
noise_unit = u.electron/u.s

PSG_PORT = 3000

PSG_EXT_URL = 'https://psg.gsfc.nasa.gov'