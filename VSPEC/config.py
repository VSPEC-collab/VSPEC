"""
VSPEC configurations
"""
from astropy import units as u
from VSPEC.helpers import MSH
stellar_area_unit = MSH
flux_unit = u.Unit('W m-2 um-1')
noise_unit = u.electron/u.s
planet_distance_unit = u.AU
planet_radius_unit = u.km
period_unit = u.day

psg_encoding = 'UTF-8'

PSG_PORT = 3000

PSG_EXT_URL = 'https://psg.gsfc.nasa.gov'

PSG_CFG_MAX_LINES = 1500