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


psg_pressure_unit = u.bar
psg_aerosol_size_unit = u.m

atmosphere_type_dict = {'H2':45,'He':0,'H2O':1,'CO2':2,'O3':3,'N2O':4,'CO':5,'CH4':6,'O2':7,
                        'NO':8,'SO2':9,'NO2':10,'N2':22,'HNO3':12,'HO2NO2':'SEC[26404-66-0] Peroxynitric acid',
                        'N2O5':'XSEC[10102-03-1] Dinitrogen pentoxide','O':'KZ[08] Oxygen',
                        'OH':'EXO[OH]'}

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

aerosol_type_dict = {
    'Water': 'AFCRL_Water_HRI',
    'WaterIce': 'Warren_ice_HRI'
}