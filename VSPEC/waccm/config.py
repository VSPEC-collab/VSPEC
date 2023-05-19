"""
configs for WACCM module
"""
from astropy import units as u

psg_pressure_unit = u.bar
psg_aerosol_size_unit = u.m

atmosphere_type_dict = {'H2':45,'He':0,'H2O':1,'CO2':2,'O3':3,'N2O':4,'CO':5,'CH4':6,'O2':7,
                        'NO':8,'SO2':9,'NO2':10,'N2':22,'HNO3':12,'HO2NO2':'SEC[26404-66-0] Peroxynitric acid',
                        'N2O5':'XSEC[10102-03-1] Dinitrogen pentoxide','O':'KZ[08] Oxygen'}

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