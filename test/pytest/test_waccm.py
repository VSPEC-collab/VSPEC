
from os import chdir
from pathlib import Path
import pytest
import numpy as np
import matplotlib.pyplot as plt

import netCDF4 as nc

    
from VSPEC.waccm.read_nc import validate_variables, get_time_index, time_unit,get_shape
import VSPEC.waccm.read_nc as rw
import VSPEC.waccm.write_psg as wp

chdir(Path(__file__).parent)

# DATA_PATH = Path('/Users/tjohns39/Documents/GCMs/WACCM/TR1e_flare_1yr_psg.nc')
DATA_PATH = Path.home() / 'gcms' / 'waccm' / 'TR1e_flare_1month_var.nc'


def read_TR1():
    path = Path('/Users/tjohns39/Documents/GCMs/WACCM/TR1e_flare_1yr_psg.nc')
    with nc.Dataset(path,'r',format='NETCDF4') as data:
        validate_variables(data)
def test_validate_vars():
    with nc.Dataset(DATA_PATH,'r',format='NETCDF4') as data:
        validate_variables(data)

def test_get_time_index():
    with nc.Dataset(DATA_PATH,'r',format='NETCDF4') as data:
        time = np.array(data.variables['time'][:])*time_unit
        for i,t in enumerate(time):
            assert get_time_index(data,t) == i
def test_get_shape():
    with nc.Dataset(DATA_PATH,'r',format='NETCDF4') as data:
        shape = get_shape(data)
        assert data.variables['T'].shape == shape
def test_surface_pressure():
    with nc.Dataset(DATA_PATH,'r',format='NETCDF4') as data:
        ps = rw.get_psurf(data,0)
        _,_,N_lat,N_lon = get_shape(data)
        assert ps.shape == (N_lat,N_lon)
def test_pressure():
    with nc.Dataset(DATA_PATH,'r',format='NETCDF4') as data:
        press = rw.get_pressure(data,0)
        _,N_layer,N_lat,N_lon = get_shape(data)
        assert press.shape == (N_layer,N_lat,N_lon)
def test_tsurf():
    with nc.Dataset(DATA_PATH,'r',format='NETCDF4') as data:
        tsurf = rw.get_tsurf(data,0)
        _,_,N_lat,N_lon = get_shape(data)
        assert tsurf.shape == (N_lat,N_lon)

def test_temperature():
    with nc.Dataset(DATA_PATH,'r',format='NETCDF4') as data:
        temp = rw.get_temperature(data,0)
        tsurf = rw.get_tsurf(data,0)
        _,N_layer,N_lat,N_lon = get_shape(data)
        assert temp.shape == (N_layer,N_lat,N_lon)
        res = temp[0,:,:] - tsurf
        assert np.median(np.abs(res/tsurf)) < 0.1
def test_get_winds():
    with nc.Dataset(DATA_PATH,'r',format='NETCDF4') as data:
        U,V = rw.get_winds(data,0)
        _,N_layer,N_lat,N_lon = get_shape(data)
        assert U.shape == (N_layer,N_lat,N_lon)
        assert V.shape == (N_layer,N_lat,N_lon)

def test_get_coords():
    with nc.Dataset(DATA_PATH,'r',format='NETCDF4') as data:
        lat,lon = rw.get_coords(data)
        _,_,N_lat,N_lon = get_shape(data)
        assert lat.shape == (N_lat,)
        assert lon.shape == (N_lon,)
        tsurf = rw.get_tsurf(data,0)
        plt.pcolormesh(lon,lat,tsurf)

def test_albedo():
    with nc.Dataset(DATA_PATH,'r',format='NETCDF4') as data:
        albedo = rw.get_albedo(data,0)
        _,_,N_lat,N_lon = get_shape(data)
        assert albedo.shape == (N_lat,N_lon)
def test_aerosol():
    with nc.Dataset(DATA_PATH,'r',format='NETCDF4') as data:
        water,water_size = rw.get_aerosol(data,0,'CLDLIQ','REL')
        _,N_layer,N_lat,N_lon = get_shape(data)
        assert water.shape == (N_layer,N_lat,N_lon)
        assert water_size.shape == (N_layer,N_lat,N_lon)
        water,water_size = rw.get_water(data,0)
        assert water.shape == (N_layer,N_lat,N_lon)
        assert water_size.shape == (N_layer,N_lat,N_lon)
        ice,ice_size = rw.get_ice(data,0)
        assert ice.shape == (N_layer,N_lat,N_lon)
        assert ice_size.shape == (N_layer,N_lat,N_lon)
def test_molecules():
    with nc.Dataset(DATA_PATH,'r',format='NETCDF4') as data:
        co2 = rw.get_molecule(data,0,'CO2')
        _,N_layer,N_lat,N_lon = get_shape(data)
        assert co2.shape == (N_layer,N_lat,N_lon)
        molecs = rw.get_molecule_suite(data,0,['CO2','H2O'],background='N2')
        assert molecs['CO2'].shape == (N_layer,N_lat,N_lon)
        assert molecs['H2O'].shape == (N_layer,N_lat,N_lon)
        assert molecs['N2'].shape == (N_layer,N_lat,N_lon)
        assert np.all(np.abs(molecs['CO2']+molecs['H2O']+molecs['N2']-1) < 1e-6)

def test_write_cfg_params():
    with nc.Dataset(DATA_PATH,'r',format='NETCDF4') as data:
        molecs = rw.get_molecule_suite(data,0,['CO2','H2O'],background='N2')
        aerosols = ['Water','WaterIce']
        params = wp.get_cfg_params(data,molecs,aerosols)
        assert isinstance(params,dict)
def test_write_binary_array():
    with nc.Dataset(DATA_PATH,'r',format='NETCDF4') as data:
        b,mol = wp.get_binary_array(data,0,['CO2','H2O'],['Water','WaterIce'],'N2')
def test_get_cfg_contents():
    with nc.Dataset(DATA_PATH,'r',format='NETCDF4') as data:
        contents = wp.get_cfg_contents(data,0,['CO2','H2O'],['Water','WaterIce'],'N2')
        # call_api(output_type='set',app='globes',config_data=contents)
    wpath = Path(__file__).parent / 'temp.cfg'
    with open(wpath,'wb') as file:
        file.write(contents)
    wpath.unlink()


if __name__ in '__main__':
    pytest.main(args=[__file__])