"""
This module is designed to read output files from the
Whole Atmosphere Community Climate Model (WACCM) code
and convert it from netCDF to PSG file types.

The WACCM models used in writing this were produced by 
Howard Chen in 2023

This code is based on PSG conversion scripts for exoCAM
written by Geronimo Villanueva

"""
from netCDF4 import Dataset
from pathlib import Path
import json
from astropy import units as u
import numpy as np

from VSPEC.waccm.config import psg_pressure_unit

VAR_LIST = Path(__file__).parent / 'variables.json'

time_unit = u.day

def validate_variables(data:Dataset):
    """
    Check to make sure that the file
    contains all necessary variables.

    Parameters
    ----------
    data : netCDF4.Dataset
        The data to be checked
    """
    missing_vars = []
    with open(VAR_LIST,'r',encoding='UTF-8') as file:
        vars = json.loads(file.read())
    for var in vars:
        try:
            data.variables[var]
        except KeyError:
            missing_vars.append(var)
    if len(missing_vars) == 0:
        return None
    else:
        raise KeyError(
            f'Dataset is missing required variables: {",".join(missing_vars)}'
        )
def get_time_index(data:Dataset,time:u.Quantity):
    """
    Get the index `itime` given a time quantity.

    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.
    time : astropy.units.Quantity
        The time Quantity.
    
    Returns
    -------
    itime : int
        The time index of `time`.
    """
    time_in_days = time.to_value(time_unit)
    time_left = data.variables['time_bnds'][:,0]
    time_right = data.variables['time_bnds'][:,1]
    itime = np.argwhere((time_in_days > time_left) & (time_in_days <= time_right))[0][0]
    return itime

def get_shape(data:Dataset):
    """
    Get the shape of a Dataset

    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.
    
    Returns
    -------
    tuple
        The shape of `data`
    """
    N_time = data.variables['T'].shape[0]
    N_layers = data.variables['T'].shape[1]
    N_lat = data.variables['T'].shape[2]
    N_lon = data.variables['T'].shape[3]
    return N_time,N_layers,N_lat,N_lon

def get_psurf(data:Dataset,itime:int):
    """
    Get the surface pressure.

    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.
    itime : int
        The timestep to use.
    
    Returns
    -------
    psurf : np.ndarray
        The surface pressure (N_lat,N_lon) in bar
    """
    psurf = data.variables['PS'][itime,:,:]
    ps_unit = u.Unit(data.variables['PS'].units)
    return psurf * (1*ps_unit).to_value(psg_pressure_unit)

def get_pressure(data:Dataset,itime:int):
    """
    Get the pressure.
    
    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.
    itime : int
        The timestep to use.
    
    Returns
    -------
    pressure : np.ndarray
        The pressure (N_layers,N_lat,N_lon) in bar
    """
    hyam = np.flipud(data.variables['hyam'][:])
    hybm = np.flipud(data.variables['hybm'][:])
    ps = get_psurf(data,itime)
    pressure = hyam[:, np.newaxis, np.newaxis] + hybm[:, np.newaxis, np.newaxis] * ps[np.newaxis, :, :]
    return pressure

def get_temperature(data:Dataset,itime:int):
    """
    Get the temperature.
    
    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.
    itime : int
        The timestep to use.
    
    Returns
    -------
    temperature : np.ndarray
        The temperature (N_layers,N_lat,N_lon) in K
    """
    temperature = np.flip(np.array(data.variables['T'][itime,:,:,:]),axis=0)
    return temperature
def get_tsurf(data:Dataset,itime:int):
    """
    Get the surface temperature.
    
    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.
    itime : int
        The timestep to use.
    
    Returns
    -------
    tsurf : np.ndarray
        The surface temperature (N_lat,N_lon) in K
    """
    tsurf = np.array(data.variables['TS'][itime,:,:])
    return tsurf

def get_winds(data:Dataset,itime:int):
    """
    Get the winds.
    
    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.
    itime : int
        The timestep to use.
    
    Returns
    -------
    U : np.ndarray
        The wind speed in the U direction (N_layers,N_lat,N_lon) in m/s
    V : np.ndarray
        The wind speed in the V direction (N_layers,N_lat,N_lon) in m/s
    """
    U = np.flip(np.array(data.variables['U'][itime,:,:,:]),axis=0)
    V = np.flip(np.array(data.variables['V'][itime,:,:,:]),axis=0)
    return U, V

def get_coords(data:Dataset):
    """
    Get latitude and longitude coordinates.

    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.

    Returns
    -------
    lat : np.ndarray
        The latitude coordinates in degrees (N_lat,)
    lon : np.ndarray
        The longitude coodinates in degrees (N_lon,)
    """
    lat = np.array(data.variables['lat'][:])
    lon = np.array(data.variables['lon'][:])
    return lat,lon

def get_albedo(data:Dataset,itime:int):
    """
    Get the albedo.
    
    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.
    itime : int
        The timestep to use.
    
    Returns
    -------
    albedo : np.ndarray
        The albedo (N_lat,N_lon)
    """
    albedo = np.array(data.variables['ASDIR'][itime,:,:])
    return albedo

