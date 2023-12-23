"""
This module is designed to read output files from the
Whole Atmosphere Community Climate Model (WACCM) code
and convert it from netCDF to PSG file types.

The WACCM models used in writing this were produced by 
Howard Chen in 2023

This code is based on PSG conversion scripts for exoCAM
written by Geronimo Villanueva

"""
import warnings
from netCDF4 import Dataset
from astropy import units as u
import numpy as np

from VSPEC.config import psg_pressure_unit, psg_aerosol_size_unit

TIME_UNIT = u.day
ALBEDO_DEFAULT = 0.3

REQUIRED_VARIABLES = [
    "hyam",
    "hybm",
    "P0",
    "PS",
    "T",
    "lat",
    "lon",
    "PS",
    "time",
    "time_bnds"
]
OPTIONAL_VARIABLES = [
    "TS",
    "ASDIR",
    "U",
    "V",
]


class VariableAssumptionWarning(UserWarning):
    """
    A warning raised when a variable
    is not found in the netCDF file.
    """


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
    for var in REQUIRED_VARIABLES:
        try:
            data.variables[var]
        except KeyError:
            missing_vars.append(var)
    if len(missing_vars) == 0:
        pass
    else:
        raise KeyError(
            f'Dataset is missing required variables: {",".join(missing_vars)}'
        )
    
    missing_vars = []
    for var in OPTIONAL_VARIABLES:
        try:
            data.variables[var]
        except KeyError:
            missing_vars.append(var)
    if len(missing_vars) == 0:
        pass
    else:
        warnings.warn(
            f'Dataset is missing optional variables: {",".join(missing_vars)}',
            VariableAssumptionWarning
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
    time_in_days = time.to_value(TIME_UNIT)
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
        The shape of `data`, (N_time,N_layers,N_lat,N_lon)
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
    try:
        tsurf = np.array(data.variables['TS'][itime,:,:])
        return tsurf
    except KeyError:
        msg = 'Surface Temperature not explicitly stated. '
        msg += 'Using the value from the lowest layer.'
        warnings.warn(msg,VariableAssumptionWarning)
        temp = get_temperature(data,itime)
        return temp[0,:,:]
        

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
    try:
        U = np.flip(np.array(data.variables['U'][itime,:,:,:]),axis=0)
    except KeyError:
        msg = 'Wind Speed U not explicitly stated. Assuming zero.'
        warnings.warn(msg,VariableAssumptionWarning)
        _, nlayers, nlat, nlon = get_shape(data)
        U = np.zeros((nlayers,nlat,nlon))
    try:
        V = np.flip(np.array(data.variables['V'][itime,:,:,:]),axis=0)
    except KeyError:
        msg = 'Wind Speed V not explicitly stated. Assuming zero.'
        warnings.warn(msg,VariableAssumptionWarning)
        _, nlayers, nlat, nlon = get_shape(data)
        V = np.zeros((nlayers,nlat,nlon))
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
    try:
        albedo = np.array(data.variables['ASDIR'][itime,:,:])
        albedo = np.where((albedo>=0) & (albedo<=1.0) & (np.isfinite(albedo)), albedo, 0.3)
    except KeyError:
        msg = f'Albedo not explicitly stated. Using {ALBEDO_DEFAULT}.'
        warnings.warn(msg,VariableAssumptionWarning)
        _, _, nlat, nlon = get_shape(data)
        albedo = np.ones((nlat,nlon)) * ALBEDO_DEFAULT
    return albedo

def get_aerosol(data:Dataset,itime:int,name:str,size:str):
    """
    Get the abundance and size of an aerosol.
    
    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.
    itime : int
        The timestep to use.
    name : str
        The variable name of the aerosol.
    size : str
        The variable name of the aerosol size.
    
    Returns
    -------
    aero : np.ndarray
        The abundance of the aerosol in kg/kg (N_layers,N_lat,N_lon)
    aero_size : np.ndarray
        The size of the aerosol in m (N_layers,N_lat,N_lon)
    
    """
    aero = np.flip(np.array(data.variables[name][itime,:,:,:]),axis=0)
    aero = np.where((aero>0) & (np.isfinite(aero)), aero, 1e-30)

    aero_size = np.flip(np.array(data.variables[size][itime,:,:,:]),axis=0)
    aero_size = np.where((aero_size>0) & (np.isfinite(aero_size)), aero_size, 1.)

    aero_size_unit = u.Unit(data.variables[size].units)
    return aero, aero_size * (1*aero_size_unit).to_value(psg_aerosol_size_unit)

def get_water(data:Dataset,itime:int):
    """
    Shortcut for calling `get_aerosol` for liquid water.

    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.
    itime : int
        The timestep to use.
    
    Returns
    -------
    water : np.ndarray
        The abundance of water in kg/kg (N_layers,N_lat,N_lon)
    water_size : np.ndarray
        The size of water in m (N_layers,N_lat,N_lon)
    """
    name = 'CLDLIQ'
    size = 'REL'
    return get_aerosol(data,itime,name,size)
def get_ice(data:Dataset,itime:int):
    """
    Shortcut for calling `get_aerosol` for water ice.

    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.
    itime : int
        The timestep to use.
    
    Returns
    -------
    waterice : np.ndarray
        The abundance of ice in kg/kg (N_layers,N_lat,N_lon)
    waterice_size : np.ndarray
        The size of ice in m (N_layers,N_lat,N_lon)
    """
    name = 'CLDICE'
    size = 'REI'
    return get_aerosol(data,itime,name,size)

def get_molecule(data:Dataset,itime:int,name:str):
    """
    Get the abundance of a molecule.
    
    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.
    itime : int
        The timestep to use.
    name : str
        The variable name of the molecule.
    
    Returns
    -------
    molec : np.ndarray
        The volume mixing ratio of the molecule in mol/mol (N_layers,N_lat,N_lon)
    """
    def molec_translator(name):
        if name == 'NO2':
            return 'NOX'
        else:
            return name
    molec = np.flip(np.array(data.variables[molec_translator(name)][itime,:,:,:]),axis=0)
    molec = np.where((molec>0) & (np.isfinite(molec)), molec, 1e-30)
    return molec

def get_molecule_suite(data:Dataset,itime:int,names:list,background:str=None)->dict:
    """
    Get the abundance of a suite of molecules.
    
    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.
    itime : int
        The timestep to use.
    names : list of str
        The variable names of the molecules.
    background : str, default=None
        The variable name of a background gas to include.
    
    Returns
    -------
    molec : dict
        The volume mixing ratios of the molecules in mol/mol (N_layers,N_lat,N_lon)
    """
    molecs = dict()
    for name in names:
        molecs[name] = get_molecule(data,itime,name)
    if background is not None:
        if background in names:
            raise ValueError(
                'Do not know how to handle specifying a background'
                'gas that is already in our dataset.'
            )
        else:
            _,N_layer,N_lat,N_lon = get_shape(data)
            background_abn = np.ones(shape=(N_layer,N_lat,N_lon))
            for abn in molecs.values():
                background_abn -= abn
            if np.any(background_abn<0):
                raise ValueError('Cannot have negative abundance.')
            molecs[background] = background_abn
    return molecs



