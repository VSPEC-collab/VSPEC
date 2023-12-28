"""
Write extracted netCDF data in the PSG format.
"""
from netCDF4 import Dataset
from io import BytesIO
import numpy as np
from typing import List

import pypsg
from pypsg.cfg.globes import GCM

from VSPEC.waccm.read_nc import get_shape,get_coords
import VSPEC.waccm.read_nc as rw
from VSPEC.config import atmosphere_type_dict as mtype, aerosol_type_dict as atype, aerosol_name_dict



def get_gcm_params(data:Dataset,molecules:list,aerosols:list):
    """
    Get the <ATMOSPHERE-GCM-PARAMETERS> string for PSG

    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.
    molecules : list
        The variable names of the molecules.
    aerosols : list
        The variable names of the aerosols.
    
    Returns
    -------
    str:
        The parameter string.
    """
    _,N_layers,N_lat,N_lon = get_shape(data)
    lat,lon = get_coords(data)
    coords = f'{N_lon},{N_lat},{N_layers},{lon[0]:.1f},{lat[0]:.1f},{lon[1]-lon[0]:.2f},{lat[1]-lat[0]:.2f},'
    vars = f'Winds,Tsurf,Psurf,Albedo,Temperature,Pressure,'
    molecs = ','.join(molecules) + ','
    aeros = ','.join(aerosols + [f'{aero}_size' for aero in aerosols])
    return coords + vars + molecs + aeros


def get_cfg_params(data:Dataset,molecules:list,aerosols:list):
    """
    Get parameters for a PSG config file

    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.
    molecules : list
        The variable names of the molecules.
    aerosols : list
        The variable names of the aerosols.
    
    Returns
    -------
    dict:
        The config parameters.
    """
    _,N_layers,_,_ = get_shape(data)
    gases = molecules
    gas_types = [f'HIT[{mtype[gas]}]' if isinstance(mtype[gas],int) else mtype[gas] for gas in gases]
    aerosol_types = [atype[aerosol] for aerosol in aerosols]

    gcm_params = get_gcm_params(data,molecules,aerosols)
    params = {
        'ATMOSPHERE-DESCRIPTION': 'Whole Atmosphere Community Climate Model (WACCM) simulation',
        'ATMOSPHERE-STRUCTURE': 'Equilibrium',
        'ATMOSPHERE-LAYERS': f'{N_layers}',
        'ATMOSPHERE-NGAS': f'{len(gases)}',
        'ATMOSPHERE-GAS': ','.join(gases),
        'ATMOSPHERE-TYPE': ','.join(gas_types),
        'ATMOSPHERE-ABUN': ','.join(['1']*len(gases)),
        'ATMOSPHERE-UNIT': ','.join(['scl']*len(gases)),
        'ATMOSPHERE-NAERO': f'{len(aerosols)}',
        'ATMOSPHERE-AEROS': ','.join(aerosols),
        'ATMOSPHERE-ATYPE': ','.join(aerosol_types),
        'ATMOSPHERE-AABUN': ','.join(['1']*len(aerosols)),
        'ATMOSPHERE-AUNIT': ','.join(['scl']*len(aerosols)),
        'ATMOSPHERE-ASIZE': ','.join(['1']*len(aerosols)),
        'ATMOSPHERE-ASUNI': ','.join(['scl']*len(aerosols)),
        'ATMOSPHERE-GCM-PARAMETERS': gcm_params
    }
    return params

def get_binary_array(data:Dataset,itime:int,molecules:list,aerosols:list,background=None):
    """
    Get the content that goes between the <BINARY></BINARY> tags
    in the PSG config file.

    Parameters
    ----------
    data : netCDF4.Dataset
        The dataset to use.
    molecules : list
        The variable names of the molecules.
    aerosols : list
        The variable names of the aerosols.
    background : str, default=None
        The optional background gas to include.

    Notes
    -----
    The sections are as follows:
    U, V, Tsurf, Psurf, albedo, temp, press, molecs, aeros, aero sizes
    """
    def to_buffer(array:np.ndarray,buffer:BytesIO):
        # array.astype('float32').flatten('C').tofile(buffer)
        buffer.write(array.astype('float32').flatten('C').tobytes())
    buffer = BytesIO()
    U,V = rw.get_winds(data,itime)
    to_buffer(U,buffer)
    to_buffer(V,buffer)
    tsurf = rw.get_tsurf(data,itime)
    to_buffer(tsurf,buffer)
    psurf = rw.get_psurf(data,itime)
    to_buffer(np.log10(psurf),buffer)
    albedo = rw.get_albedo(data,itime)
    to_buffer(albedo,buffer)
    temperature = rw.get_temperature(data,itime)
    to_buffer(temperature,buffer)
    pressure = rw.get_pressure(data,itime)
    to_buffer(np.log10(pressure),buffer)
    mol_dict = rw.get_molecule_suite(data,itime,molecules,background)
    mol_list = [] # could use .values(), but this ensures the order regardless of python version
    for mol, abn in mol_dict.items():
        mol_list.append(mol)
        to_buffer(np.log10(abn),buffer)
    aero_abns,aero_sizes = [],[]
    for aerosol in aerosols:
        var_name = aerosol_name_dict[aerosol]['name']
        size_name = aerosol_name_dict[aerosol]['size']
        abn, size = rw.get_aerosol(data,itime,var_name,size_name)
        aero_abns.append(abn)
        aero_sizes.append(size)
    for abn in aero_abns:
        to_buffer(np.log10(abn),buffer)
    for size in aero_sizes:
        to_buffer(np.log10(size),buffer)
    return buffer, mol_list

def get_cfg_contents(data:Dataset,itime:int,molecules:list,aerosols:list,background=None):
    buffer,mol_list = get_binary_array(data,itime,molecules,aerosols,background)
    params = get_cfg_params(data,mol_list,aerosols)
    header = '\n'.join([f'<{key}>{value}' for key,value in params.items()])
    contents = bytes(header,encoding='UTF-8') + b'<BINARY>' + buffer.getvalue() + b'</BINARY>'
    buffer.close()
    return contents

def get_pycfg(
    data:Dataset,
    itime:int,
    molecules:list,
    aerosols:list,
    background=None
):
    """
    Get a PyConfig object.
    
    Parameters
    ----------
    data : netCDF4.Dataset
        The GCM dataset.
    itime : int
        The time index.
    molecules : list
        The variable names of the molecules.
    aerosols : list
        The variable names of the aerosols.
    
    Returns
    -------
    pycfg : pypsg.PyConfig
        The PyConfig object.
    """
    buffer,mol_list = get_binary_array(data,itime,molecules,aerosols,background)
    params = get_cfg_params(data,mol_list,aerosols)
    
    return pypsg.PyConfig(
        atmosphere=pypsg.cfg.EquilibriumAtmosphere.from_cfg(params),
        gcm=GCM(
            header=params['ATMOSPHERE-GCM-PARAMETERS'],
            dat=buffer.getvalue()
        )
    )
    
