import pytest
from pathlib import Path

from VSPEC.params.gcm import psgParameters

def test_psgParameters_from_dict():
    # Create a dictionary with the PSG parameters data
    psg_params_data = {
        'gcm_binning': '3',
        'phase_binning': '1',
        'use_molecular_signatures': 'True',
        'nmax':'0',
        'lmax':'0',
        'continuum':[],        
    }

    # Create a psgParameters instance using the _from_dict class method
    psg_params = psgParameters.from_dict(psg_params_data)

    # Perform assertions on the instance attributes
    assert psg_params.gcm_binning == 3
    assert psg_params.phase_binning == 1
    assert psg_params.use_molecular_signatures is True
    assert psg_params.nmax == 0
    assert psg_params.lmax == 0
    assert isinstance(psg_params.continuum,list)

def test_psgParameters_to_psg():
    # Create a psgParameters instance
    psg_params = psgParameters(
        gcm_binning=3,
        phase_binning=1,
        use_molecular_signatures=True,
        nmax=0,
        lmax=0,
        continuum=['Rayleigh', 'Refraction']
        )

    # Convert the PSG parameters to the PSG input format
    psg_input = psg_params.to_psg()

    # Perform assertions on the PSG input dictionary
    assert psg_input['GENERATOR-GCM-BINNING'] == '3'
    assert psg_input['GENERATOR-GAS-MODEL'] == 'Y'
    assert psg_input['ATMOSPHERE-NMAX'] == '0'
    assert psg_input['ATMOSPHERE-LMAX'] == '0'
    assert psg_input['ATMOSPHERE-CONTINUUM'] == 'Rayleigh,Refraction'

