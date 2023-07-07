#!/usr/bin/env python

"""
Tests for `VSPEC.psg_api` module
"""
from pathlib import Path
import pytest
from astropy import units as u
import matplotlib.pyplot as plt

from VSPEC.psg_api import call_api,call_api_from_file, PSGrad, get_reflected, parse_full_output
from VSPEC.helpers import is_port_in_use,set_psg_state

API_KEY_PATH = Path.home() / 'psg_key.txt'
PSG_CONFIG_PATH = Path(__file__).parent / 'data' / 'test_cfg.txt'
VSPEC_CONFIG_PATH = Path(__file__).parent / 'default.yaml'
PSG_PORT = 3000

RAD_PATH = Path(__file__).parent / 'data' / 'transit_reflected'




@pytest.mark.skipif(not API_KEY_PATH.exists(),reason='This test requires an API key')
def test_call_api_nonlocal():
    """
    Run tests for `VSPEC.psg_api.call_api()`
    This test expects that you have a file at
    `~/` called `psg_key.txt` that contains your own
    API key. Otherwise it is skipped.
    """
    psg_url = 'https://psg.gsfc.nasa.gov'
    with open(API_KEY_PATH,'r',encoding='UTF-8') as file:
        api_key = file.read()
    data = '<OBJECT>Exoplanet\n<OBJECT-NAME>ProxCenb'
    content = call_api(psg_url=psg_url,api_key=api_key,output_type='cfg',config_data=data)
    assert b'<OBJECT>Exoplanet' in content
    
    
@pytest.mark.skipif(not is_port_in_use(3000),reason='PSG must be running locally to run this test')
def test_call_api_local():
    """
    Run tests for `VSPEC.psg_api.call_api()`
    """
    previous_state = is_port_in_use(PSG_PORT)
    set_psg_state(True)
    psg_url = 'http://localhost:3000'
    data = '<OBJECT>Exoplanet\n<OBJECT-NAME>ProxCenb'
    content = call_api(psg_url=psg_url,output_type='cfg',config_data=data)
    assert b'<OBJECT>Exoplanet' in content
    set_psg_state(previous_state)

@pytest.mark.skipif(not is_port_in_use(3000),reason='PSG must be running locally to run this test')
def test_call_api_from_file():
    """
    Run tests for `VSPEC.psg_api.call_api()` while giving the file contents rather than the path.
    """
    previous_state = is_port_in_use(PSG_PORT)
    set_psg_state(True)
    psg_url = 'http://localhost:3000'
    content = call_api_from_file(config_path=PSG_CONFIG_PATH,psg_url=psg_url,output_type='cfg',app='globes')
    assert b'<OBJECT>Exoplanet' in content
    
    set_psg_state(previous_state)


def test_PSGrad():
    """
    Run tests for `VSPEC.psg_api.PSGrad()`
    """
    file = Path(__file__).parent / 'data' / 'test_rad.rad'
    rad = PSGrad.from_rad(file)
    assert isinstance(rad.header,dict)
    assert isinstance(rad.data,dict)
    with pytest.raises(ValueError):
        file = Path(__file__).parent / 'data' / 'gcm_error.rad'
        PSGrad.from_rad(file)

def test_get_reflected():
    """
    Run tests for `VSPEC.psg_api.get_reflected()`
    """
    data_dir = Path(__file__).parent / 'data' / 'test_reflected'
    planet_name = 'proxima-Cen-b'

    atm_cmb = PSGrad.from_rad(data_dir / 'atm_cmb.rad')
    atm_therm = PSGrad.from_rad(data_dir / 'atm_therm.rad')
    ref = get_reflected(atm_cmb,atm_therm,planet_name)
    assert isinstance(ref,u.Quantity)
    assert len(ref) == len(atm_cmb.data['Wave/freq'])

    no_atm_cmb = PSGrad.from_rad(data_dir / 'no_atm_cmb.rad')
    no_atm_therm = PSGrad.from_rad(data_dir / 'no_atm_therm.rad')
    ref = get_reflected(no_atm_cmb,no_atm_therm,planet_name)
    assert isinstance(ref,u.Quantity)
    assert len(ref) == len(atm_cmb.data['Wave/freq'])

    # also works if just one has separated thermal+reflected columns
    ref = get_reflected(atm_cmb,no_atm_therm,planet_name)
    assert isinstance(ref,u.Quantity)
    assert len(ref) == len(atm_cmb.data['Wave/freq'])
    ref = get_reflected(no_atm_cmb,atm_therm,planet_name)
    assert isinstance(ref,u.Quantity)
    assert len(ref) == len(atm_cmb.data['Wave/freq'])

    hires = PSGrad.from_rad(data_dir / 'hires_cmb.rad')
    with pytest.raises(ValueError):
        get_reflected(hires,atm_therm,planet_name)

@pytest.mark.skipif(not is_port_in_use(3000),reason='PSG must be running locally to run this test')
def test_text_parse():
    previous_state = is_port_in_use(PSG_PORT)
    set_psg_state(True)
    psg_url = 'http://localhost:3000'
    with open(PSG_CONFIG_PATH,'r',encoding='UTF-8') as file:
        file_contents = file.read()
    content = call_api(psg_url=psg_url,output_type='all',config_data=file_contents)
    result = parse_full_output(content)
    assert b'cfg' in result.keys()
    set_psg_state(previous_state)
