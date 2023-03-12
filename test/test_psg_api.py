#!/usr/bin/env python

"""
Tests for `VSPEC.psg_api` module
"""
from pathlib import Path
import pytest

from VSPEC.psg_api import call_api, write_static_config, PSGrad
from VSPEC.helpers import is_port_in_use
from VSPEC.read_info import ParamModel

API_KEY_PATH = Path.home() / 'psg_key.txt'
PSG_CONFIG_PATH = Path(__file__).parent / 'data' / 'test_cfg.txt'
VSPEC_CONFIG_PATH = Path(__file__).parent / 'default.cfg'



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
    outfile = Path('test.rad')
    call_api(PSG_CONFIG_PATH,psg_url,api_key,output_type='rad',outfile=outfile)
    try:
        assert outfile.exists()
    except Exception as exc:
        outfile.unlink()
        raise exc
    outfile.unlink()
@pytest.mark.skipif(not is_port_in_use(3000),reason='PSG must be running locally to run this test')
def test_call_api_local():
    """
    Run tests for `VSPEC.psg_api.call_api()`
    """
    psg_url = 'http://localhost:3000'
    outfile = Path('test.rad')
    call_api(PSG_CONFIG_PATH,psg_url,output_type='rad',outfile=outfile)
    try:
        assert outfile.exists()
    except Exception as exc:
        outfile.unlink()
        raise exc
    outfile.unlink()

def test_write_static_config():
    """
    Run tests for `VSPEC.psg_api.write_static_config()`
    """
    params = ParamModel(VSPEC_CONFIG_PATH)
    outfile = Path('test_cfg.txt')
    try:
        write_static_config(outfile,params,'w')
    except Exception as exc:
        outfile.unlink()
        raise exc
    # if is_port_in_use(3000):
    #     psg_url = 'http://localhost:3000'
    #     call_api(outfile,psg_url,output_type='rad')
    outfile.unlink()

def test_PSGrad():
    file = Path(__file__).parent / 'data' / 'test_rad.rad'
    rad = PSGrad.from_rad(file)
    assert isinstance(rad.header,dict)
    assert isinstance(rad.data,dict)
    with pytest.raises(ValueError):
        file = Path(__file__).parent / 'data' / 'gcm_error.rad'
        PSGrad.from_rad(file)





if __name__ in '__main__':
    if API_KEY_PATH.exists():
        test_call_api_nonlocal()
    if is_port_in_use(3000):
        test_call_api_local()
    test_write_static_config()
    test_PSGrad()
    
