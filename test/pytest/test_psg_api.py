#!/usr/bin/env python

"""
Tests for `VSPEC.psg_api` module
"""
from pathlib import Path
import pytest
from astropy import units as u
from pypsg import PyRad

from VSPEC.psg_api import get_reflected

API_KEY_PATH = Path.home() / 'psg_key.txt'
PSG_CONFIG_PATH = Path(__file__).parent / 'data' / 'test_cfg.txt'
VSPEC_CONFIG_PATH = Path(__file__).parent / 'default.yaml'
PSG_PORT = 3000

RAD_PATH = Path(__file__).parent / 'data' / 'transit_reflected'

def test_pyrad():
    """
    Run tests for `VSPEC.psg_api.PSGrad()`
    """
    file = Path(__file__).parent / 'data' / 'test_rad.rad'
    rad = PyRad.from_bytes(file.read_bytes())
    assert isinstance(rad,PyRad)

def test_get_reflected():
    """
    Run tests for `VSPEC.psg_api.get_reflected()`
    """
    data_dir = Path(__file__).parent / 'data' / 'test_reflected'
    planet_name = 'proxima-Cen-b'

    atm_cmb = PyRad.from_bytes((data_dir / 'atm_cmb.rad').read_bytes())
    atm_therm = PyRad.from_bytes((data_dir / 'atm_therm.rad').read_bytes())
    ref = get_reflected(atm_cmb,atm_therm,planet_name)
    assert isinstance(ref,u.Quantity)
    assert len(ref) == len(atm_cmb.wl)

    no_atm_cmb = PyRad.from_bytes((data_dir / 'no_atm_cmb.rad').read_bytes())
    no_atm_therm = PyRad.from_bytes((data_dir / 'no_atm_therm.rad').read_bytes())
    ref = get_reflected(no_atm_cmb,no_atm_therm,planet_name)
    assert isinstance(ref,u.Quantity)
    assert len(ref) == len(atm_cmb.wl)

    # also works if just one has separated thermal+reflected columns
    ref = get_reflected(atm_cmb,no_atm_therm,planet_name)
    assert isinstance(ref,u.Quantity)
    assert len(ref) == len(atm_cmb.wl)
    ref = get_reflected(no_atm_cmb,atm_therm,planet_name)
    assert isinstance(ref,u.Quantity)
    assert len(ref) == len(atm_cmb.wl)

    hires = PyRad.from_bytes((data_dir / 'hires_cmb.rad').read_bytes())
    with pytest.raises(ValueError):
        get_reflected(hires,atm_therm,planet_name)
