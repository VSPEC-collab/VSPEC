import numpy as np
import pytest
from pathlib import Path
from astropy import units as u

from VSPEC.gcm.decoder import GCMdecoder

# Test data
header = '10,10,10,180,-90,18,9,O2,H2O,CO2'
dat = np.random.rand(3000)
config = 'test_file.config'


def test_get_shape():
    decoder = GCMdecoder(header, dat)
    assert decoder.get_shape() == (10, 10, 10)

def test_get_lats():
    decoder = GCMdecoder(header, dat)
    lats = decoder.get_lats()
    assert len(lats) == 10
    assert np.allclose(lats, np.arange(10) * float(decoder.header.split(',')[6]) + float(decoder.header.split(',')[4]))

def test_get_lons():
    decoder = GCMdecoder(header, dat)
    lons = decoder.get_lons()
    assert len(lons) == 10
    assert np.allclose(lons, np.arange(10) * float(decoder.header.split(',')[5]) + float(decoder.header.split(',')[3]))

def test_get_molecules():
    decoder = GCMdecoder(header, dat)
    molecs = decoder.get_molecules()
    assert len(molecs) == 3
    assert 'O2' in molecs
    assert 'H2O' in molecs
    assert 'CO2' in molecs

def test_get_aerosols():
    decoder = GCMdecoder(header, dat)
    decoder.header += ',Aerosol1,Aerosol1_size,Aerosol2,Aerosol2_size'
    aerosols, aerosol_sizes = decoder.get_aerosols()
    assert len(aerosols) == 2
    assert 'Aerosol1' in aerosols
    assert 'Aerosol2' in aerosols
    assert len(aerosol_sizes) == 2
    assert 'Aerosol1_size' in aerosol_sizes
    assert 'Aerosol2_size' in aerosol_sizes

def test_rename_var():
    decoder = GCMdecoder(header, dat)
    decoder.rename_var('O2', 'NewVar')
    assert 'NewVar' in decoder.header

def test_getitem():
    decoder = GCMdecoder(header, dat)
    var = decoder['O2']
    assert isinstance(var, np.ndarray)

def test_setitem():
    decoder = GCMdecoder(header, dat)
    decoder['O2'] = np.random.rand(1000).reshape(10,10,10)
    var = decoder['O2']
    assert isinstance(var, np.ndarray)

def test_remove():
    decoder = GCMdecoder(header, dat)
    decoder.remove('O2')
    assert not np.any([var=='O2' for var in decoder.header.split(',')])
    assert len(decoder.dat)==2000
