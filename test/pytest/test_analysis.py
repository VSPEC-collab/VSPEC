#!/usr/bin/env python

"""
Tests for `VSPEC.analysis` module
"""
from pathlib import Path
import pytest
import numpy as np
from astropy import units as u
from astropy.table import QTable

from VSPEC import PhaseAnalyzer
from VSPEC.helpers import isclose

TEST1_DIR = Path(__file__).parent.parent / 'end_to_end_tests' / 'test1' / '.vspec' / 'test1' / 'AllModelSpectraValues'
EMPTY_DIR = Path(__file__).parent / 'data' / 'empty'

def test_init():
    """
    Test the `__init__` method of `VSPEC.PhaseAnalyzer`
    """
    path = TEST1_DIR
    data = PhaseAnalyzer(path)
    assert isinstance(data.observation_data, QTable)
    assert isinstance(data.N_images, int)
    assert isinstance(data.time, u.Quantity)
    assert isinstance(data.phase, u.Quantity)
    assert isinstance(data.unique_phase, u.Quantity)
    assert isinstance(data.wavelength, u.Quantity)
    assert isinstance(data.star, u.Quantity)
    assert isinstance(data.reflected, u.Quantity)
    assert isinstance(data.thermal, u.Quantity)
    assert isinstance(data.total, u.Quantity)
    assert isinstance(data.noise, u.Quantity)
    assert isinstance(data.layers, dict)

    assert np.all(data.phase >= 0 * u.deg)
    assert np.all(data.phase <= 360 * u.deg)

    assert np.all(np.diff(data.unique_phase) > 0*u.deg)

    assert data.wavelength.unit.physical_type == u.um.physical_type

    assert data.star.unit == u.W / (u.m ** 2 * u.um)
    assert data.reflected.unit == u.W / (u.m ** 2 * u.um)
    assert data.thermal.unit == u.W / (u.m ** 2 * u.um)
    assert data.total.unit == u.W / (u.m ** 2 * u.um)
    assert data.noise.unit == u.W / (u.m ** 2 * u.um)

    for hdu in data.layers.keys():
        dat = data.get_layer(hdu)
        assert isinstance(dat,u.Quantity)
    with pytest.raises(KeyError):
        data.get_layer('fake_variable')

def test_init_wrong_path():
    """
    Test `PhaseAnalyzer()` when given path to empty directory.
    """
    with pytest.raises(FileNotFoundError):
        PhaseAnalyzer(EMPTY_DIR)

def test_init_wrong_unit():
    """
    Test the `__init__` method of `VSPEC.PhaseAnalyzer`
    with a non-physical value for `fluxunit`
    """
    path = TEST1_DIR
    with pytest.raises(u.UnitConversionError):
        PhaseAnalyzer(path, fluxunit=u.s)


def test_lightcurve():
    """
    Test `PhaseAnalyzer.lightcurve()`
    """
    path = TEST1_DIR
    data = PhaseAnalyzer(path)

    assert data.lightcurve('total', 0).shape == (data.N_images,)
    assert data.lightcurve('star', 0).shape == (data.N_images,)
    assert data.lightcurve('reflected', 0).shape == (data.N_images,)
    assert data.lightcurve('thermal', 0).shape == (data.N_images,)
    assert data.lightcurve('noise', 0).shape == (data.N_images,)
    assert data.lightcurve('total', len(
        data.wavelength)//2).shape == (data.N_images,)
    assert data.lightcurve('total', -1).shape == (data.N_images,)
    assert data.lightcurve('total', (0, -1)).shape == (data.N_images,)
    assert data.lightcurve(
        'total', 0, normalize=0).shape == (data.N_images,)
    assert data.lightcurve(
        'total', (0, -1), normalize=0).shape == (data.N_images,)
    assert data.lightcurve(
        'total', 0, noise=True).shape == (data.N_images,)
    assert data.lightcurve(
        'total', (0, -1), noise=True).shape == (data.N_images,)

    assert np.all(data.lightcurve('total', 0, normalize='max') <= 1)

    assert data.lightcurve('total', 0).unit == data.total.unit
    with pytest.raises(AttributeError):
        getattr(data.lightcurve('total', 0, normalize=0), 'unit')
    tol = data.total[0, :]*1e-10
    assert np.all(isclose(data.lightcurve(
        'total', 0), data.total[0, :], tol))

def test_spectrum():
    """
    Test `PhaseAnalyzer.spectrum()`
    """
    path = TEST1_DIR
    data = PhaseAnalyzer(path)
    result = data.spectrum('total', 0, noise=False)
    assert isinstance(result, u.Quantity)
    assert result.shape == (len(data.wavelength),)

    # Test with noise=True
    result = data.spectrum('total', 0, noise=True)
    assert isinstance(result, u.Quantity)
    assert result.shape == (len(data.wavelength),)

    # Test with noise=0.1
    result = data.spectrum('total', 0, noise=0.1)
    assert isinstance(result, u.Quantity)
    assert result.shape == (len(data.wavelength),)

    # Test with images=(1,3)
    result = data.spectrum('total', (0, -1), noise=False)
    assert isinstance(result, u.Quantity)
    assert result.shape == (len(data.wavelength),)

    # Test with source='noise'
    if data.N_images > 1:
        epoch0 = data.spectrum('noise', 0)
        epoch1 = data.spectrum('noise', 1)
        multi_epoch = data.spectrum('noise', (0, 2))
        tol = multi_epoch * 1e-2
        assert np.all(isclose(multi_epoch, 0.5 *
                      np.sqrt(epoch0**2+epoch1**2), tol))





if __name__ in '__main__':
    pytest.main(args=[Path(__file__)])