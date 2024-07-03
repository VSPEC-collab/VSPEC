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

EMPTY_DIR = Path(__file__).parent / 'data' / 'empty'

def test_init(test1_data:PhaseAnalyzer):
    """
    Test the `__init__` method of `VSPEC.PhaseAnalyzer`
    """
    assert isinstance(test1_data._observation_data, QTable)
    assert isinstance(test1_data.n_images, int)
    assert isinstance(test1_data.time, u.Quantity)
    assert isinstance(test1_data.phase, u.Quantity)
    assert isinstance(test1_data.unique_phase, u.Quantity)
    assert isinstance(test1_data.wavelength, u.Quantity)
    assert isinstance(test1_data.star, u.Quantity)
    assert isinstance(test1_data.reflected, u.Quantity)
    assert isinstance(test1_data.thermal, u.Quantity)
    assert isinstance(test1_data.total, u.Quantity)
    assert isinstance(test1_data.noise, u.Quantity)
    assert isinstance(test1_data.layers, dict)

    assert np.all(test1_data.phase >= 0 * u.deg)
    assert np.all(test1_data.phase <= 360 * u.deg)

    assert np.all(np.diff(test1_data.unique_phase) > 0*u.deg)

    assert test1_data.wavelength.unit.physical_type == u.um.physical_type

    assert test1_data.star.unit == u.W / (u.m ** 2 * u.um)
    assert test1_data.reflected.unit == u.W / (u.m ** 2 * u.um)
    assert test1_data.thermal.unit == u.W / (u.m ** 2 * u.um)
    assert test1_data.total.unit == u.W / (u.m ** 2 * u.um)
    assert test1_data.noise.unit == u.W / (u.m ** 2 * u.um)

    for hdu in test1_data.layers.keys():
        dat = test1_data.get_layer(hdu)
        assert isinstance(dat,u.Quantity)
    with pytest.raises(KeyError):
        test1_data.get_layer('fake_variable')

def test_init_wrong_path():
    """
    Test `PhaseAnalyzer()` when given path to empty directory.
    """
    with pytest.raises(FileNotFoundError):
        PhaseAnalyzer(EMPTY_DIR)

def test_lightcurve(test1_data:PhaseAnalyzer):
    """
    Test `PhaseAnalyzer.lightcurve()`
    """
    assert test1_data.lightcurve('total', 0).shape == (test1_data.n_images,)
    assert test1_data.lightcurve('star', 0).shape == (test1_data.n_images,)
    assert test1_data.lightcurve('reflected', 0).shape == (test1_data.n_images,)
    assert test1_data.lightcurve('thermal', 0).shape == (test1_data.n_images,)
    assert test1_data.lightcurve('noise', 0).shape == (test1_data.n_images,)
    assert test1_data.lightcurve('total', len(
        test1_data.wavelength)//2).shape == (test1_data.n_images,)
    assert test1_data.lightcurve('total', -1).shape == (test1_data.n_images,)
    assert test1_data.lightcurve('total', (0, -1)).shape == (test1_data.n_images,)
    assert test1_data.lightcurve(
        'total', 0, normalize=0).shape == (test1_data.n_images,)
    assert test1_data.lightcurve(
        'total', (0, -1), normalize=0).shape == (test1_data.n_images,)
    assert test1_data.lightcurve(
        'total', 0, noise=True).shape == (test1_data.n_images,)
    assert test1_data.lightcurve(
        'total', (0, -1), noise=True).shape == (test1_data.n_images,)

    assert np.all(test1_data.lightcurve('total', 0, normalize='max') <= 1)

    assert test1_data.lightcurve('total', 0).unit == test1_data.total.unit
    with pytest.raises(AttributeError):
        getattr(test1_data.lightcurve('total', 0, normalize=0), 'unit')
    tol = test1_data.total[0, :]*1e-10
    assert np.all(isclose(test1_data.lightcurve(
        'total', 0), test1_data.total[0, :], tol))

def test_spectrum(test1_data:PhaseAnalyzer):
    """
    Test `PhaseAnalyzer.spectrum()`
    """
    result = test1_data.spectrum('total', 0, noise=False)
    assert isinstance(result, u.Quantity)
    assert result.shape == (len(test1_data.wavelength),)

    # Test with noise=True
    result = test1_data.spectrum('total', 0, noise=True)
    assert isinstance(result, u.Quantity)
    assert result.shape == (len(test1_data.wavelength),)

    # Test with noise=0.1
    result = test1_data.spectrum('total', 0, noise=0.1)
    assert isinstance(result, u.Quantity)
    assert result.shape == (len(test1_data.wavelength),)

    # Test with images=(1,3)
    result = test1_data.spectrum('total', (0, -1), noise=False)
    assert isinstance(result, u.Quantity)
    assert result.shape == (len(test1_data.wavelength),)

    # Test with source='noise'
    if test1_data.n_images > 1:
        epoch0 = test1_data.spectrum('noise', 0)
        epoch1 = test1_data.spectrum('noise', 1)
        multi_epoch = test1_data.spectrum('noise', (0, 2))
        tol = multi_epoch * 1e-2
        assert np.all(isclose(multi_epoch, 0.5 *
                      np.sqrt(epoch0**2+epoch1**2), tol))





if __name__ in '__main__':
    pytest.main(args=[Path(__file__)])