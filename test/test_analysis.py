#!/usr/bin/env python

"""
Tests for `VSPEC.analysis` module
"""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from astropy import units as u
import xarray
from VSPEC import PhaseAnalyzer
from VSPEC.analysis import read_lyr
from VSPEC.helpers import isclose

DATA_DIR = Path(__file__).parent / 'data' / 'test_analysis'
EMPTY_DIR = Path(__file__).parent / 'data' / 'empty'


def test_init():
    """
    Test the `__init__` method of `VSPEC.PhaseAnalyzer`
    """
    path = DATA_DIR
    analyzer = PhaseAnalyzer(path)
    assert isinstance(analyzer.observation_data, pd.DataFrame)
    assert isinstance(analyzer.N_images, int)
    assert isinstance(analyzer.time, u.Quantity)
    assert isinstance(analyzer.phase, u.Quantity)
    assert isinstance(analyzer.unique_phase, u.Quantity)
    assert isinstance(analyzer.wavelength, u.Quantity)
    assert isinstance(analyzer.star, u.Quantity)
    assert isinstance(analyzer.reflected, u.Quantity)
    assert isinstance(analyzer.thermal, u.Quantity)
    assert isinstance(analyzer.total, u.Quantity)
    assert isinstance(analyzer.noise, u.Quantity)
    assert isinstance(analyzer.layers, xarray.DataArray)

    assert np.all(analyzer.phase >= 0 * u.deg)
    assert np.all(analyzer.phase <= 360 * u.deg)

    assert np.all(np.diff(analyzer.unique_phase) > 0*u.deg)

    assert analyzer.wavelength.unit.physical_type == u.um.physical_type

    assert analyzer.star.unit == u.W / (u.m ** 2 * u.um)
    assert analyzer.reflected.unit == u.W / (u.m ** 2 * u.um)
    assert analyzer.thermal.unit == u.W / (u.m ** 2 * u.um)
    assert analyzer.total.unit == u.W / (u.m ** 2 * u.um)
    assert analyzer.noise.unit == u.W / (u.m ** 2 * u.um)


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
    path = DATA_DIR
    with pytest.raises(u.UnitConversionError):
        PhaseAnalyzer(path, fluxunit=u.s)


def test_lightcurve():
    """
    Test `PhaseAnalyzer.lightcurve()`
    """
    path = DATA_DIR
    analyzer = PhaseAnalyzer(path)

    assert analyzer.lightcurve('total', 0).shape == (analyzer.N_images,)
    assert analyzer.lightcurve('star', 0).shape == (analyzer.N_images,)
    assert analyzer.lightcurve('reflected', 0).shape == (analyzer.N_images,)
    assert analyzer.lightcurve('thermal', 0).shape == (analyzer.N_images,)
    assert analyzer.lightcurve('noise', 0).shape == (analyzer.N_images,)
    assert analyzer.lightcurve('total', len(
        analyzer.wavelength)//2).shape == (analyzer.N_images,)
    assert analyzer.lightcurve('total', -1).shape == (analyzer.N_images,)
    assert analyzer.lightcurve('total', (0, -1)).shape == (analyzer.N_images,)
    assert analyzer.lightcurve(
        'total', 0, normalize=0).shape == (analyzer.N_images,)
    assert analyzer.lightcurve(
        'total', (0, -1), normalize=0).shape == (analyzer.N_images,)
    assert analyzer.lightcurve(
        'total', 0, noise=True).shape == (analyzer.N_images,)
    assert analyzer.lightcurve(
        'total', (0, -1), noise=True).shape == (analyzer.N_images,)

    assert np.all(analyzer.lightcurve('total', 0, normalize='max') <= 1)

    assert analyzer.lightcurve('total', 0).unit == analyzer.total.unit
    with pytest.raises(AttributeError):
        getattr(analyzer.lightcurve('total', 0, normalize=0), 'unit')
    tol = analyzer.total[0, :]*1e-10
    assert np.all(isclose(analyzer.lightcurve(
        'total', 0), analyzer.total[0, :], tol))


def test_spectrum():
    """
    Test `PhaseAnalyzer.spectrum()`
    """
    path = DATA_DIR
    pa = PhaseAnalyzer(path)
    result = pa.spectrum('total', 0, noise=False)
    assert isinstance(result, u.Quantity)
    assert result.shape == (len(pa.wavelength),)

    # Test with noise=True
    result = pa.spectrum('total', 0, noise=True)
    assert isinstance(result, u.Quantity)
    assert result.shape == (len(pa.wavelength),)

    # Test with noise=0.1
    result = pa.spectrum('total', 0, noise=0.1)
    assert isinstance(result, u.Quantity)
    assert result.shape == (len(pa.wavelength),)

    # Test with images=(1,3)
    result = pa.spectrum('total', (0, -1), noise=False)
    assert isinstance(result, u.Quantity)
    assert result.shape == (len(pa.wavelength),)

    # Test with source='noise'
    if pa.N_images > 1:
        epoch0 = pa.spectrum('noise', 0)
        epoch1 = pa.spectrum('noise', 1)
        multi_epoch = pa.spectrum('noise', (0, 2))
        tol = multi_epoch * 1e-2
        assert np.all(isclose(multi_epoch, 0.5 *
                      np.sqrt(epoch0**2+epoch1**2), tol))


def test_read_lyr():
    file = DATA_DIR / 'layer00000.csv'
    fake_file = EMPTY_DIR / 'layer00000.csv'
    wrong_file = DATA_DIR / 'phase00000.csv'

    with pytest.raises(FileNotFoundError):
        read_lyr(fake_file)
    with pytest.raises(ValueError):
        read_lyr(wrong_file)
    data = read_lyr(file)
    assert isinstance(data, pd.DataFrame)


if __name__ in '__main__':
    test_init()
    test_init_wrong_path()
    test_init_wrong_unit()
    test_lightcurve()
    test_spectrum()
