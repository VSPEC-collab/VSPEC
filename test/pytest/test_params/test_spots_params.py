"""
SpotParameter tests
"""
import pytest

from VSPEC.params.stellar import SpotParameters
from astropy import units as u
from VSPEC.config import MSH

def test_preset_solar():
    params_dict = {'preset': 'solar'}
    params = SpotParameters.from_dict(params_dict)
    assert params.distribution == 'solar'
    params = SpotParameters.solar()
    assert params.distribution == 'solar'

def test_custom_dict():
    params_dict = {
        'distribution': 'iso',
        'initial_coverage': 0.2,
        'equillibrium_coverage': 0.1,
        'burn_in': 30 * u.day,
        'area_mean': 500 * MSH,
        'area_logsigma': 0.2,
        'teff_umbra': 2500 * u.K,
        'teff_penumbra': 2700 * u.K,
        'growth_rate': 0.52 / u.day,
        'decay_rate': 10.8 * MSH / u.day,
        'initial_area': 10 * MSH
    }
    params = SpotParameters.from_dict(params_dict)
    assert params.distribution == 'iso'

def test_preset_none():
    params = SpotParameters.none()
    assert params.distribution == 'iso'

def test_preset_mdwarf():
    params = SpotParameters.mdwarf()
    assert params.distribution == 'iso'    

def test_invalid_distribution():
    params_dict = {
        'distribution': 'invalid',
        'initial_coverage': 0.2,
        'equillibrium_coverage': 0.1,
        'burn_in': 30 * u.day,
        'area_mean': 500 * MSH,
        'area_logsigma': 0.2,
        'teff_umbra': 2500 * u.K,
        'teff_penumbra': 2700 * u.K,
        'growth_rate': 0.52 / u.day,
        'decay_rate': 10.8 * MSH / u.day,
        'initial_area': 10 * MSH
    }
    with pytest.raises(ValueError):
        _ = SpotParameters.from_dict(params_dict)

def test_invalid_initial_coverage():
    params_dict = {
        'distribution': 'iso',
        'initial_coverage': -0.2,
        'equillibrium_coverage': 0.1,
        'burn_in': 30 * u.day,
        'area_mean': 500 * MSH,
        'area_logsigma': 0.2,
        'teff_umbra': 2500 * u.K,
        'teff_penumbra': 2700 * u.K,
        'growth_rate': 0.52 / u.day,
        'decay_rate': 10.8 * MSH / u.day,
        'initial_area': 10 * MSH
    }
    with pytest.raises(ValueError):
        _ = SpotParameters.from_dict(params_dict)

def test_accessing_attributes():
    params = SpotParameters.solar()
    assert params.initial_coverage == 0.1
    assert params.teff_umbra == 2500 * u.K


