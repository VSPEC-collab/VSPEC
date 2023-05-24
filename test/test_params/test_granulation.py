"""
GranulationParameters tests
"""
from VSPEC.params.stellar import GranulationParameters
from astropy import units as u
import pytest

def test_preset_none():
    params = GranulationParameters.none()
    assert params.mean == 0.0

def test_preset_std():
    params = GranulationParameters.std()
    assert params.mean == 0.2

def test_custom_values():
    params = GranulationParameters(
        mean=0.4,
        amp=0.02,
        period=10 * u.day,
        dteff=300 * u.K
    )
    assert params.mean == 0.4

def test_invalid_mean():
    with pytest.raises(ValueError):
        _ = GranulationParameters(
            mean=1.5,
            amp=0.02,
            period=10 * u.day,
            dteff=300 * u.K
        )

def test_invalid_amp():
    with pytest.raises(ValueError):
        _ = GranulationParameters(
            mean=0.4,
            amp=-0.1,
            period=10 * u.day,
            dteff=300 * u.K
        )

def test_accessing_attributes():
    params = GranulationParameters.std()
    assert params.mean == 0.2
    assert params.period == 5 * u.day

def test_custom_from_dict():
    params_dict = {
        'mean': 0.4,
        'amp': 0.02,
        'period': 10 * u.day,
        'dteff': 300 * u.K
    }
    params = GranulationParameters.from_dict(params_dict)
    assert params.mean == 0.4

def test_preset_from_dict():
    params_dict = {
        'preset': 'std'
    }
    params = GranulationParameters.from_dict(params_dict)
    assert params.mean == 0.2
    assert params.period == 5 * u.day
