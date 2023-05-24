"""
FaculaParameters tests
"""
from VSPEC.params.stellar import FaculaParameters
from astropy import units as u
import pytest

def test_preset_none():
    params = FaculaParameters.none()
    assert params.distribution == 'iso'

def test_preset_std():
    params = FaculaParameters.std()
    assert params.distribution == 'iso'

def test_custom_values():
    params = FaculaParameters(
        distribution='iso',
        equillibrium_coverage=0.2,
        warmup=30 * u.hour,
        mean_radius=500 * u.km,
        hwhm_radius=200 * u.km,
        mean_timescale=10 * u.hour,
        hwhm_timescale=4 * u.hour
    )
    assert params.distribution == 'iso'

def test_invalid_distribution():
    with pytest.raises(ValueError):
        _ = FaculaParameters(
            distribution='invalid',
            equillibrium_coverage=0.2,
            warmup=30 * u.hour,
            mean_radius=500 * u.km,
            hwhm_radius=200 * u.km,
            mean_timescale=10 * u.hour,
            hwhm_timescale=4 * u.hour
        )

def test_invalid_equilibrium_coverage():
    with pytest.raises(ValueError):
        _ = FaculaParameters(
            distribution='iso',
            equillibrium_coverage=1.2,
            warmup=30 * u.hour,
            mean_radius=500 * u.km,
            hwhm_radius=200 * u.km,
            mean_timescale=10 * u.hour,
            hwhm_timescale=4 * u.hour
        )
    
def test_accessing_attributes():
    params = FaculaParameters.std()
    assert params.distribution == 'iso'
    assert params.mean_radius == 500 * u.km

def test_custom_from_dict():
    params_dict = {
        'distribution': 'iso',
        'equillibrium_coverage': 0.2,
        'warmup': 30 * u.hour,
        'mean_radius': 500 * u.km,
        'hwhm_radius': 200 * u.km,
        'mean_timescale': 10 * u.hour,
        'hwhm_timescale': 4 * u.hour
    }
    params = FaculaParameters.from_dict(params_dict)
    assert params.distribution == 'iso'
    assert params.mean_radius == 500 * u.km

def test_preset_from_dict():
    params_dict = {
        'preset':'std'
    }
    params = FaculaParameters.from_dict(params_dict)
    assert params.distribution == 'iso'
    assert params.mean_radius == 500 * u.km
