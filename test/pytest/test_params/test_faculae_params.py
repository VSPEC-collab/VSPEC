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
        burn_in=30 * u.hour,
        mean_radius=500 * u.km,
        logsigma_radius=0.2,
        mean_timescale=10 * u.hour,
        logsigma_timescale=0.2,
        depth=100*u.km,
        floor_teff_slope=0*u.K/u.km,
        floor_teff_min_rad=20*u.km,
        floor_teff_base_dteff=-100*u.K,
        wall_teff_intercept=100*u.K,
        wall_teff_slope=0*u.K/u.km
    )
    assert params.distribution == 'iso'

def test_invalid_distribution():
    with pytest.raises(ValueError):
        _ = FaculaParameters(
            distribution='invalid',
            equillibrium_coverage=0.2,
            burn_in=30 * u.hour,
            mean_radius=500 * u.km,
            logsigma_radius=0.2,
            mean_timescale=10 * u.hour,
            logsigma_timescale=0.2,
            depth=100*u.km,
            floor_teff_slope=0*u.K/u.km,
            floor_teff_min_rad=20*u.km,
            floor_teff_base_dteff=-100*u.K,
            wall_teff_intercept=100*u.K,
            wall_teff_slope=0*u.K/u.km
        )

def test_invalid_equilibrium_coverage():
    with pytest.raises(ValueError):
        _ = FaculaParameters(
            distribution='iso',
            equillibrium_coverage=1.2,
            burn_in=30 * u.hour,
            mean_radius=500 * u.km,
            logsigma_radius=0.2,
            mean_timescale=10 * u.hour,
            logsigma_timescale=0.2,
            depth=100*u.km,
            floor_teff_slope=0*u.K/u.km,
            floor_teff_min_rad=20*u.km,
            floor_teff_base_dteff=-100*u.K,
            wall_teff_intercept=100*u.K,
            wall_teff_slope=0*u.K/u.km
        )
    
def test_accessing_attributes():
    params = FaculaParameters.std()
    assert params.distribution == 'iso'
    assert params.mean_radius == 800 * u.km

def test_custom_from_dict():
    params_dict = {
        'distribution': 'iso',
        'equillibrium_coverage': '0.2',
        'burn_in': '30 hour',
        'mean_radius': '500 km',
        'logsigma_radius': '0.2',
        'mean_timescale': '8 hour',
        'logsigma_timescale': '0.2',
        'depth': '100 km',
        'floor_teff_slope':'0 K km-1',
        'floor_teff_min_rad':'20 km',
        'floor_teff_base_dteff':'-100 K',
        'wall_teff_intercept':'100 K',
        'wall_teff_slope':'0 K km-1'
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
    assert params.mean_radius == 800 * u.km
