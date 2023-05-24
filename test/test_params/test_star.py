"""
StarParameters tests
"""
from VSPEC.params.stellar import StarParameters, LimbDarkeningParameters, SpotParameters, FaculaParameters, FlareParameters, GranulationParameters
from astropy import units as u
import pytest

def test_preset_static_proxima():
    params = StarParameters.static_proxima()
    assert params.template == 'M'
    assert params.mass == 0.12 * u.M_sun

def test_preset_spotted_proxima():
    params = StarParameters.spotted_proxima()
    assert params.template == 'M'
    assert params.spots.distribution == 'iso'

def test_preset_flaring_proxima():
    params = StarParameters.flaring_proxima()
    assert params.template == 'M'
    assert params.flares.E_steps == 100

def test_preset_proxima():
    params = StarParameters.proxima()
    assert params.template == 'M'
    assert params.granulation.mean == 0.2

def test_custom_values():
    params = StarParameters(
        template='K',
        teff=4000 * u.K,
        mass=0.8 * u.M_sun,
        radius=0.6 * u.R_sun,
        period=25 * u.day,
        offset_magnitude=10 * u.deg,
        offset_direction=30 * u.deg,
        ld=LimbDarkeningParameters.lambertian(),
        spots=SpotParameters.none(),
        faculae=FaculaParameters.none(),
        flares=FlareParameters.std(),
        granulation=GranulationParameters.none(),
        Nlat=200,
        Nlon=500
    )
    assert params.template == 'K'
    assert params.mass == 0.8 * u.M_sun

def test_accessing_attributes():
    params = StarParameters.spotted_proxima()
    assert params.spots.distribution == 'iso'
    assert params.template == 'M'

def test_custom_from_dict():
    params_dict = {
        'template': 'G',
        'teff': 5500 * u.K,
        'mass': 1.2 * u.M_sun,
        'radius': 1.1 * u.R_sun,
        'period': 30 * u.day,
        'offset_magnitude': 5 * u.deg,
        'offset_direction': 45 * u.deg,
        'ld': {'preset':'lambertian'},
        'spots': {'preset':'none'},
        'faculae': {'preset':'none'},
        'flares': {'preset':'none'},
        'granulation': {'preset':'none'},
        'Nlat': 300,
        'Nlon': 700
    }
    params = StarParameters.from_dict(params_dict)
    assert params.template == 'G'
    assert params.mass == 1.2 * u.M_sun

def test_preset_from_dict():
    params_dict = {
        'preset': 'proxima'
    }
    params = StarParameters.from_dict(params_dict)
    assert params.template == 'M'
    assert params.flares.E_steps == 100
