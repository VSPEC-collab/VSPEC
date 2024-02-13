"""
StarParameters tests
"""
from VSPEC.params.stellar import StarParameters, LimbDarkeningParameters, SpotParameters, FaculaParameters, FlareParameters, GranulationParameters
from astropy import units as u

def test_preset_static_proxima():
    params = StarParameters.static_proxima()
    assert params.psg_star_template == 'M'
    assert params.mass == 0.12 * u.M_sun

def test_preset_spotted_proxima():
    params = StarParameters.spotted_proxima()
    assert params.psg_star_template == 'M'
    assert params.spots.distribution == 'iso'

def test_preset_flaring_proxima():
    params = StarParameters.flaring_proxima()
    assert params.psg_star_template == 'M'

def test_preset_proxima():
    params = StarParameters.proxima()
    assert params.psg_star_template == 'M'
    assert params.granulation.mean == 0.2

def test_custom_values():
    params = StarParameters(
        psg_star_template='K',
        teff=4000 * u.K,
        mass=0.8 * u.M_sun,
        radius=0.6 * u.R_sun,
        period=25 * u.day,
        misalignment=10 * u.deg,
        misalignment_dir=30 * u.deg,
        ld=LimbDarkeningParameters.lambertian(),
        spots=SpotParameters.none(),
        faculae=FaculaParameters.none(),
        flares=FlareParameters.std(),
        granulation=GranulationParameters.none(),
        grid_params=(200,400),
        spectral_grid='bb'
    )
    assert params.psg_star_template == 'K'
    assert params.mass == 0.8 * u.M_sun

def test_accessing_attributes():
    params = StarParameters.spotted_proxima()
    assert params.spots.distribution == 'iso'
    assert params.psg_star_template == 'M'

def test_custom_from_dict():
    params_dict = {
        'psg_star_template': 'G',
        'teff': 5500 * u.K,
        'mass': 1.2 * u.M_sun,
        'radius': 1.1 * u.R_sun,
        'period': 30 * u.day,
        'misalignment': 5 * u.deg,
        'misalignment_dir': 45 * u.deg,
        'ld': {'preset':'lambertian'},
        'spots': {'preset':'none'},
        'faculae': {'preset':'none'},
        'flares': {'preset':'none'},
        'granulation': {'preset':'none'},
        'grid_params': (200,400),
        'spectral_grid': 'default'
    }
    params = StarParameters.from_dict(params_dict)
    assert params.psg_star_template == 'G'
    assert params.mass == 1.2 * u.M_sun

def test_preset_from_dict():
    params_dict = {
        'preset': 'proxima'
    }
    params = StarParameters.from_dict(params_dict)
    assert params.psg_star_template == 'M'
