"""
FlareParameters tests
"""
from VSPEC.params.stellar import FlareParameters
from astropy import units as u

def test_preset_none():
    params = FlareParameters.none()
    assert params.group_probability == 0.5

def test_preset_std():
    params = FlareParameters.std()
    assert params.group_probability == 0.5

def test_custom_values():
    params = FlareParameters(
        group_probability=0.2,
        dist_teff_mean=8000 * u.K,
        dist_teff_sigma=300 * u.K,
        fwhm_mean=0.1 * u.day,
        fwhm_sigma=0.2,
        E_min=10 ** 33 * u.erg,
        E_max=10 ** 35 * u.erg,
        E_steps=50
    )
    assert params.group_probability == 0.2

def test_custom_from_dict():
    params_dict = {
        'group_probability': 0.2,
        'teff_mean': 8000 * u.K,
        'teff_sigma': 300 * u.K,
        'fwhm_mean': 0.1 * u.day,
        'fwhm_sigma': 0.2,
        'E_min': 10 ** 33 * u.erg,
        'E_max': 10 ** 35 * u.erg,
        'E_steps': 50
    }
    params = FlareParameters.from_dict(params_dict)
    assert params.group_probability == 0.2

def test_preset_from_dict():
    params_dict = {
        'preset': 'std'
    }
    params = FlareParameters.from_dict(params_dict)
    assert params.group_probability == 0.5
