"""
FlareParameters tests
"""
from VSPEC.params.stellar import FlareParameters
from astropy import units as u

def test_preset_none():
    params = FlareParameters.none()

def test_preset_std():
    params = FlareParameters.std()

def test_custom_values():
    params = FlareParameters(
        dist_teff_mean=9000*u.K,
        dist_teff_sigma=500*u.K,
        dist_fwhm_mean=8*u.hr,
        dist_fwhm_logsigma=0.2,
        alpha=-0.8,
        beta=27.0,
        min_energy=1e33*u.erg,
        cluster_size=2
    )

def test_custom_from_dict():
    params_dict = {
        'dist_teff_mean': '9000 K',
        'dist_teff_sigma': '500 K',
        'dist_fwhm_mean': '8 hr',
        'dist_fwhm_logsigma': '0.2',
        'alpha': '0.8',
        'beta': '27',
        'min_energy': '1e33 erg',
        'cluster_size': '2'
        }
    params = FlareParameters.from_dict(params_dict)

def test_preset_from_dict():
    params_dict = {
        'preset': 'std'
    }
    params = FlareParameters.from_dict(params_dict)
