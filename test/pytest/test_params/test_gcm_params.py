"""
Tests for GCM parameters
"""
from pathlib import Path
import pytest
from astropy import units as u

from libpypsg.globes import PyGCM
from libpypsg.globes.waccm.waccm import TEST_PATH, download_test_data

from VSPEC.params.gcm import gcmParameters


@pytest.fixture
def waccm_path():
    if not TEST_PATH.exists():
        download_test_data()
    return TEST_PATH


def test_gcmParameters_from_dict_waccm(waccm_path: Path):
    # Create a dictionary representation of the gcmParameters instance with waccmGCM

    gcm_dict = {
        'star': None,
        'planet': None,
        'gcm': {
            'waccm': {
                'path': str(waccm_path.resolve()),
                'tstart': '4630 day',
                'molecules': ['O2', 'CO2'],
                'aerosols': [],
                'background': 'N2'
            },
            'mean_molec_weight': '28'
        }
    }

    # Create a gcmParameters instance from the dictionary
    gcm_params = gcmParameters.from_dict(gcm_dict)

    # Assert that the gcm attribute is an instance of waccmGCM
    assert not gcm_params.is_staic
    assert isinstance(gcm_params.get_gcm(0*u.day), PyGCM)



def test_gcmParameters_from_dict_invalid():
    # Create an invalid dictionary representation without 'binary' or 'waccm' keys
    gcm_dict = {
        'star': None,
        'planet': None,
        'gcm': {
            'invalid': {
                'path': '/path/to/waccm.nc',
                'tstart': '100',
                'molecules': 'O2, CO2',
                'aerosols': 'Water, WaterIce',
                'background': 'N2'
            },
            'mean_molec_weight': '28'
        }
    }

    # Test that KeyError is raised when constructing gcmParameters from the invalid dictionary
    with pytest.raises(KeyError):
        gcmParameters.from_dict(gcm_dict)


if __name__ == "__main__":
    pytest.main(args=[Path(__file__)])
