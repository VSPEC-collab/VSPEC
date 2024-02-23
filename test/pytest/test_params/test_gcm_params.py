"""
Tests for GCM parameters
"""
from pathlib import Path
import pytest
from astropy import units as u

from pypsg.globes.waccm.waccm import TEST_PATH,download_test_data

from VSPEC.params.gcm import binaryGCM, waccmGCM, gcmParameters

@pytest.fixture
def waccm_path():
    if not TEST_PATH.exists():
        download_test_data()
    return TEST_PATH

def test_binaryGCM_with_path():
    # Create a temporary GCM file
    gcm_path = Path("test.gcm")
    gcm_content = b"This is a test GCM file."
    with open(gcm_path, "wb") as file:
        file.write(gcm_content)

    # Instantiate binaryGCM with a path
    gcm = binaryGCM(path=gcm_path)

    # Check that the path and content are set correctly
    assert gcm.path == gcm_path
    assert gcm.content() == gcm_content

    # Clean up the temporary file
    gcm_path.unlink()

def test_binaryGCM_with_data():
    # Create test GCM data
    gcm_data = b"This is test GCM data."

    # Instantiate binaryGCM with data
    gcm = binaryGCM(data=gcm_data)

    # Check that the path is None and the content is set correctly
    assert gcm.path is None
    assert gcm.content() == gcm_data

def test_binaryGCM_without_data():
    # Attempt to instantiate binaryGCM without path or data
    with pytest.raises(ValueError):
        binaryGCM()

def test_binaryGCM_from_dict():
    # Create a dictionary representation of binaryGCM
    gcm_dict = {
        'data': "This is a test GCM."
    }

    # Instantiate binaryGCM from the dictionary
    gcm = binaryGCM.from_dict(gcm_dict)

    # Check that the attributes are set correctly
    assert gcm.path == None
    assert gcm.content() == b"This is a test GCM."

def test_waccmGCM_content(waccm_path):
    """
    Test creation from a WACCM file
    """

    tstart = 4621*u.day
    molecules = ['O2', 'CO2']
    background = 'N2'

    # Instantiate waccmGCM
    waccm = waccmGCM(waccm_path, tstart, molecules, None, background)

    # Define the observation time
    obs_time = 3*u.day

    # Get the content of the GCM for the specified observation time
    content = waccm.content(obs_time)
    assert b'<ATMOSPHERE-NGAS>3' in content

def test_waccmGCM_from_dict():
    # Create a dictionary representation of waccmGCM
    waccm_dict = {
        'path': '/path/to/netcdf',
        'tstart': '1 day',
        'molecules': 'O2, CO2',
        'aerosols': 'Water',
        'background': 'N2'
    }

    # Instantiate waccmGCM from the dictionary
    waccm = waccmGCM.from_dict(waccm_dict)

    # Check that the attributes are set correctly
    assert waccm.path == Path('/path/to/netcdf')
    assert waccm.tstart == 1*u.day
    assert waccm.molecules == ['O2', 'CO2']
    assert waccm.aerosols == ['Water']
    assert waccm.background == 'N2'


def test_gcmParameters_content_binary():
    # Create a binaryGCM instance
    binary_gcm = binaryGCM(data = b'GCM data')
    mmw = 28.
    
    # Create a gcmParameters instance with binaryGCM
    gcm_params = gcmParameters(gcm=binary_gcm,mean_molec_weight=mmw)
    
    # Test the content method
    content = gcm_params.content()
    
    # Assert that the content is not empty
    assert content == b'GCM data'

    assert gcm_params.to_psg()['ATMOSPHERE-WEIGHT'] == f'{mmw:.1f}'


def test_gcmParameters_content_waccm():
    # Create a waccmGCM instance
    waccm_gcm = waccmGCM(path=Path('/path/to/waccm.nc'), tstart=100 * u.day,
                         molecules=['O2', 'CO2'], aerosols=['Water', 'WaterIce'],
                         background='N2')
    mmw = 28.
    
    # Create a gcmParameters instance with waccmGCM
    gcm_params = gcmParameters(gcm=waccm_gcm,mean_molec_weight=mmw)
    
    # Assert that the content is not empty
    assert gcm_params.gcm.path == Path('/path/to/waccm.nc')

    assert gcm_params.to_psg()['ATMOSPHERE-WEIGHT'] == f'{mmw:.1f}'


def test_gcmParameters_from_dict_binary():
    # Create a dictionary representation of the gcmParameters instance with binaryGCM
    gcm_dict = {
        'star':None,
        'planet':None,
        'gcm':{
            'binary':{
                'data':'GCM data'
            },
            'mean_molec_weight': '28'
        }
    }
    
    
    # Create a gcmParameters instance from the dictionary
    gcm_params = gcmParameters.from_dict(gcm_dict)
    
    # Assert that the gcm attribute is an instance of binaryGCM
    assert isinstance(gcm_params.gcm, binaryGCM)
    assert gcm_params.content() == b'GCM data'

    assert gcm_params.to_psg()['ATMOSPHERE-WEIGHT'] == '28.0'


def test_gcmParameters_from_dict_waccm():
    # Create a dictionary representation of the gcmParameters instance with waccmGCM

    gcm_dict = {
        'star':None,
        'planet':None,
        'gcm':{
            'waccm': {
                'path': '/path/to/waccm.nc',
                'tstart': '100',
                'molecules': 'O2, CO2',
                'aerosols': 'Water, WaterIce',
                'background': 'N2'
            },
            'mean_molec_weight': '28'
        }
    }
    
    # Create a gcmParameters instance from the dictionary
    gcm_params = gcmParameters.from_dict(gcm_dict)
    
    # Assert that the gcm attribute is an instance of waccmGCM
    assert isinstance(gcm_params.gcm, waccmGCM)

    assert gcm_params.to_psg()['ATMOSPHERE-WEIGHT'] == '28.0'


def test_gcmParameters_from_dict_invalid():
    # Create an invalid dictionary representation without 'binary' or 'waccm' keys
    gcm_dict = {
        'star':None,
        'planet':None,
        'gcm':{
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