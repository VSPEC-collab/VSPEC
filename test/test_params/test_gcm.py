import pytest
from pathlib import Path
from astropy import units as u

from VSPEC.params.gcm import binaryGCM, waccmGCM

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

@pytest.mark.skip() # I don't have a NetCDF file that I can use for testing yet.
def test_waccmGCM_content():
    # Create a temporary netCDF file
    nc_path = Path("test.nc")

    # Define the parameters for the waccmGCM instance
    path = nc_path
    tstart = 1 * u.year
    molecules = ['O2', 'N2']
    aerosols = ['H2O']
    background = 'CH4'

    # Instantiate waccmGCM
    waccm = waccmGCM(path, tstart, molecules, aerosols, background)

    # Define the observation time
    obs_time = 10 * u.year

    # Get the content of the GCM for the specified observation time
    content = waccm.content(obs_time)

    # Clean up the temporary file
    nc_path.unlink()

def test_waccmGCM_from_dict():
    # Create a dictionary representation of waccmGCM
    waccm_dict = {
        'path': '/path/to/netcdf',
        'tstart': '1 day',
        'molecules': ['O2', 'CO2'],
        'aerosols': ['Water'],
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
