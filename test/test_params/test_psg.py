import pytest
from pathlib import Path

from VSPEC.config import PSG_EXT_URL

from VSPEC.params.gcm import APIkey, psgParameters

TEST_API_KEY = 'abcdefghijklmnopqrstuvwxyz'

def test_APIkey_value():
    # Create a temporary file containing the API key
    api_key_path = Path('api_key.txt')
    api_key_value = TEST_API_KEY
    api_key_path.write_text(api_key_value)

    # Create an APIkey instance with the temporary file path
    api_key = APIkey(path=api_key_path)

    # Get the API key value
    value = api_key.value

    # Perform assertions on the value
    assert value == api_key_value

    # Clean up the temporary file
    api_key_path.unlink()

def test_APIkey_value_no_path():
    # Create an APIkey instance without providing a path
    api_key = APIkey(value=TEST_API_KEY)

    # Get the API key value
    value = api_key.value

    # Perform assertions on the value
    assert value == TEST_API_KEY

def test_APIkey_value_invalid_path():
    # Create an APIkey instance with an invalid path
    api_key = APIkey(path=Path('invalid_path.txt'))

    with pytest.raises(FileNotFoundError):
        _ = api_key.value


def test_APIkey_value_empty_file():
    # Create a temporary empty file
    empty_file_path = Path('empty_file.txt')
    empty_file_path.touch()

    # Create an APIkey instance with the empty file path
    api_key = APIkey(path=empty_file_path)

    # Get the API key value
    value = api_key.value

    # Perform assertions on the value
    assert value == ''

    # Clean up the temporary file
    empty_file_path.unlink()

def test_APIkey_from_dict():
    # Create a dictionary with the API key data
    api_key_data = {
        'path': 'api_key.txt',
        'value': None
    }

    # Create an APIkey instance using the _from_dict class method
    api_key = APIkey.from_dict(api_key_data)

    # Perform assertions on the instance attributes
    assert api_key.path == Path('api_key.txt')
    assert api_key._value is None

    # Create a dictionary with different API key data
    api_key_data = {
        'path': None,
        'value': TEST_API_KEY
    }

    # Create another APIkey instance using the _from_dict class method
    api_key = APIkey.from_dict(api_key_data)

    # Perform assertions on the instance attributes
    assert api_key.path is None
    assert api_key._value == TEST_API_KEY



def test_psgParameters_from_dict():
    # Create a dictionary with the PSG parameters data
    psg_params_data = {
        'gcm_binning': '3',
        'phase_binning': '1',
        'use_molecular_signatures': 'True',
        'nmax':'0',
        'lmax':'0',
        'continuum':[],
        'url': PSG_EXT_URL,
        'api_key': {
            'path': 'api_key.txt',
        }
    }

    # Create a psgParameters instance using the _from_dict class method
    psg_params = psgParameters.from_dict(psg_params_data)

    # Perform assertions on the instance attributes
    assert psg_params.gcm_binning == 3
    assert psg_params.phase_binning == 1
    assert psg_params.use_molecular_signatures is True
    assert psg_params.nmax == 0
    assert psg_params.lmax == 0
    assert isinstance(psg_params.continuum,list)
    assert psg_params.url == PSG_EXT_URL
    assert isinstance(psg_params.api_key, APIkey)
    assert psg_params.api_key.path == Path('api_key.txt')
    assert psg_params.api_key._value is None

def test_psgParameters_to_psg():
    # Create a psgParameters instance
    psg_params = psgParameters(
        gcm_binning=3,
        phase_binning=1,
        use_molecular_signatures=True,
        nmax=0,
        lmax=0,
        continuum=['Rayleigh', 'Refraction'],
        url=PSG_EXT_URL,
        api_key=APIkey(path='api_key.txt')
    )

    # Convert the PSG parameters to the PSG input format
    psg_input = psg_params.to_psg()

    # Perform assertions on the PSG input dictionary
    assert psg_input['GENERATOR-GCM-BINNING'] == '3'
    assert psg_input['GENERATOR-GAS-MODEL'] == 'Y'
    assert psg_input['ATMOSPHERE-NMAX'] == '0'
    assert psg_input['ATMOSPHERE-LMAX'] == '0'
    assert psg_input['ATMOSPHERE-CONTINUUM'] == 'Rayleigh,Refraction'

