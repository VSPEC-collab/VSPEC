import pytest
from pathlib import Path
from astropy import units as u

from VSPEC.params.read import InternalParameters

test_file = Path(__file__).parent / 'test.yaml'


@pytest.fixture
def parameters():
    # Load the test.yaml file and create a InternalParameters instance
    return InternalParameters.from_yaml(test_file)


def test_star_parameters(parameters: InternalParameters):
    # Access and verify star parameters
    star_params = parameters.star
    assert star_params.psg_star_template == 'M'
    assert star_params.teff == 3300*u.K
    assert star_params.mass == 0.12*u.M_sun
    assert star_params.radius == 0.154*u.R_sun


def test_planet_parameters(parameters: InternalParameters):
    # Access and verify planet parameters
    planet_params = parameters.planet
    assert planet_params.name == 'Exoplanet'
    assert planet_params.radius == 1*u.R_earth
    assert planet_params.gravity.mode == 'kg'
    assert planet_params.gravity.value == 1.0*u.M_earth
