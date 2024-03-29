import pytest
import astropy.units as u
from VSPEC.params.planet import GravityParameters


def test_gravity_parameters():
    # Test initialization
    gp = GravityParameters('g', 9.8 * u.m/u.s**2)
    assert gp.mode == 'g'
    assert gp.value == 9.8 * u.m/u.s**2

    # Test value property
    assert gp.value == 9.8 * u.m/u.s**2

    # Test from_dict classmethod
    params_dict = {
        'mode': 'rho',
        'value': 1.2 * u.g/u.cm**3
    }
    gp2 = GravityParameters.from_dict(params_dict)
    assert gp2.mode == 'rho'
    assert gp2.value == 1.2 * u.g/u.cm**3

    # Test from_dict classmethod with mass
    params_dict = {
        'mode': 'kg',
        'value': 1*u.M_earth
    }
    gp3 = GravityParameters.from_dict(params_dict)
    assert gp3.mode == 'kg'
    assert gp3.value.to_value(u.kg) == pytest.approx(
        (1*u.M_earth).to_value(u.kg), rel=1e-6)
