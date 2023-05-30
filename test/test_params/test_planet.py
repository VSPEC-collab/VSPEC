import pytest
import astropy.units as u
from VSPEC.params.planet import PlanetParameters, GravityParameters

@pytest.fixture
def planet_parameters():
    return PlanetParameters(
        name='Test Planet',
        radius=1.5 * u.R_earth,
        gravity=GravityParameters('g', 10*u.m/u.s**2),
        semimajor_axis=0.1 * u.AU,
        orbit_period=20 * u.day,
        rotation_period=10 * u.day,
        eccentricity=0.1,
        obliquity=30 * u.deg,
        obliquity_direction=45 * u.deg,
        init_phase=0 * u.deg,
        init_substellar_lon=180 * u.deg
    )

def test_planet_parameters_creation(planet_parameters:PlanetParameters):
    assert planet_parameters.name == 'Test Planet'
    assert planet_parameters.radius == 1.5 * u.R_earth
    assert isinstance(planet_parameters.gravity,GravityParameters)
    assert planet_parameters.semimajor_axis == 0.1 * u.AU
    assert planet_parameters.orbit_period == 20 * u.day
    assert planet_parameters.rotation_period == 10 * u.day
    assert planet_parameters.eccentricity == 0.1
    assert planet_parameters.obliquity == 30 * u.deg
    assert planet_parameters.obliquity_direction == 45 * u.deg
    assert planet_parameters.init_phase == 0 * u.deg
    assert planet_parameters.init_substellar_lon == 180 * u.deg

def test_planet_parameters_to_psg(planet_parameters:PlanetParameters):
    psg_dict = planet_parameters.to_psg()
    assert isinstance(psg_dict, dict)
    assert psg_dict['OBJECT'] == 'Exoplanet'
    assert psg_dict['OBJECT-NAME'] == 'Test Planet'
    assert psg_dict['OBJECT-DIAMETER'] == f'{2*(1.5*u.R_earth).to_value(u.km):.4f}'
    assert psg_dict['OBJECT-STAR-DISTANCE'] == '0.1000'
    assert psg_dict['OBJECT-PERIOD'] == '20.0000'
    assert psg_dict['OBJECT-ECCENTRICITY'] == '0.10000'
    assert psg_dict['OBJECT-GRAVITY'] == '1.0000e+01'
    assert psg_dict['OBJECT-GRAVITY-UNIT'] == 'g'

def test_planet_parameters_from_dict(planet_parameters:PlanetParameters):
    planet_dict = {
        'name': 'Test Planet',
        'radius': '1.5 earthRad',
        'gravity': {
            'mode': 'g',
            'value': '10 m s-2'
        },
        'semimajor_axis': '0.1 AU',
        'orbit_period': '20 day',
        'rotation_period': '10 day',
        'eccentricity': '0.1',
        'obliquity': '30 deg',
        'obliquity_direction': '45 deg',
        'init_phase': '0 deg',
        'init_substellar_lon': '180 deg'
    }
    new_parameters = PlanetParameters.from_dict(planet_dict)
    assert new_parameters.name == planet_parameters.name
    assert new_parameters.radius == planet_parameters.radius
    assert new_parameters.gravity.mode == planet_parameters.gravity.mode
    assert new_parameters.gravity.value == planet_parameters.gravity.value
    assert new_parameters.semimajor_axis == planet_parameters.semimajor_axis
    assert new_parameters.orbit_period == planet_parameters.orbit_period
    assert new_parameters.rotation_period == planet_parameters.rotation_period
    assert new_parameters.eccentricity == planet_parameters.eccentricity
    assert new_parameters.obliquity == planet_parameters.obliquity
    assert new_parameters.obliquity_direction == planet_parameters.obliquity_direction
    assert new_parameters.init_phase == planet_parameters.init_phase
    assert new_parameters.init_substellar_lon == planet_parameters.init_substellar_lon

def test_planet_parameters_proxcenb():
    init_phase = 90 * u.deg
    init_substellar_lon = 0 * u.deg
    planet = PlanetParameters.proxcenb(init_phase, init_substellar_lon)
    assert planet.name == 'Prox Cen b'
    assert planet.radius == 1.03 * u.M_earth
    assert planet.gravity.mode == 'kg'
    assert planet.gravity._value == 1.07 * u.M_earth
    assert planet.semimajor_axis == 0.04856 * u.AU
    assert planet.orbit_period == 11.1868 * u.day
    assert planet.rotation_period == 11.1868 * u.day
    assert planet.eccentricity == 0.0
    assert planet.obliquity == 0 * u.deg
    assert planet.obliquity_direction == 0 * u.deg
    assert planet.init_phase == init_phase
    assert planet.init_substellar_lon == init_substellar_lon

def test_planet_parameters_std():
    init_phase = 45 * u.deg
    init_substellar_lon = 90 * u.deg
    planet = PlanetParameters.std(init_phase, init_substellar_lon)
    assert planet.name == 'Exoplanet'
    assert planet.radius == 1.0 * u.M_earth
    assert planet.gravity.mode == 'kg'
    assert planet.gravity._value == 1.0 * u.M_earth
    assert planet.semimajor_axis == 0.05 * u.AU
    assert planet.orbit_period == 10 * u.day
    assert planet.rotation_period == 10 * u.day
    assert planet.eccentricity == 0.0
    assert planet.obliquity == 0 * u.deg
    assert planet.obliquity_direction == 0 * u.deg
    assert planet.init_phase == init_phase
    assert planet.init_substellar_lon == init_substellar_lon
