"""
Tests for main.py
"""

import pytest
from pathlib import Path
from os import chdir
from astropy import units as u
import numpy as np

from VSPEC.main import ObservationModel
from VSPEC.params.read import InternalParameters

cfg_path = Path(__file__).parent / 'test_params' / 'test.yaml'
chdir(Path(__file__).parent / 'data')


@pytest.fixture
def observation_model():
    # Create an ObservationModel instance with a test configuration file
    return ObservationModel.from_yaml(cfg_path)


def test_observation_model_initialization(observation_model:ObservationModel):
    # Verify that the ObservationModel instance is properly initialized
    assert observation_model.verbose == 1
    assert isinstance(observation_model.params, InternalParameters)
    assert observation_model.star is None




def test_get_observation_parameters(observation_model:ObservationModel):
    # Call the get_observation_parameters method
    system_geometry = observation_model.get_observation_parameters()

    # Verify that the returned SystemGeometry object has the correct attributes
    assert system_geometry.inclination == 90 * u.deg
    assert system_geometry.init_stellar_lon == 0 * u.deg
    assert system_geometry.init_planet_phase == 0 * u.deg
    assert system_geometry.stellar_period == 40 * u.day
    assert system_geometry.orbital_period == 10 * u.day
    assert system_geometry.semimajor_axis == 0.05 * u.AU
    assert system_geometry.planetary_rot_period == 10 * u.day
    assert system_geometry.planetary_init_substellar_lon == 0 * u.deg
    assert system_geometry.alpha == 0 * u.deg
    assert system_geometry.beta == 0 * u.deg
    assert system_geometry.eccentricity == 0.
    assert system_geometry.phase_of_periasteron == 0 * u.deg
    assert system_geometry.system_distance == 10.0 * u.pc
    assert system_geometry.obliquity == 0 * u.deg
    assert system_geometry.obliquity_direction == 0 * u.deg

def test_get_observation_plan(observation_model:ObservationModel):

    observation_parameters = observation_model.get_observation_parameters()

    # create observation plan for planet
    observation_plan = observation_model.get_observation_plan(observation_parameters,planet=True)
    expected_N_obs = int((observation_model.params.obs.observation_time / observation_model.params.obs.integration_time \
        / observation_model.params.psg.phase_binning)) + 1
    assert len(observation_plan['time']) == expected_N_obs

    # create observation plan for star
    observation_plan = observation_model.get_observation_plan(observation_parameters,planet=False)
    expected_N_obs = int((observation_model.params.obs.observation_time / observation_model.params.obs.integration_time))
    assert len(observation_plan['time']) == expected_N_obs





