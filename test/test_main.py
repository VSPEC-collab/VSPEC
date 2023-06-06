"""
Tests for main.py
"""

import pytest
from pathlib import Path
import shutil
from os import chdir
from astropy import units as u
import numpy as np

from VSPEC.main import ObservationModel
from VSPEC.params.read import Parameters

cfg_path = Path(__file__).parent / 'test_params' / 'test.yaml'
chdir(Path(__file__).parent / 'data')


@pytest.fixture
def observation_model():
    # Create an ObservationModel instance with a test configuration file
    return ObservationModel.from_yaml(cfg_path)


def test_observation_model_initialization(observation_model:ObservationModel):
    # Verify that the ObservationModel instance is properly initialized
    assert observation_model.verbose == 1
    assert isinstance(observation_model.params, Parameters)
    assert observation_model.star is None


def test_build_directories(observation_model:ObservationModel):
    # Verify that the build_directories function creates the correct directory structure
    observation_model.build_directories()
    assert isinstance(observation_model.dirs, dict)
    assert 'parent' in observation_model.dirs
    assert 'data' in observation_model.dirs
    assert 'binned' in observation_model.dirs
    assert 'all_model' in observation_model.dirs
    assert 'psg_combined' in observation_model.dirs
    assert 'psg_thermal' in observation_model.dirs
    assert 'psg_noise' in observation_model.dirs
    assert 'psg_layers' in observation_model.dirs
    assert 'psg_configs' in observation_model.dirs
    for dir in observation_model.dirs.values():
        assert dir.exists()
    shutil.rmtree(observation_model.dirs['parent'])

def test_bin_spectra(observation_model:ObservationModel):
    observation_model.build_directories()
    observation_model.bin_spectra()
    assert any(observation_model.dirs['binned'].iterdir())
    shutil.rmtree(observation_model.dirs['parent'])

def test_read_binned_spectra(observation_model:ObservationModel):
    observation_model.build_directories()
    observation_model.bin_spectra()
    wl,flux = observation_model.read_spectrum(3000*u.K)
    # Check shape of arrays
    assert wl.shape == flux.shape
    # Check units
    assert wl.unit == observation_model.params.inst.bandpass.wavelength_unit
    assert flux.unit == observation_model.params.inst.bandpass.flux_unit
    shutil.rmtree(observation_model.dirs['parent'])

def test_get_model_spectrum(observation_model:ObservationModel):
    observation_model.build_directories()
    observation_model.bin_spectra()
    teff1 = 3000*u.K
    teff2 = 3100*u.K
    c1,c2 = 0.5,0.5
    teff3 = teff1*c1 + c2*teff2
    # check recall for uninterpolated spectrum
    wl1,flux1 = observation_model.read_spectrum(teff1)
    wl2,flux2 = observation_model.get_model_spectrum(teff1)
    assert np.all(wl1 == wl2)
    assert np.all(flux1*observation_model.params.flux_correction == flux2)
    # check that different teffs do not give the same results
    _,flux1 = observation_model.get_model_spectrum(teff1)
    _,flux2 = observation_model.get_model_spectrum(teff2)
    assert not np.all(flux1==flux2)
    # check that teff3 is an interpolation of teff1, teff2
    _,flux1 = observation_model.get_model_spectrum(teff1)
    _,flux2 = observation_model.get_model_spectrum(teff2)
    _,flux3 = observation_model.get_model_spectrum(teff3)
    pred = c1*flux1 + c2*flux2
    assert np.all(np.abs((flux3/pred).to_value(u.dimensionless_unscaled) - 1) < 1e-6)

    shutil.rmtree(observation_model.dirs['parent'])

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
        / observation_model.params.psg.phase_binning))
    assert len(observation_plan['time']) == expected_N_obs

    # create observation plan for star
    observation_plan = observation_model.get_observation_plan(observation_parameters,planet=False)
    expected_N_obs = int((observation_model.params.obs.observation_time / observation_model.params.obs.integration_time))
    assert len(observation_plan['time']) == expected_N_obs





