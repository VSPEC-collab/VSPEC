"""
Configuration for pytest.
"""
import pytest
from pathlib import Path

from libpypsg import settings
import libpypsg

import VSPEC

def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Add options to pytest.
    """
    parser.addoption('--external', action='store_true', help='use the external psg URL')
    parser.addoption('--test1', action='store_true', help='rerun test 1')

@pytest.fixture
def psg_url(request: pytest.FixtureRequest)->str:
    """
    Decide which psg URL to use.
    """
    external = request.config.getoption('--external')
    if not external:
        if not libpypsg.docker.is_psg_installed():
            raise RuntimeError('PSG is not installed. Please use the pytest `--external` option.')
        else:
            libpypsg.docker.set_url_and_run()
            return settings.INTERNAL_PSG_URL
    else:
        return settings.PSG_URL
@pytest.fixture
def test1_data(request: pytest.FixtureRequest, psg_url: str, test1_model: VSPEC.ObservationModel) -> VSPEC.PhaseAnalyzer:
    """
    Run end-to-end test 1.
    """
    rerun = request.config.getoption('--test1')
    if rerun:
        with libpypsg.settings.temporary_settings(url=psg_url):
            test1_model.build_planet()
            test1_model.build_spectra()
    return VSPEC.PhaseAnalyzer(test1_model.directories['all_model'])

@pytest.fixture
def test1_model(request: pytest.FixtureRequest) -> VSPEC.ObservationModel:
    """
    Model for test 1.
    """
    path = Path(__file__).parent / 'end_to_end_tests' / 'test1' / 'test1.yaml'
    return VSPEC.ObservationModel.from_yaml(path)