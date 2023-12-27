"""
Configuration for pytest.
"""
import pytest

from pypsg import settings

def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Add options to pytest.
    """
    parser.addoption('--external', action='store_true', help='use the external psg URL')

@pytest.fixture
def psg_url(request: pytest.FixtureRequest)->str:
    """
    Decide which psg URL to use.
    """
    external = request.config.getoption('--external')
    return settings.PSG_URL if external else settings.INTERNAL_PSG_URL