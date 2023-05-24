"""
Tests for parameters in stellar.py
"""

import pytest

from VSPEC.params.stellar import LimbDarkeningParameters

class TestLimbDarkeningParameters:
    @pytest.fixture
    def solar_params(self):
        return LimbDarkeningParameters.solar()

    @pytest.fixture
    def proxima_params(self):
        return LimbDarkeningParameters.proxima()

    @pytest.fixture
    def trappist_params(self):
        return LimbDarkeningParameters.trappist()

    @pytest.fixture
    def lambertian_params(self):
        return LimbDarkeningParameters.lambertian()

    def test_init(self):
        u1 = 0.3
        u2 = 0.1
        params = LimbDarkeningParameters(u1, u2)
        assert params.u1 == u1
        assert params.u2 == u2

    def test_solar_preset(self, solar_params):
        assert isinstance(solar_params, LimbDarkeningParameters)
        assert isinstance(solar_params.u1,float)
        assert isinstance(solar_params.u2,float)

    def test_proxima_preset(self, proxima_params):
        assert isinstance(proxima_params, LimbDarkeningParameters)
        assert isinstance(proxima_params.u1,float)
        assert isinstance(proxima_params.u2,float)

    def test_trappist_preset(self, trappist_params):
        assert isinstance(trappist_params, LimbDarkeningParameters)
        assert isinstance(trappist_params.u1,float)
        assert isinstance(trappist_params.u2,float)

    def test_lambertian_preset(self, lambertian_params):
        assert isinstance(lambertian_params, LimbDarkeningParameters)
        assert lambertian_params.u1 == 1.0
        assert lambertian_params.u2 == 0.0

    def test_from_dict_preset(self):
        params_dict = {'preset': 'solar'}
        params = LimbDarkeningParameters.from_dict(params_dict)
        assert isinstance(params, LimbDarkeningParameters)

    def test_from_dict_custom(self):
        u1,u2 = 0.9,0.1
        params_dict = {'u1': u1, 'u2': u2}
        params = LimbDarkeningParameters.from_dict(params_dict)
        assert isinstance(params, LimbDarkeningParameters)
        assert params.u1 == u1
        assert params.u2 == u2
