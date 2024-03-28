"""
Test BaseParameters
"""
from VSPEC.params.base import BaseParameters


def test_base_parameters_from_dict():
    params_dict = {'param1': 1, 'param2': 2}
    params = BaseParameters.from_dict(params_dict)
    assert params.param1 == 1
    assert params.param2 == 2
