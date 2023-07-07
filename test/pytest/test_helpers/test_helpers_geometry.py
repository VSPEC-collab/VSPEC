

from astropy import units as u
import pytest
import numpy as np

from VSPEC import helpers


def test_get_angle_between():
    cases = [
        {'args': [0, 0, 0, 0]*u.deg, 'pred':0*u.deg},
        {'args': [0, 0, 90, 0]*u.deg, 'pred':90*u.deg},
        {'args': [0, 0, 40, 0]*u.deg, 'pred':40*u.deg},
        {'args': [0, 0, 0, 120]*u.deg, 'pred':120*u.deg},
        {'args': [0, 0, 0, -120]*u.deg, 'pred':120*u.deg},
        {'args': [90, 0, -90, 0]*u.deg, 'pred':180*u.deg},
    ]
    for case in cases:
        assert helpers.get_angle_between(
            *case['args']).to_value(u.deg) == pytest.approx(case['pred'].to_value(u.deg), abs=1)


def test_proj_ortho():
    lat0 = 30 * u.deg
    lon0 = 45 * u.deg
    lats = np.array([40, 50, 60]) * u.deg
    lons = np.array([60, 70, 80]) * u.deg

    x, y = helpers.proj_ortho(lat0, lon0, lats, lons)

    # Check the length of output arrays
    assert len(x) == len(y) == len(lats)

    # Check individual projected coordinates
    assert x[0] == pytest.approx(0.19825408, rel=0.01)
    assert x[1] == pytest.approx(0.27164737, rel=0.01)
    assert x[2] == pytest.approx(0.28682206, rel=0.01)

    assert y[0] == pytest.approx(0.18671294, rel=0.01)
    assert y[1] == pytest.approx(0.37213692, rel=0.01)
    assert y[2] == pytest.approx(0.54519419, rel=0.01)

    # Check for incorrect input type
    with pytest.raises(TypeError):
        helpers.proj_ortho(lat0.value, lon0.value, lats, lons)


def test_circle_intersection():
    cases = [
        {'args': [0, 0, 1], 'pred':1.0},
        {'args': [0, 0, 0.5], 'pred':1.0},
        {'args': [0, 0.5, 0.5], 'pred':1.0},
        {'args': [2, 0, 0.5], 'pred':0.0},
    ]
    for case in cases:
        calc = helpers.calc_circ_fraction_inside_unit_circle(*case['args'])
        pred = case['pred']
        assert calc == pytest.approx(pred, rel=1e-6)
    assert helpers.calc_circ_fraction_inside_unit_circle(1, 0, 1) < 0.5
    assert helpers.calc_circ_fraction_inside_unit_circle(0.6, 0, 0.5) > 0.5
