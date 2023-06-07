#!/usr/bin/env python

"""
Tests for `VSPEC.helpers` module
"""
from os import system

from astropy import units as u
import numpy as np
import pytest
from time import sleep,time

from VSPEC import helpers
from VSPEC.geometry import SystemGeometry


def test_isclose():
    """
    Test `is_close()`

    Run tests for `VSPEC.helpers.isclose()`
    This function extends the `numpy.isclose()` function to
    support `astropy.units.Quantity` objects
    """
    param1 = np.linspace(0, 4, 5)*u.m
    param2 = np.linspace(0, 4, 5)*u.m
    tol = 0*u.m
    assert np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5)*u.m
    param2 = (np.linspace(0, 4, 5) + 1e-5)*u.m
    tol = 1e-4*u.m
    assert np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5)*u.m
    param2 = (np.linspace(0, 4, 5) + 1e-5)*u.m
    tol = 1e-6*u.m
    assert not np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5)*u.m
    param2 = (np.linspace(0, 4, 5) + 1e-5)*u.m
    tol = 1e-5*u.m  # boundary case
    assert np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5)*u.m
    param2 = (np.linspace(0, 400, 5))*u.cm
    tol = 1e-6*u.m
    assert np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5)*u.m
    param2 = (np.linspace(0, 4, 5))*u.cm
    tol = 1e-6*u.m
    assert not np.all(helpers.isclose(param1, param2, tol)), 'test failed'

    param1 = np.linspace(0, 4, 5)*u.m
    param2 = (np.linspace(0, 4, 5))*u.day
    tol = 1e-6*u.m
    try:
        np.all(helpers.isclose(param1, param2, tol))
        assert False, 'test failed'
    except u.UnitConversionError:
        pass

    param1 = np.linspace(0, 4, 5)*u.m
    param2 = (np.linspace(0, 4, 5))*u.m
    tol = 1e-6*u.s
    try:
        np.all(helpers.isclose(param1, param2, tol))
        assert False, 'test failed'
    except u.UnitConversionError:
        pass

    param1 = np.linspace(0, 4, 5)*u.m
    param2 = (np.linspace(0, 4, 5))*u.m
    tol = 1e-6*u.cm
    try:
        np.all(helpers.isclose(param1, param2, tol))
    except u.UnitConversionError:
        assert False, 'test failed'


def test_get_transit_radius():
    """
    Test `get_transit_radius()`

    Run tests for VSPEC.helpers.get_transit_radius()
    This function calculates the minimum radius from mid-transit
    (i.e. phase=180, i=90 deg) that a planet must be to have no
    overlap between the planetary disk and stellar disk.
    """
    # Use parameters for GJ 1214 b
    # Taken from NExSci Archive on
    # 2022-03-03 by Ted Johnson
    system_distance = 14.6427*u.pc
    stellar_radius = 0.215*u.R_sun
    semimajor_axis = 0.01490*u.AU
    planet_radius = 2.742*u.R_earth
    transit_duration = 0.8788*u.hr
    orbital_period = 1.58040433*u.day
    true_radius = np.pi*u.rad * \
        (transit_duration/orbital_period).to_value(u.dimensionless_unscaled)
    predicted_radius = helpers.get_transit_radius(system_distance,
                                                  stellar_radius,
                                                  semimajor_axis,
                                                  planet_radius)
    true_radius = true_radius.to_value(u.deg)
    predicted_radius = predicted_radius.to_value(u.deg)
    message = f'True: {true_radius:.2f}, Calculated: {predicted_radius:.2f}'
    assert predicted_radius == pytest.approx(
        true_radius, rel=0.1), 'test failed: ' + message

def port_is_running(port:int,timeout:float,target_state:bool):
    """
    It takes some time to know if a port has been switched off.
    We tell this function what we expect, and it listens for it.
    Say we turn PSG off, it asks "Is PSG running?" for 10 seconds
    until it hears back `False`. If it never does, it tells us that
    PSG is still running.

    Parameters
    ----------
    port : int
        The port to listen on.
    timeout : float
        The length of time to listen for in seconds.
    target_state : bool
        The expected state at the end of listening.
    
    Returns
    -------
    psg_state : bool
        The state of PSG. `True` if running on port `port`, else `False`.
    """
    timeout_time = time() + timeout
    while True:
        if target_state==True:
            if helpers.is_port_in_use(port):
                return True
            elif time() > timeout_time:
                return False
        else:
            if not helpers.is_port_in_use(port):
                return False
            elif time() > timeout_time:
                return True


def test_is_port_in_use():
    """
    Test `VSPEC.is_port_in_use`
    """
    default_psg_port = 3000
    timeout_duration = 10 # timeout after 10s
    previous_state = helpers.is_port_in_use(default_psg_port)
    system('docker stop psg')
    system('docker start psg')
    if not port_is_running(default_psg_port,timeout_duration,True):
        raise RuntimeError('Test failed -- timeout')
    system('docker stop psg')
    if port_is_running(default_psg_port,timeout_duration,False):
        raise RuntimeError('Test failed -- timeout')
    helpers.set_psg_state(previous_state)

def test_set_psg_state():
    """
    Test `VSPEC.helpers.set_psg_state`
    """
    psg_port = 3000
    timeout = 20
    previous_state = helpers.is_port_in_use(psg_port)
    helpers.set_psg_state(True)
    assert port_is_running(psg_port,timeout,True)
    helpers.set_psg_state(True)
    assert port_is_running(psg_port,timeout,True)
    helpers.set_psg_state(False)
    assert not port_is_running(psg_port,timeout,False)
    helpers.set_psg_state(False)
    assert not port_is_running(psg_port,timeout,False)
    helpers.set_psg_state(True)
    assert port_is_running(psg_port,timeout,True)
    helpers.set_psg_state(previous_state)

def test_arrange_teff():
    """
    Test `VSPEC.helpers.arrange_teff`
    """
    teff1 = 3010*u.K
    teff2 = 3090*u.K
    assert np.all(helpers.arrange_teff(teff1,teff2) == [3000,3100]*u.K)

    teff1 = 3000*u.K
    teff2 = 3100*u.K
    assert np.all(helpers.arrange_teff(teff1,teff2) == [3000,3100]*u.K)

    teff1 = 2750*u.K
    teff2 = 3300*u.K
    assert np.all(helpers.arrange_teff(teff1,teff2) == [27,28,29,30,31,32,33]*(100*u.K))

def test_get_surrounding_teffs():
    """
    Test `VSPEC.helpers.arrange_teff`
    """
    Teff = 3050*u.K
    low,high = helpers.get_surrounding_teffs(Teff)
    assert low == 3000*u.K
    assert high == 3100*u.K

    with pytest.raises(ValueError):
        Teff = 3000*u.K
        helpers.get_surrounding_teffs(Teff)

def test_plan_to_df():
    """
    Test `VSPEC.helpers.plan_to_df`
    """
    geo = SystemGeometry()
    plan = geo.get_observation_plan(0*u.deg,10*u.day,N_obs=10)
    df = helpers.plan_to_df(plan)
    for key in plan.keys():
        assert np.any(df.columns.str.contains(key))

def test_CoordinateGrid():
    """
    Test `VSPEC.helpers.CoordinateGrid`
    """
    helpers.CoordinateGrid()
    with pytest.raises(TypeError):
        helpers.CoordinateGrid(100,100.)
    with pytest.raises(TypeError):
        helpers.CoordinateGrid(100.,100)
    i,j = 100,50
    grid = helpers.CoordinateGrid(i,j)
    lat, lon = grid.oned()
    assert len(lat)==i
    assert len(lon)==j

    lats, lons = grid.grid()
    # switch if meshgrid index changes to `ij`
    assert lats.shape == (j,i)
    assert lons.shape == (j,i)

    zeros = grid.zeros(dtype='int')
    assert zeros.shape == (j,i)
    assert zeros.sum() == 0

    other = ''
    with pytest.raises(TypeError):
        grid == other
    
    other = helpers.CoordinateGrid(i+1,j)
    assert grid != other

    other = helpers.CoordinateGrid(i,j)
    assert grid == other

def test_round_teff():
    """
    Test `VSPEC.helpers.round_teff`
    """
    teff = 100.3*u.K
    assert helpers.round_teff(teff) == 100*u.K


def test_get_angle_between():
    cases = [
        {   'args':[0,0,0,0]*u.deg,'pred':0*u.deg   },
        {   'args':[0,0,90,0]*u.deg,'pred':90*u.deg   },
        {   'args':[0,0,40,0]*u.deg,'pred':40*u.deg   },
        {   'args':[0,0,0,120]*u.deg,'pred':120*u.deg   },
        {   'args':[0,0,0,-120]*u.deg,'pred':120*u.deg   },
        {   'args':[90,0,-90,0]*u.deg,'pred':180*u.deg   },
    ]
    for case in cases:
        assert helpers.get_angle_between(*case['args']).to_value(u.deg) == pytest.approx(case['pred'].to_value(u.deg),abs=1)


def test_circle_intersection():
    cases = [
        {   'args':[0,0,1], 'pred':1.0    },
        {   'args':[0,0,0.5], 'pred':1.0    },
        {   'args':[0,0.5,0.5], 'pred':1.0    },
        {   'args':[2,0,0.5], 'pred':0.0    },
    ]
    for case in cases:
        calc = helpers.calc_circ_fraction_inside_unit_circle(*case['args'])
        pred = case['pred']
        assert calc == pytest.approx(pred,rel=1e-6)
    assert helpers.calc_circ_fraction_inside_unit_circle(1,0,1) < 0.5
    assert helpers.calc_circ_fraction_inside_unit_circle(0.6,0,0.5) >0.5

def test_get_planet_indices():
    planet_times = np.array([0, 1, 2, 3, 4]) * u.day
    tindex = 2.5 * u.day

    N1, N2 = helpers.get_planet_indicies(planet_times, tindex)

    planet_times = np.array([0, 1, 2, 3, 4]) * u.day
    tindex = 2 * u.day
    N1, N2 = helpers.get_planet_indicies(planet_times, tindex)

    assert N1 == 2
    assert N2 == 2





if __name__ in '__main__':
    # test_to_float()
    # test_isclose()
    # test_get_transit_radius()
    # test_is_port_in_use()
    # test_set_psg_state()
    # test_arrange_teff()
    # test_plan_to_df()
    # test_CoordinateGrid()
    # test_circle_intersection()
    test_get_planet_indices()
