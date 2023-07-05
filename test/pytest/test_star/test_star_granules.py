"""
Test module for VSPEC Granulation model
"""
import numpy as np
from tinygp import GaussianProcess
from astropy import units as u
from VSPEC.variable_star_model import granules

def test_GranulationKernel():
    """
    Test for `granules.GranulationKernel`
    """
    scale=0.1
    period=3
    k = granules.GranulationKernel(scale,period)
    k.evaluate(0,1)

def test_build_gp():
    """
    Test for `granules.build_gp()`
    """
    params = dict(
            mean=0.5,
            scale=0.1,
            period=3
    )
    X = np.linspace(0,10,51)
    gp = granules.build_gp(params,X)
    assert isinstance(gp,GaussianProcess)

def test_safe_cast_coverage():
    """
    Test for `granules.safe_cast_coverage()`
    """
    a = np.linspace(0,1,5)
    assert np.all(a == granules.save_cast_coverage(a))
    a = np.zeros(5)+1.1
    assert np.all(granules.save_cast_coverage(a)==1)
    a = np.zeros(5)-0.1
    assert np.all(granules.save_cast_coverage(a)==0)

def test_granulation_init():
    """
    Test for `granules.Granulation.__init__()`
    """
    gran = granules.Granulation(0.2,0.01,3*u.day,200*u.K)
    assert hasattr(gran,'params')

def test_granulation_build_gp():
    gran = granules.Granulation(0.2,0.01,3*u.day,200*u.K)
    X = np.linspace(0,10,51)
    gp = gran._build_gp(X)
    assert isinstance(gp,GaussianProcess)

def test_granulation_get_coverage():
    t = np.linspace(0,10,51)*u.day
    gran = granules.Granulation(0.2,0.01,3*u.day,200*u.K)
    coverage = gran.get_coverage(t)
    assert not np.any(coverage > 1)
    assert not np.any(coverage < 0)
    
