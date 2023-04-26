"""
Tests for VSPEC.variable_star_model.faculae
"""

from astropy import units as u
import numpy as np
import pytest

from VSPEC.variable_star_model import Facula, FaculaCollection, FaculaGenerator
from VSPEC.helpers import MSH, CoordinateGrid, to_float

def init_facula(**kwargs):
    """
    Make a default Facula for testing.
    """
    return Facula(
        lat = kwargs.get('lat',0*u.deg),
        lon = kwargs.get('lon',0*u.deg),
        Rmax = kwargs.get('Rmax',300*u.km),
        R0 = kwargs.get('R0',100*u.km),
        Teff_floor = kwargs.get('Teff_floor',2900*u.K),
        Teff_wall = kwargs.get('Teff_wall',3500*u.K),
        lifetime = kwargs.get('lifetime',10*u.hr),
        growing = kwargs.get('growing',True),
        floor_threshold = kwargs.get('floor_threshold',20*u.km),
        Zw = kwargs.get('Zw',100*u.km),
        Nlat = kwargs.get('Nlat', 300),
        Nlon = kwargs.get('Nlon', 600),
        gridmaker = kwargs.get('gridmaker', None)
    )

def test_facula_init():
    """
    Test `Facula.__init__()`
    """
    fac = init_facula()
    assert np.all(fac.r > 0*u.deg)
    assert np.all(fac.r < 180*u.deg)
    assert np.max(to_float(fac.r, u.deg)) == pytest.approx(180, abs=1)

def test_facula_age():
    """
    Test `Facula.age()`
    """
    lifetime = 10*u.hr
    fac = init_facula(Rmax=100*u.km,R0=100/np.e*u.km,lifetime=lifetime,growing=True)
    fac.age(lifetime*0.5)
    assert fac.current_R == 100*u.km
    assert not fac.is_growing
    fac.age(lifetime*0.5)
    assert fac.current_R == 100/np.e*u.km

def test_facula_effective_area():
    """
    Test `Facula.effective_area()`
    """
    aunit = u.km**2
    tfloor = 2500*u.K
    twall = 3900*u.K
    # flat cold spot case
    radius = 100*u.km
    depth = 0*u.km
    fac = init_facula(R0=radius,Teff_floor=tfloor,Teff_wall=twall,Zw=depth)
    angle=0*u.deg
    d = fac.effective_area(angle,N=201)
    assert d[twall]==0*aunit
    assert d[tfloor].to_value(aunit)==pytest.approx((np.pi*(radius)**2).to_value(aunit),rel=1e-3)
    angle=5*u.deg
    d = fac.effective_area(angle,N=201)
    assert d[twall]==0*aunit
    assert d[tfloor].to_value(aunit)==pytest.approx((np.pi*(radius)**2*np.cos(angle)).to_value(aunit),rel=1e-3)
    # flat hot spot case (~inf depth)
    radius = 100*u.km
    depth = 1e9*u.km
    fac = init_facula(R0=radius,Teff_floor=tfloor,Teff_wall=twall,Zw=depth)
    angle=5*u.deg
    d = fac.effective_area(angle,N=201)
    assert d[tfloor]==0*aunit
    assert d[twall].to_value(aunit)==pytest.approx((np.pi*(radius)**2*np.cos(angle)).to_value(aunit),rel=1e-3)
    # normal case
    radius = 100*u.km
    depth = 100*u.km
    fac = init_facula(R0=radius,Teff_floor=tfloor,Teff_wall=twall,Zw=depth)
    angle=5*u.deg
    d1 = fac.effective_area(angle,N=201)
    angle=6*u.deg
    d2 = fac.effective_area(angle,N=201)
    assert d1[tfloor]/d1[twall] > d2[tfloor]/d2[twall]
    assert d1[tfloor] > d2[tfloor]
    assert d1[twall] < d2[twall]
    angle = np.arctan(2*radius/depth) # critical
    d = fac.effective_area(angle,N=201)
    assert d[tfloor].to_value(aunit) == pytest.approx(0,abs=1e-6)
    angle = np.arctan(2*radius/depth) - 1*u.deg # not quite critical
    d = fac.effective_area(angle,N=201)
    assert not d[tfloor].to_value(aunit) == pytest.approx(0,abs=1e-6)
    # threshold not reached case
    radius = 100*u.km
    depth = 100*u.km
    threshold = 200*u.km
    fac = init_facula(R0=radius,Teff_floor=tfloor,Teff_wall=twall,Zw=depth,floor_threshold=threshold)
    angle=5*u.deg
    d = fac.effective_area(angle,N=201)
    assert d[tfloor]==0*aunit
    assert d[twall].to_value(aunit)==pytest.approx((np.pi*(radius)**2*np.cos(angle)).to_value(aunit),rel=1e-3)







if __name__ in '__main__':
    test_facula_init()
    test_facula_age()
    test_facula_effective_area()