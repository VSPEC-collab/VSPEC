"""
Tests for VSPEC.variable_star_model.faculae
"""

from astropy import units as u
import numpy as np
import pytest

from VSPEC.variable_star_model import Facula, FaculaCollection, FaculaGenerator
from VSPEC.helpers import CoordinateGrid


def init_facula(**kwargs):
    """
    Make a default Facula for testing.
    """
    return Facula(
        lat=kwargs.get('lat', 0*u.deg),
        lon=kwargs.get('lon', 0*u.deg),
        r_max=kwargs.get('r_max', 300*u.km),
        r_init=kwargs.get('r_init', 100*u.km),
        floor_teff_slope=kwargs.get('floor_teff_slope', 0*u.K/u.km),
        wall_teff_slope=kwargs.get('wall_teff_slope', 0*u.K/u.km),
        lifetime=kwargs.get('lifetime', 10*u.hr),
        growing=kwargs.get('growing', True),
        depth=kwargs.get('Zw', 100*u.km),
        nlat=kwargs.get('Nlat', 300),
        nlon=kwargs.get('Nlon', 600),
        gridmaker=kwargs.get('gridmaker', None),
        floor_teff_base_dteff=kwargs.get('floor_teff_base_dteff',100*u.K),
        wall_teff_intercept=kwargs.get('wall_teff_intercept',0*u.K),
        floor_teff_min_rad=kwargs.get('floor_teff_min_rad',10*u.km)
    )


def test_facula_init():
    """
    Test `Facula.__init__()`
    """
    fac = init_facula()
    assert np.all(fac._r > 0*u.deg)
    assert np.all(fac._r < 180*u.deg)
    assert np.max(fac._r.to_value(u.deg)) == pytest.approx(180, abs=1)


def test_facula_age():
    """
    Test `Facula.age()`
    """
    lifetime = 10*u.hr
    fac = init_facula(r_max=100*u.km, r_init=100/np.e*u.km,
                      lifetime=lifetime, growing=True)
    fac.age(lifetime*0.5)
    assert fac.radius == 100*u.km
    assert not fac.is_growing
    fac.age(lifetime*0.5)
    assert fac.radius == 100/np.e*u.km

@pytest.mark.filterwarnings('error')
def test_facula_effective_area():
    """
    Test `Facula.effective_area()`
    """
    aunit = u.km**2
    dtfloor = -200*u.K
    dtwall = 300*u.K
    # flat cold spot case
    radius = 100*u.km
    depth = 0*u.km
    fac = init_facula(r_init=radius, floor_teff_base_dteff=dtfloor, wall_teff_intercept=dtwall, Zw=depth)
    angle = 0*u.deg
    d = fac.effective_area(angle, N=201)
    assert d[dtwall] == 0*aunit
    assert d[dtfloor].to_value(aunit) == pytest.approx(
        (np.pi*(radius)**2).to_value(aunit), rel=1e-3)
    angle = 5*u.deg
    d = fac.effective_area(angle, N=201)
    assert d[dtwall] == 0*aunit
    assert d[dtfloor].to_value(aunit) == pytest.approx(
        (np.pi*(radius)**2*np.cos(angle)).to_value(aunit), rel=1e-3)
    # flat hot spot case (~inf depth)
    radius = 100*u.km
    depth = 1e9*u.km
    fac = init_facula(r_init=radius, floor_teff_base_dteff=dtfloor, wall_teff_intercept=dtwall, Zw=depth)
    angle = 5*u.deg
    d = fac.effective_area(angle, N=201)
    assert d[dtfloor] == 0*aunit
    assert d[dtwall].to_value(aunit) == pytest.approx(
        (np.pi*(radius)**2*np.cos(angle)).to_value(aunit), rel=1e-3)
    # normal case
    radius = 100*u.km
    depth = 100*u.km
    fac = init_facula(r_init=radius, floor_teff_base_dteff=dtfloor, wall_teff_intercept=dtwall, Zw=depth)
    angle = 5*u.deg
    d1 = fac.effective_area(angle, N=201)
    angle = 6*u.deg
    d2 = fac.effective_area(angle, N=201)
    assert d1[dtfloor]/d1[dtwall] > d2[dtfloor]/d2[dtwall]
    assert d1[dtfloor] > d2[dtfloor]
    assert d1[dtwall] < d2[dtwall]
    angle = np.arctan(2*radius/depth)  # critical
    d = fac.effective_area(angle, N=201)
    assert d[dtfloor].to_value(aunit) == pytest.approx(0, abs=1e-6)
    angle = np.arctan(2*radius/depth) - 1*u.deg  # not quite critical
    d = fac.effective_area(angle, N=201)
    assert not d[dtfloor].to_value(aunit) == pytest.approx(0, abs=1e-6)
    # threshold not reached case
    radius = 100*u.km
    depth = 100*u.km
    threshold = 200*u.km
    fac = init_facula(r_init=radius, floor_teff_base_dteff=dtfloor,
                      wall_teff_intercept=dtwall, Zw=depth, floor_teff_min_rad=threshold)
    angle = 5*u.deg
    d = fac.effective_area(angle, N=201)
    assert d[dtfloor] == 0*aunit
    assert d[dtwall].to_value(aunit) == pytest.approx(
        (np.pi*(radius)**2*np.cos(angle)).to_value(aunit), rel=1e-3)


def test_facula_fractional_effective_area():
    """
    Test `Facula.fractional_effective_area()`
    """
    dtfloor = 2500*u.K
    dtwall = 3900*u.K
    radius = 100*u.km
    depth = 100*u.km
    fac = init_facula(r_init=radius, floor_teff_base_dteff=dtfloor, wall_teff_intercept=dtwall, Zw=depth)
    angle = 0*u.deg
    d = fac.fractional_effective_area(angle, N=201)
    assert (d[dtfloor]+d[dtwall]
            ).to_value(u.dimensionless_unscaled) == pytest.approx(1.0, rel=1e-6)
    angle = 5*u.deg
    d = fac.fractional_effective_area(angle, N=201)
    assert (d[dtfloor]+d[dtwall]
            ).to_value(u.dimensionless_unscaled) == pytest.approx(1.0, rel=1e-6)
    angle = 90*u.deg
    d = fac.fractional_effective_area(angle, N=201)
    assert (d[dtfloor]+d[dtwall]
            ).to_value(u.dimensionless_unscaled) == pytest.approx(1.0, rel=1e-6)


def test_facula_angular_radius():
    """
    Test `Facula.angular_radius()`
    """
    radius = 100*u.km
    r_star = 0.15*u.R_sun
    fac = init_facula(r_init=radius)
    arc_length = (radius/r_star).to_value(u.dimensionless_unscaled)*u.rad
    assert fac.angular_radius(r_star).to_value(
        u.deg) == pytest.approx(arc_length.to_value(u.deg), rel=1e-6)


def test_facula_map_pixels():
    """
    Test `Facula.map_pixels()`
    """

    r_star = 0.15*u.R_sun
    lat = 0*u.deg
    lon = 0*u.deg
    radius = 0*u.km
    fac = init_facula(r_init=radius, lat=lat, lon=lon)
    pmap = fac.map_pixels(r_star)
    assert np.all(pmap == 0)
    radius = np.pi*r_star
    fac = init_facula(r_init=radius, lat=lat, lon=lon)
    pmap = fac.map_pixels(r_star)
    assert np.all(pmap == 1)
    radius = np.pi*r_star*0.9
    fac = init_facula(r_init=radius, lat=lat, lon=lon)
    pmap = fac.map_pixels(r_star)
    assert not np.all(pmap == 1)


def test_fac_collection_init():
    """
    Test `FaculaCollection.__init__()`
    """
    N = 4
    collec = FaculaCollection(*[init_facula(Nlat=400, Nlon=600)
                              for i in range(N)], nlat=300, nlon=600)
    expected_grid = CoordinateGrid(300, 600)
    for facula in collec.faculae:
        assert isinstance(facula, Facula)
    assert collec.gridmaker == expected_grid
    for facula in collec.faculae:
        assert facula.gridmaker == collec.gridmaker


def test_fac_collection_add_facula():
    """
    Test `FaculaCollection.add_facula()`
    """
    N = 4
    collec = FaculaCollection(*[init_facula(Nlat=400, Nlon=600)
                              for i in range(N)], nlat=300, nlon=600)
    assert len(collec.faculae) == N
    collec.add_faculae(init_facula(Nlat=400, Nlon=600))
    assert len(collec.faculae) == N+1
    for facula in collec.faculae:
        assert facula._r.shape == collec.gridmaker.zeros().shape
        assert facula.gridmaker == collec.gridmaker
    collec.add_faculae((init_facula(Nlat=400, Nlon=600),
                       init_facula(Nlat=400, Nlon=600)))
    assert len(collec.faculae) == N+1+2
    for facula in collec.faculae:
        assert facula.gridmaker == collec.gridmaker


def test_fac_collection_clean_faclist():
    """
    Test `FaculaCollection.clean_faclist()`
    """
    rmax = 100*u.km
    collec = FaculaCollection(init_facula(
        r_max=rmax, r_init=0.8*rmax, growing=True))
    collec.clean_faclist()
    assert len(collec.faculae) == 1
    collec = FaculaCollection(init_facula(
        r_max=rmax, r_init=0.8*rmax, growing=False))
    collec.clean_faclist()
    assert len(collec.faculae) == 1
    collec = FaculaCollection(init_facula(
        r_max=rmax, r_init=0.01*rmax, growing=True))
    collec.clean_faclist()
    assert len(collec.faculae) == 1
    collec = FaculaCollection(init_facula(
        r_max=rmax, r_init=0.01*rmax, growing=False))
    collec.clean_faclist()
    assert len(collec.faculae) == 0


def test_fac_collection_age():
    """
    Test `FaculaCollection.age()`
    """
    lifetime = 10*u.hr
    N = 4
    collec = FaculaCollection(
        *[init_facula(r_max=100*u.km, r_init=100/np.e*u.km, lifetime=lifetime, growing=True) for i in range(N)]
    )
    collec.age(lifetime*0.5)
    for facula in collec.faculae:
        assert facula.radius == 100*u.km
    collec.age(lifetime*0.5)
    for facula in collec.faculae:
        assert facula.radius == 100*u.km/np.e
    collec.age(lifetime*0.5)
    assert len(collec.faculae) == 0



def test_fac_gen_init():
    """
    Test for `FaculaGenerator.__init__()`
    """
    gen = FaculaGenerator(
        dist_r_peak=600*u.km,
        dist_life_logsigma=0.2,
        depth=100*u.km,
        dist_life_peak=6*u.hr,
        dist_r_logsigma=0.4,
        floor_teff_slope=0*u.K/u.km,
        floor_teff_min_rad=20*u.km,
        floor_teff_base_dteff=-100*u.K,
        wall_teff_slope=0*u.K/u.km,
        wall_teff_intercept=100*u.K,
        coverage=0.01,
        nlon=600,nlat=300
    )
    assert isinstance(gen.dist_r_peak, u.Quantity)
    assert isinstance(gen.dist_r_logsigma, float)
    assert isinstance(gen.dist_life_peak, u.Quantity)
    assert isinstance(gen.dist_life_logsigma, float)
    assert gen.gridmaker == CoordinateGrid(300, 600)

