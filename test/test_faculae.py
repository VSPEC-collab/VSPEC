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
        r_max=kwargs.get('Rmax', 300*u.km),
        r_init=kwargs.get('R0', 100*u.km),
        teff_floor=kwargs.get('Teff_floor', 2900*u.K),
        teff_wall=kwargs.get('Teff_wall', 3500*u.K),
        lifetime=kwargs.get('lifetime', 10*u.hr),
        growing=kwargs.get('growing', True),
        floor_threshold=kwargs.get('floor_threshold', 20*u.km),
        depth=kwargs.get('Zw', 100*u.km),
        nlat=kwargs.get('Nlat', 300),
        nlon=kwargs.get('Nlon', 600),
        gridmaker=kwargs.get('gridmaker', None)
    )


def test_facula_init():
    """
    Test `Facula.__init__()`
    """
    fac = init_facula()
    assert np.all(fac.r > 0*u.deg)
    assert np.all(fac.r < 180*u.deg)
    assert np.max(fac.r.to_value(u.deg)) == pytest.approx(180, abs=1)


def test_facula_age():
    """
    Test `Facula.age()`
    """
    lifetime = 10*u.hr
    fac = init_facula(Rmax=100*u.km, R0=100/np.e*u.km,
                      lifetime=lifetime, growing=True)
    fac.age(lifetime*0.5)
    assert fac.radius == 100*u.km
    assert not fac.is_growing
    fac.age(lifetime*0.5)
    assert fac.radius == 100/np.e*u.km


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
    fac = init_facula(R0=radius, Teff_floor=tfloor, Teff_wall=twall, Zw=depth)
    angle = 0*u.deg
    d = fac.effective_area(angle, N=201)
    assert d[twall] == 0*aunit
    assert d[tfloor].to_value(aunit) == pytest.approx(
        (np.pi*(radius)**2).to_value(aunit), rel=1e-3)
    angle = 5*u.deg
    d = fac.effective_area(angle, N=201)
    assert d[twall] == 0*aunit
    assert d[tfloor].to_value(aunit) == pytest.approx(
        (np.pi*(radius)**2*np.cos(angle)).to_value(aunit), rel=1e-3)
    # flat hot spot case (~inf depth)
    radius = 100*u.km
    depth = 1e9*u.km
    fac = init_facula(R0=radius, Teff_floor=tfloor, Teff_wall=twall, Zw=depth)
    angle = 5*u.deg
    d = fac.effective_area(angle, N=201)
    assert d[tfloor] == 0*aunit
    assert d[twall].to_value(aunit) == pytest.approx(
        (np.pi*(radius)**2*np.cos(angle)).to_value(aunit), rel=1e-3)
    # normal case
    radius = 100*u.km
    depth = 100*u.km
    fac = init_facula(R0=radius, Teff_floor=tfloor, Teff_wall=twall, Zw=depth)
    angle = 5*u.deg
    d1 = fac.effective_area(angle, N=201)
    angle = 6*u.deg
    d2 = fac.effective_area(angle, N=201)
    assert d1[tfloor]/d1[twall] > d2[tfloor]/d2[twall]
    assert d1[tfloor] > d2[tfloor]
    assert d1[twall] < d2[twall]
    angle = np.arctan(2*radius/depth)  # critical
    d = fac.effective_area(angle, N=201)
    assert d[tfloor].to_value(aunit) == pytest.approx(0, abs=1e-6)
    angle = np.arctan(2*radius/depth) - 1*u.deg  # not quite critical
    d = fac.effective_area(angle, N=201)
    assert not d[tfloor].to_value(aunit) == pytest.approx(0, abs=1e-6)
    # threshold not reached case
    radius = 100*u.km
    depth = 100*u.km
    threshold = 200*u.km
    fac = init_facula(R0=radius, Teff_floor=tfloor,
                      Teff_wall=twall, Zw=depth, floor_threshold=threshold)
    angle = 5*u.deg
    d = fac.effective_area(angle, N=201)
    assert d[tfloor] == 0*aunit
    assert d[twall].to_value(aunit) == pytest.approx(
        (np.pi*(radius)**2*np.cos(angle)).to_value(aunit), rel=1e-3)


def test_facula_fractional_effective_area():
    """
    Test `Facula.fractional_effective_area()`
    """
    tfloor = 2500*u.K
    twall = 3900*u.K
    radius = 100*u.km
    depth = 100*u.km
    fac = init_facula(R0=radius, Teff_floor=tfloor, Teff_wall=twall, Zw=depth)
    angle = 0*u.deg
    d = fac.fractional_effective_area(angle, N=201)
    assert (d[tfloor]+d[twall]
            ).to_value(u.dimensionless_unscaled) == pytest.approx(1.0, rel=1e-6)
    angle = 5*u.deg
    d = fac.fractional_effective_area(angle, N=201)
    assert (d[tfloor]+d[twall]
            ).to_value(u.dimensionless_unscaled) == pytest.approx(1.0, rel=1e-6)
    angle = 90*u.deg
    d = fac.fractional_effective_area(angle, N=201)
    assert (d[tfloor]+d[twall]
            ).to_value(u.dimensionless_unscaled) == pytest.approx(1.0, rel=1e-6)


def test_facula_angular_radius():
    """
    Test `Facula.angular_radius()`
    """
    radius = 100*u.km
    r_star = 0.15*u.R_sun
    fac = init_facula(R0=radius)
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
    fac = init_facula(R0=radius, lat=lat, lon=lon)
    pmap = fac.map_pixels(r_star)
    assert np.all(pmap == 0)
    radius = np.pi*r_star
    fac = init_facula(R0=radius, lat=lat, lon=lon)
    pmap = fac.map_pixels(r_star)
    assert np.all(pmap == 1)
    radius = np.pi*r_star*0.9
    fac = init_facula(R0=radius, lat=lat, lon=lon)
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
        assert facula.r.shape == collec.gridmaker.zeros().shape
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
        Rmax=rmax, R0=0.8*rmax, growing=True))
    collec.clean_faclist()
    assert len(collec.faculae) == 1
    collec = FaculaCollection(init_facula(
        Rmax=rmax, R0=0.8*rmax, growing=False))
    collec.clean_faclist()
    assert len(collec.faculae) == 1
    collec = FaculaCollection(init_facula(
        Rmax=rmax, R0=0.01*rmax, growing=True))
    collec.clean_faclist()
    assert len(collec.faculae) == 1
    collec = FaculaCollection(init_facula(
        Rmax=rmax, R0=0.01*rmax, growing=False))
    collec.clean_faclist()
    assert len(collec.faculae) == 0


def test_fac_collection_age():
    """
    Test `FaculaCollection.age()`
    """
    lifetime = 10*u.hr
    N = 4
    collec = FaculaCollection(
        *[init_facula(Rmax=100*u.km, R0=100/np.e*u.km, lifetime=lifetime, growing=True) for i in range(N)]
    )
    collec.age(lifetime*0.5)
    for facula in collec.faculae:
        assert facula.current_R == 100*u.km
    collec.age(lifetime*0.5)
    for facula in collec.faculae:
        assert facula.current_R == 100*u.km/np.e
    collec.age(lifetime*0.5)
    assert len(collec.faculae) == 0


def test_fac_collection_map_pixels():
    """
    Test `FaculaCollection.map_pixels()`
    """
    lats = [-20, 0, 20]
    collec = FaculaCollection(
        *[init_facula(lat=lat*u.deg, R0=1000*u.km) for lat in lats],
        nlat=400, nlon=800
    )
    r_star = 0.15*u.R_sun
    teff = 3000*u.K
    pixmap = (collec.gridmaker.zeros()+1)*teff
    pmap, d = collec.map_pixels(pixmap, r_star, teff)
    n_faculae = len(collec.faculae)
    for i in range(n_faculae+1):
        assert np.any(pmap == i)
    for i in range(n_faculae):
        assert i in d.keys()


def test_fac_gen_init():
    """
    Test for `FaculaGenerator.__init__()`
    """
    gen = FaculaGenerator(nlat=300, nlon=600)
    assert isinstance(gen.dist_logr_peak, float)
    assert isinstance(gen.dist_logr_sigma, float)
    assert isinstance(gen.dist_loglife_peak, float)
    assert isinstance(gen.dist_loglife_sigma, float)
    assert gen.gridmaker == CoordinateGrid(300, 600)


def test_fac_gen_get_floor_teff():
    """
    Test for `FaculaGenerator.get_floor_teff()`
    """
    R = [100]*u.km
    teff = 3300*u.K
    gen = FaculaGenerator()
    teff_floor = gen.get_floor_teff(R, teff)
    assert np.all(teff_floor < teff)
    teff = 2400*u.K
    gen = FaculaGenerator()
    teff_floor = gen.get_floor_teff(R, teff)
    assert np.all(teff_floor == 2500*u.K)
    teff = 10000*u.K
    gen = FaculaGenerator()
    teff_floor = gen.get_floor_teff(R, teff)
    assert np.all(teff_floor == 3900*u.K)


def test_fac_gen_get_wall_teff():
    """
    Test for `FaculaGenerator.get_wall_teff()`
    """
    R = [100]*u.km
    teff_floor = 2500*u.K
    gen = FaculaGenerator()
    teff_wall = gen.get_wall_teff(R, teff_floor)
    assert np.all(teff_wall > teff_floor)
    teff_floor = 3900*u.K
    gen = FaculaGenerator()
    teff_wall = gen.get_wall_teff(R, teff_floor)
    assert np.all(teff_wall == 3900*u.K)


def test_fac_gen_get_n_expected():
    """
    Test for `FaculaGenerator.get_n_faculae_expected()`
    """
    R_peak = 1000*u.km
    T_peak = 10*u.hr
    coverage = 0.01
    r_star = 0.1*u.R_sun
    gen = FaculaGenerator(dist_r_peak=R_peak, dist_life_peak=T_peak, coverage=coverage)
    exp_area = 4*np.pi*r_star**2 * coverage
    exp_number = (exp_area / (np.pi*R_peak**2)
                  ).to_value(u.dimensionless_unscaled)
    assert gen.get_n_faculae_expected(
        2*T_peak, r_star) == pytest.approx(exp_number, rel=1e-6)
    assert gen.get_n_faculae_expected(
        T_peak, r_star) == pytest.approx(exp_number*0.5, rel=1e-6)


if __name__ in '__main__':
    test_facula_init()
    test_facula_age()
    test_facula_effective_area()
    test_facula_fractional_effective_area()
    test_facula_angular_radius()
    test_facula_map_pixels()
    test_fac_collection_init()
    test_fac_collection_add_facula()
    test_fac_collection_clean_faclist()
    test_fac_collection_age()
    test_fac_collection_map_pixels()
    test_fac_gen_init()
    test_fac_gen_get_floor_teff()
    test_fac_gen_get_wall_teff()
    test_fac_gen_get_n_expected()
