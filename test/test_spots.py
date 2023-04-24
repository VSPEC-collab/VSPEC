"""
Tests for VSPEC.variable_star_model.spots
"""
from astropy import units as u
import numpy as np
import pytest
import matplotlib.pyplot as plt

from VSPEC.variable_star_model import StarSpot, SpotCollection, SpotGenerator
from VSPEC.helpers import MSH, CoordinateGrid, to_float


def init_test_spot(**kwargs):
    """
    Initialize a default StarSpot for testing
    """
    lat = kwargs.get('lat', 0*u.deg)
    lon = kwargs.get('lon', 0*u.deg)
    Amax = kwargs.get('Amax', 500*MSH)
    A0 = kwargs.get('A0', 400*MSH)
    Teff_umbra = kwargs.get('Teff_umbra', 2700*u.K)
    Teff_penumbra = kwargs.get('Teff_penumbra', 2500*u.K)
    r_A = kwargs.get('r_A', 5.)
    growing = kwargs.get('growing', True)
    growth_rate = kwargs.get('growth_rate', 0/u.day)
    decay_rate = kwargs.get('decay_rate', 0*MSH/u.day)
    Nlat = kwargs.get('Nlat', 500)
    Nlon = kwargs.get('Nlon', 1000)
    gridmaker = kwargs.get('gridmaker', None)

    spot = StarSpot(
        lat=lat,
        lon=lon,
        Amax=Amax,
        A0=A0,
        Teff_umbra=Teff_umbra,
        Teff_penumbra=Teff_penumbra,
        r_A=r_A,
        growing=growing,
        growth_rate=growth_rate,
        decay_rate=decay_rate,
        Nlat=Nlat,
        Nlon=Nlon,
        gridmaker=gridmaker
    )
    return spot


def test_spot_init():
    """
    Test __init__() for starspots
    """
    spot = init_test_spot()
    assert np.all(spot.r > 0*u.deg)
    assert np.all(spot.r < 180*u.deg)
    assert np.max(to_float(spot.r, u.deg)) == pytest.approx(180, abs=1)

    other = init_test_spot(
        gridmaker=CoordinateGrid(500+1, 1000),
        Nlat=None,
        Nlon=None
    )
    assert spot.gridmaker != other.gridmaker


def test_spot_str():
    """
    Make sure the __str__ method works.
    """
    spot = init_test_spot()
    s = str(spot)
    assert isinstance(s, str)


def test_spot_radius():
    """
    Test the spot radius
    """
    current_area = 400*MSH
    spot = init_test_spot(A0=current_area)
    # A = pi r**2
    # r = sqrt(A/pi)
    pred = np.sqrt(current_area/np.pi)
    obs = spot.radius()
    assert to_float(pred, u.km) == pytest.approx(to_float(obs, u.km), rel=1e-6)


def test_spot_angular_radius():
    stellar_rad = 1000*u.km
    star_surface_area = 4*np.pi*stellar_rad**2
    spot = init_test_spot(A0=star_surface_area)

    pred = 180*u.deg
    obs = spot.angular_radius(stellar_rad)
    assert to_float(obs, u.deg) == pytest.approx(
        to_float(pred, u.deg), abs=0.01)

    spot = init_test_spot(A0=0.5*star_surface_area)
    pred = 90*u.deg
    obs = spot.angular_radius(stellar_rad)
    assert to_float(obs, u.deg) == pytest.approx(
        to_float(pred, u.deg), abs=0.01)


def test_spot_map_pixels():
    """
    Test StarSpot.map_pixels
    """
    stellar_rad = 1000*u.km
    star_surface_area = 4*np.pi*stellar_rad**2
    spot = init_test_spot(A0=star_surface_area, r_A=1)  # no penumbra
    pixmaps = spot.map_pixels(stellar_rad)
    umbra = pixmaps[spot.Teff_umbra]
    penumbra = pixmaps[spot.Teff_penumbra]
    assert np.all(umbra)
    assert not np.any(penumbra & ~umbra)

    spot = init_test_spot(A0=star_surface_area, r_A=np.inf)  # no umbra
    pixmaps = spot.map_pixels(stellar_rad)
    umbra = pixmaps[spot.Teff_umbra]
    penumbra = pixmaps[spot.Teff_penumbra]
    assert np.all(penumbra)
    assert not np.any(umbra)

    spot = init_test_spot(A0=star_surface_area, r_A=4, lat=0*u.deg)
    # half and half --> This is a sphere, so our 2D approximation goes out the window
    # for a large radius spot.
    pixmaps = spot.map_pixels(stellar_rad)
    umbra = pixmaps[spot.Teff_umbra]
    penumbra = pixmaps[spot.Teff_penumbra]
    lats, lons = spot.gridmaker.grid()
    sin_theta = np.cos(lats)
    assert (umbra*sin_theta.value).sum() == pytest.approx(((penumbra &
                                                            ~umbra)*sin_theta.value).sum(), rel=0.01)

    spot = init_test_spot(A0=1e-3*star_surface_area,
                          r_A=2, Nlon=2000, Nlat=4000)
    # half and half --> we try to get close to 2D without the discrete grid messing things up
    pixmaps = spot.map_pixels(stellar_rad)
    umbra = pixmaps[spot.Teff_umbra]
    penumbra = pixmaps[spot.Teff_penumbra]
    lats, lons = spot.gridmaker.grid()
    sin_theta = np.cos(lats)
    assert (umbra*sin_theta.value).sum() == pytest.approx(
        ((penumbra & ~umbra)*sin_theta.value).sum(), rel=0.05)


def test_spot_surface_fraction():
    """
    Test StarSpot.sufrace_fraction
    """
    stellar_rad = 1000*u.km
    star_surface_area = 4*np.pi*stellar_rad**2
    spot = init_test_spot(A0=0.01*star_surface_area)
    sub_obs = {'lat': 0*u.deg, 'lon': 0*u.deg}
    assert spot.surface_fraction(
        sub_obs, stellar_rad) == pytest.approx(0.02, rel=0.01)
    sub_obs = {'lat': 0*u.deg, 'lon': 180*u.deg}
    assert spot.surface_fraction(
        sub_obs, stellar_rad) == pytest.approx(0.0, rel=0.01)


def test_spot_age():
    """
    Test StarSpot.age
    """
    spot = init_test_spot(A0=10*MSH, Amax=100*MSH, growth_rate=1 /
                          u.day, decay_rate=10*MSH/u.day, growing=True)
    step = 1*u.day
    assert to_float(spot.area_current, MSH) == pytest.approx(10, rel=0.01)
    spot.age(step)
    assert to_float(spot.area_current, MSH) == pytest.approx(20, rel=0.01)
    spot.age(step)
    assert to_float(spot.area_current, MSH) == pytest.approx(40, rel=0.01)
    spot.age(step)
    assert to_float(spot.area_current, MSH) == pytest.approx(80, rel=0.01)
    spot.age(step)
    assert not spot.is_growing

    spot = init_test_spot(A0=100*MSH, Amax=100*MSH, growth_rate=1 /
                          u.day, decay_rate=10*MSH/u.day, growing=False)
    assert to_float(spot.area_current, MSH) == pytest.approx(100, rel=0.01)
    spot.age(step)
    assert to_float(spot.area_current, MSH) == pytest.approx(90, rel=0.01)
    spot.age(9*step)
    assert to_float(spot.area_current, MSH) == pytest.approx(0, rel=0.01)

    spot = init_test_spot(A0=10*MSH, Amax=100*MSH, growth_rate=1 /
                          u.day, decay_rate=10*MSH/u.day, growing=True)
    step = 1*u.day
    assert to_float(spot.area_current, MSH) == pytest.approx(10, rel=0.01)
    spot.age(20*step)
    assert to_float(spot.area_current, MSH) == pytest.approx(0, rel=0.01)


def test_init_spot_collection():
    """
    Test spot initialization
    """
    N = 4
    collec = SpotCollection(*[init_test_spot()
                            for i in range(N)], Nlat=300, Nlon=600)
    expected_grid = CoordinateGrid(300, 600)
    for spot in collec.spots:
        assert isinstance(spot, StarSpot)
    assert collec.gridmaker == expected_grid
    for spot in collec.spots:
        assert spot.gridmaker == collec.gridmaker


def test_spot_collection_add_spot():
    """
    Test `SpotCollection.add_spot()`
    """
    N = 4
    collec = SpotCollection(*[init_test_spot()
                            for i in range(N)], Nlat=300, Nlon=600)
    assert len(collec.spots) == N
    new_spot = init_test_spot()
    collec.add_spot(new_spot)
    assert len(collec.spots) == N+1
    new_spots = [init_test_spot() for i in range(N)]
    collec.add_spot(new_spots)
    assert len(collec.spots) == 2*N+1
    for spot in collec.spots:
        assert spot.gridmaker == collec.gridmaker


def test_spot_collection_clean_spotlist():
    """
    Test `SpotCollection.clean_spotlist()`
    """
    N = 4
    spots = [init_test_spot() for i in range(N)] + \
        [init_test_spot(A0=0*MSH, growing=True)]
    collec = SpotCollection(*spots, Nlat=300, Nlon=600)
    collec.clean_spotlist()
    assert len(collec.spots) == N+1
    spots = [init_test_spot() for i in range(N)] + \
        [init_test_spot(A0=0*MSH, growing=False)]
    collec = SpotCollection(*spots, Nlat=300, Nlon=600)
    collec.clean_spotlist()
    assert len(collec.spots) == N


def test_spot_collection_map_pixels():
    """
    Test `SpotCollection.map_pixels()`
    """
    R_star = 0.15*u.R_sun
    Teff = 3300*u.K
    spots = [
        init_test_spot(Teff_umbra=2500*u.K, Teff_penumbra=2500*u.K),
        init_test_spot(Teff_umbra=2700*u.K, Teff_penumbra=2700*u.K)
    ]
    collec = SpotCollection(*spots, Nlat=300, Nlon=600)
    pmap = collec.map_pixels(R_star, Teff)
    assert not np.any(pmap == 2700*u.K)

    spots = [
        init_test_spot(Teff_umbra=3500*u.K, Teff_penumbra=3500*u.K),
        init_test_spot(Teff_umbra=3700*u.K, Teff_penumbra=3700*u.K)
    ]
    collec = SpotCollection(*spots, Nlat=300, Nlon=600)
    pmap = collec.map_pixels(R_star, Teff)
    assert np.all(pmap == Teff)


def test_spot_collection_age():
    """
    Test `SpotCollection.age()`
    """
    spot = init_test_spot(A0=10*MSH, Amax=100*MSH, growth_rate=1 /
                          u.day, decay_rate=10*MSH/u.day, growing=True)
    time = 1*u.day
    collec = SpotCollection(spot, Nlat=300, Nlon=600)
    assert collec.spots[0].area_current == 10*MSH
    collec.age(time)
    assert collec.spots[0].area_current == 20*MSH
    collec.age(time*20)
    assert len(collec.spots) == 0


if __name__ in '__main__':
    test_spot_init()
    test_spot_str()
    test_spot_radius()
    test_spot_angular_radius()
    test_spot_map_pixels()
    test_spot_surface_fraction()
    test_spot_age()
    test_init_spot_collection()
    test_spot_collection_add_spot()
    test_spot_collection_clean_spotlist()
    test_spot_collection_map_pixels()
    test_spot_collection_age()
