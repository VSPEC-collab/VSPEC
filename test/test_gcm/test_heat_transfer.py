"""

"""

from astropy import units as u, constants as c
import numpy as np
import pytest
import matplotlib.pyplot as plt

from VSPEC.gcm.heat_transfer import get_equillibrium_temp, get_flux
import VSPEC.gcm.heat_transfer as ht

def approx(val1,val2,rel):
    return np.abs(val1-val2)/val1 < rel

def test_get_flux():
    solar_constant = 1361*u.W/u.m**2
    teff_sun = 5800*u.K
    r_sun = 1*u.R_sun
    r_orbit = 1*u.AU
    assert approx(get_flux(teff_sun,r_sun,r_orbit),solar_constant,0.05)

def test_get_psi():
    l = 0*u.deg
    assert ht.get_psi(l) == l
    l = 90*u.deg
    assert ht.get_psi(l) == l
    l = 270*u.deg
    assert ht.get_psi(l) == -90*u.deg
    l = np.arange(-180,180,37)*u.deg
    assert np.all(ht.get_psi(l) == l)

def test_get_Teq():
    equillibrium_temp_earth = 255*u.K
    solar_constant = 1361*u.W/u.m**2
    
    teff_sun = 5800*u.K
    earth_albedo = 0.306
    r_sun = 1*u.R_sun
    r_orbit = 1*u.AU
    lon0 = 0*u.deg
    lat0 = 0*u.deg
    calc = ht.get_Teq(
        lon0,lat0,earth_albedo,teff_sun,r_sun,r_orbit
    )
    max_temp = ((solar_constant*(1-earth_albedo)/c.sigma_sb)**0.25).to(u.K)
    assert calc > equillibrium_temp_earth
    assert approx(calc,max_temp,0.01)

def test_get_equillibrium_temp():
    equillibrium_temp_earth = 255*u.K
    teff_sun = 5800*u.K
    earth_albedo = 0.306
    r_sun = 1*u.R_sun
    r_orbit = 1*u.AU
    assert approx(get_equillibrium_temp(teff_sun,earth_albedo,r_sun,r_orbit), equillibrium_temp_earth,0.05)

def test_get_equator_curve():
    plt.plot(*ht.get_equator_curve(0.0001,180))
    plt.plot(*ht.get_equator_curve(2*np.pi/10,180))
    plt.plot(*ht.get_equator_curve(2*np.pi,180))
    # plt.plot(*ht.get_equator_curve(10,lons))
    0

def test_get_equator_curve2():
    # plt.plot(*ht.get_equator_curve2(0.001,180))
    plt.plot(*ht.get_equator_curve2(2*np.pi/10,180))
    plt.plot(*ht.get_equator_curve2(2*np.pi,180))

def test_get_equator_curve3():
    plt.plot(*ht.get_equator_curve3(0.01,180))
    plt.plot(*ht.get_equator_curve3(2*np.pi/10,180))
    plt.plot(*ht.get_equator_curve3(2*np.pi,180))
    0

def test_get_equator_curve4():
    plt.plot(*ht.get_equator_curve4(0.01,180))
    # plt.plot(*ht.get_equator_curve4(2*np.pi/10,180))
    plt.plot(*ht.get_equator_curve4(2*np.pi,180))
    0
def test_equation_diagnotic():
    eps = np.logspace(-3,3,10)
    for i,func in enumerate([ht.get_equator_curve,ht.get_equator_curve2,ht.get_equator_curve3,ht.get_equator_curve4]):
        dat = []
        for e in eps:
            try:
                lon,temp = func(e,180)
                avg = np.mean(temp**4)**0.25
                dat.append(avg)
            except RuntimeError:
                dat.append(np.nan)
        plt.plot(eps,dat,label=f'Method {i+1}')
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('epsilon')
    plt.ylabel('Mean normalized temperature')
    plt.ylim(0.3,1.5)
    0


def compare_equation_curve():
    eps = 100
    for i,func in enumerate([ht.get_equator_curve,ht.get_equator_curve2,ht.get_equator_curve3]):
        try:
            lon,temp = func(eps,180)
            label = np.mean(temp**4)**0.25
            plt.plot(lon,temp,label=f'{label:.2f}',c=f'C{i}')
        except RuntimeError:
            pass
    plt.legend()
    # plt.plot(*ht.get_equator_curve(eps,180))
    # plt.plot(*ht.get_equator_curve2(eps,180))
    # plt.plot(*ht.get_equator_curve3(eps,180))
    0

def test_temp_map():
    eps = 6
    t0 = 300*u.K
    tmap = ht.TemperatureMap(eps,t0)
    lons = np.linspace(-np.pi,np.pi,30)
    lats = np.linspace(-np.pi/2,np.pi/2,20)
    0
    llons,llats = np.meshgrid(lons,lats)
    t = tmap.eval(llons,llats)
    plt.pcolormesh(lons,lats,t.value)
    0
    eps = 6
    star_teff=3300*u.K
    albedo=0.3
    r_star = 0.15*u.R_sun
    r_orbit = 0.05*u.AU
    tmap = ht.TemperatureMap.from_planet(
        eps,star_teff,albedo,r_star,r_orbit
    )
    t = tmap.eval(llons,llats)
    plt.pcolormesh(lons,lats,t.value)
    0
if __name__ in '__main__':
    test_equation_diagnotic()
    # test_get_equator_curve4()