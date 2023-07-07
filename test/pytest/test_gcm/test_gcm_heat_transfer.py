"""

"""

from astropy import units as u, constants as c
import numpy as np
import pytest
import matplotlib.pyplot as plt
from time import time

from VSPEC.gcm.heat_transfer import get_flux
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


def test_equation_diagnostic():
    n_steps = 30
    eps = np.logspace(-4,3,n_steps)
    expected = 0.75
    for i, mode in enumerate(['ivp_reflect','bvp','analytic']):
        start_time = time()
        color=f'C{i}'
        dat = []
        for e in eps:
            try:
                lon,temp = ht.get_equator_curve(e,180,mode)
                avg = np.mean(temp**4)**0.25
                dat.append(avg)
            except RuntimeError:
                dat.append(np.nan)
        dtime = time() - start_time
        time_str = '10$^{%.1f}$' % np.log10(dtime/n_steps)
        plt.plot(eps,np.array(dat)-expected,label=f'{mode} -- {time_str} s / iter',c=color)
    plt.xscale('log')
    plt.yscale('symlog')
    plt.xlabel('epsilon')
    plt.ylabel('Mean normalized temperature')
    plt.ylim(-1,1.3)
    plt.legend()
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
    test_equation_diagnostic()
