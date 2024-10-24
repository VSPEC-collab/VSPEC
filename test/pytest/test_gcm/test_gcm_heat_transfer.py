"""

"""
import warnings
from pathlib import Path
from astropy import units as u
import numpy as np
import pytest
import matplotlib.pyplot as plt
from time import time

from libpypsg.globes import PyGCM
from libpypsg import PyConfig, APICall

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
    assert ht.get_psi(l) == 0
    l = 90*u.deg
    assert ht.get_psi(l) == np.pi/2
    l = 270*u.deg
    assert ht.get_psi(l) == -np.pi/2

def test_pcos():
    xs = [0,np.pi,0*u.deg,90*u.deg,180*u.deg]
    ys = [1,0,1,0,0]
    for x,y in zip(xs,ys):
        assert ht.pcos(x) == pytest.approx(y,abs=1e-6)

def test_colat():
    lats = [0,45,90,-45]*u.deg
    colats = [90,45,0,135]*u.deg
    for lat, colat in zip(lats,colats):
        assert ht.colat(lat) == colat
def test_equation_curve():
    modes = ['ivp_reflect','bvp','ivp_iterate','analytic']
    epsilons = [0.1,2,0.1,30]
    n_steps = 30
    for mode,eps in zip(modes,epsilons):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=ht.EquatorSolverWarning)
            lon,temp = ht.get_equator_curve(eps,n_steps,mode)
        assert lon[0] == pytest.approx(-np.pi,abs=1e-6)
        assert lon[-1] == pytest.approx(np.pi,abs=1e-6)
        assert len(lon) == len(temp), f'mismatched steps for mode {mode}'
        avg = np.mean(temp**4)**0.25
        assert avg == pytest.approx(0.75,rel=0.05)



@pytest.mark.plot()
@pytest.mark.slow()
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
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
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

@pytest.mark.plot()
def test_temp_map():
    eps = 6
    t0 = 300*u.K
    tmap = ht.TemperatureMap(eps,t0)
    lons = np.linspace(-np.pi,np.pi,30)
    lats = np.linspace(-np.pi/2,np.pi/2,20)
    0
    llons,llats = np.meshgrid(lons,lats)
    t = tmap.eval(llons,llats,0.01)
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
    t = tmap.eval(llons,llats,0.99)
    plt.pcolormesh(lons,lats,t.value)
    0

def test_write_cfg_params():
    gcm = ht.to_pygcm(
        (10,30,20),
        epsilon=1,
        star_teff=3300*u.K,
        r_star=0.15*u.R_sun,
        r_orbit=0.05*u.AU,
        lat_redistribution=1,
        p_surf=1*u.bar,
        p_stop=1e-5*u.bar,
        wind_u=0*u.m/u.s,
        wind_v=1*u.m/u.s,
        albedo=0.3*u.dimensionless_unscaled,
        emissivity=0.95*u.dimensionless_unscaled,
        gamma=1.2,
        molecules={'H2O':1e-3}
    )
    assert isinstance(gcm,PyGCM)
    atmosphere = gcm.update_params(None)
    assert atmosphere.description.value is not None
    assert atmosphere.molecules._ngas == 1
    assert atmosphere.molecules._value[0].name == 'H2O'
    cfg = atmosphere.content
    assert cfg != b''
    # assert b'<ATMOSPHERE-LAYERS>' + str(nlayers).encode('utf-8') in cfg
    assert b'<ATMOSPHERE-NAERO>' not in cfg
    assert b'<ATMOSPHERE-GAS>H2O' in cfg
    assert b'<ATMOSPHERE-NGAS>1' in cfg
    
    pycfg = PyConfig(gcm=gcm)
    content = pycfg.content
    assert b'<ATMOSPHERE-NAERO>' not in content
    assert b'<ATMOSPHERE-GAS>H2O' in content
    assert b'<ATMOSPHERE-NGAS>1' in content

def test_call_psg():
    gcm = ht.to_pygcm(
        (10,30,20),
        epsilon=1,
        star_teff=3300*u.K,
        r_star=0.15*u.R_sun,
        r_orbit=0.05*u.AU,
        lat_redistribution=0,
        p_surf=1*u.bar,
        p_stop=1e-5*u.bar,
        wind_u=0*u.m/u.s,
        wind_v=1*u.m/u.s,
        albedo=0.3*u.dimensionless_unscaled,
        emissivity=0.95*u.dimensionless_unscaled,
        gamma=1.2,
        molecules={'H2O':1e-3}
    )
    cfg = PyConfig(gcm=gcm)
    psg = APICall(cfg,'all','globes')
    response = psg()
    assert not np.any(np.isnan(response.lyr.prof['H2O']))


if __name__ in '__main__':
    pytest.main(args=[Path(__file__)])
