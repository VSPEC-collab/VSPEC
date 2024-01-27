"""

"""
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt

from VSPEC.gcm.structure import Variable
from VSPEC.gcm import structure as st

def test_variable():
    data = np.ones((2,2))*u.K
    var = Variable('name',u.K,data)
    assert np.all(var.flat == np.ones(4))

def test_variable_log():
    data = np.linspace(1,20,30)*u.bar
    var = Variable('logname',u.LogUnit(u.bar),data)
    pred = np.log10(data.to_value(u.bar))
    assert np.all(np.isclose(var.flat,pred,atol=1e-3))

def test_wind():
    vals = np.linspace(1,4,6)*u.cm / u.s
    wind = st.Wind('wind',vals)
    assert np.all(np.isclose(wind.flat,vals.to_value(u.m/u.s),atol=1e-3))

    val = 2*u.m/u.s
    s = (2,2)
    wind = st.Wind.contant('wind',val,s)
    assert np.all(np.isclose(wind.flat,np.ones(4)*2,atol=1e-3))

def test_pressure():
    prof = np.linspace(1,10,5)*u.bar
    shape = (6,7)
    press = st.Pressure.from_profile(prof,shape=shape)
    assert press.dat.shape == (5,6,7)
    assert np.all(press.dat[:,0,0] == prof)

    high = 1*u.bar
    low = 1e-5*u.bar
    shape = (6,2,2)
    expected = 10.**(-np.arange(0,6))*u.bar
    press = st.Pressure.from_limits(high,low,shape)
    assert press.dat.shape == shape
    assert np.all( np.abs(press.dat[:,0,0] - expected)/press.dat[:,0,0] < 1e-3)

def test_surface_pressure():
    val = 10*u.bar
    psurf = st.SurfacePressure.constant(val,(30,20))
    assert np.all(psurf.flat == 1.)

def test_surface_temperature():
    shape = (300,200)
    eps = 1
    star_teff=5800*u.K
    albedo=0.3
    r_star = 1*u.R_sun
    r_orbit = 1*u.AU
    tsurf = st.SurfaceTemperature.from_map(
        shape=shape,
        epsilon=eps,
        star_teff=star_teff,
        albedo = albedo,
        r_star=r_star,
        r_orbit=r_orbit,
        lat_redistribution=0.9
    )
    assert tsurf.dat.shape==shape
    plt.imshow(tsurf.dat.value)
    plt.colorbar(label='K')
    0

def test_temperature():
    shape = (300,200)
    eps = 1
    star_teff=5800*u.K
    albedo=0.3
    r_star = 1*u.R_sun
    r_orbit = 1*u.AU
    tsurf = st.SurfaceTemperature.from_map(
        shape=shape,
        epsilon=eps,
        star_teff=star_teff,
        albedo = albedo,
        r_star=r_star,
        r_orbit=r_orbit,
        lat_redistribution=0.5
    )
    pressure = st.Pressure.from_limits(1*u.bar,1e-5*u.bar,shape=(50,300,200))
    gamma = 1.4
    temp = st.Temperature.from_adiabat(gamma,tsurf,pressure)
    pprof = pressure.dat[:,150,100]
    tprof = temp.dat[:,150,100]
    plt.plot(tprof,pprof)
    plt.yscale('log')
    plt.ylim(*np.flip(plt.ylim()))
    0

def test_molecule():
    val = 1e-4*u.mol/u.mol
    molec = st.Molecule.constant('H2O',val,(20,40,30))
    assert np.all(molec.flat == -4)

def test_aersol():
    val = 1e-4*u.kg/u.kg
    aero = st.Aerosol.constant('Water',val,(20,40,30))
    assert np.all(aero.flat == -4)

    pressure = st.Pressure.from_limits(1*u.bar,1e-5*u.bar,shape=(50,300,200))
    aero = st.Aerosol.boyant_exp('Water',1e-2*u.kg/u.kg,0.1*u.bar,pressure)
    pprof = pressure.dat[:,150,100]
    aprof = aero.dat[:,150,100]
    plt.plot(aprof,pprof)
    plt.yscale('log')
    plt.ylim(*np.flip(plt.ylim()))
    0

def test_aerosol_size():
    val = 1e-3*u.um
    aero = st.AerosolSize.constant('Water_size',val,(20,40,30))
    assert np.all(aero.flat == -9)

def test_albedo():
    val = 0.3*u.dimensionless_unscaled
    albedo = st.Albedo.constant(val,(40,30))
    assert np.all(albedo.flat == 0.3)
def test_emissivity():
    val = 1.0*u.dimensionless_unscaled
    albedo = st.Emissivity.constant(val,(40,30))
    assert np.all(albedo.flat == 1.0)
    


if __name__ in '__main__':
    test_aersol()