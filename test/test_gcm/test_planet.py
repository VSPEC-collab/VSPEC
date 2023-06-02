"""
Tests for gcm.planet

"""
from astropy import units as u
import pytest
from pathlib import Path

from VSPEC.gcm import planet as pl
from VSPEC.gcm import structure as st
from VSPEC.psg_api import call_api

OUTFILE = Path(__file__).parent / 'config.txt'


def test_winds():
    shape=(10,30,20)
    U = st.Wind.contant('U',1*u.m/u.s,shape=shape)
    V = st.Wind.contant('V',1*u.m/u.s,shape=shape)
    winds = pl.Winds(U,V)
    assert winds.flat.ndim == 1

def test_molecules():
    shape=(10,30,20)
    h2o = st.Molecule.constant('H2O',1e-4*u.mol/u.mol,shape=shape)
    co2 = st.Molecule.constant('CO2',1e-5*u.mol/u.mol,shape=shape)
    molecs = pl.Molecules((h2o,co2))
    assert molecs.flat.ndim == 1

def test_aerosols():
    shape=(10,30,20)
    water = st.Aerosol.constant('Water',0.1*u.kg/u.kg,shape=shape)
    water_size = st.AerosolSize.constant('Water_size',1*u.um,shape=shape)
    ice = st.Aerosol.constant('WaterIce',0.05*u.kg/u.kg,shape=shape)
    ice_size = st.AerosolSize.constant('WaterIce_size',1*u.um,shape=shape)
    aeros = pl.Aerosols((water,ice),(water_size,ice_size))
    assert aeros.flat.ndim == 1

@pytest.fixture
def default_planet():
    shape3d = (10,30,20)
    shape2d = (30,20)
    tsurf = st.SurfaceTemperature.from_map(shape2d,0.001,5800*u.K,0.3,1*u.R_sun,1*u.AU)
    press = st.Pressure.from_limits(1*u.bar,1e-5*u.bar,shape3d)
    planet = pl.Planet(
        wind = pl.Winds(
            st.Wind.contant('U',0*u.m/u.s,shape=shape3d),
            st.Wind.contant('V',0*u.m/u.s,shape=shape3d)
        ),
        tsurf=tsurf,
        psurf=st.SurfacePressure.from_pressure(press),
        albedo=st.Albedo.constant(0.3*u.dimensionless_unscaled,shape2d),
        emissivity=st.Emissivity.constant(1*u.dimensionless_unscaled,shape2d),
        temperature=st.Temperature.from_adiabat(1.4,tsurf,press),
        pressure=press,
        molecules=pl.Molecules(
            (st.Molecule.constant('CO2',5e-4*u.mol/u.mol,shape3d),)
        ),
        aerosols=None
        # aerosols=pl.Aerosols(
        #     aerosols=(st.Aerosol.constant('Water',0.2*u.kg/u.kg,shape3d),),
        #     sizes=(st.AerosolSize.constant('Water_size',1*u.um,shape3d),)
        # )
    )
    return planet

def test_planet():
    shape3d = (10,30,20)
    shape2d = (30,20)
    tsurf = st.SurfaceTemperature.from_map(shape2d,6,5800*u.K,0.3,1*u.R_sun,1*u.AU)
    press = st.Pressure.from_limits(1*u.bar,1e-5*u.bar,shape3d)
    planet = pl.Planet(
        wind = pl.Winds(
            st.Wind.contant('U',0*u.m/u.s,shape=shape3d),
            st.Wind.contant('V',0*u.m/u.s,shape=shape3d)
        ),
        tsurf=tsurf,
        psurf=st.SurfacePressure.constant(1*u.bar,shape2d),
        albedo=st.Albedo.constant(0.3*u.dimensionless_unscaled,shape2d),
        emissivity=st.Emissivity.constant(1*u.dimensionless_unscaled,shape2d),
        temperature=st.Temperature.from_adiabat(1.4,tsurf,press),
        pressure=press,
        molecules=pl.Molecules(
            (st.Molecule.constant('CO2',5e-4*u.mol/u.mol,shape3d),)
        ),
        aerosols=None
        # aerosols=pl.Aerosols(
        #     aerosols=(st.Aerosol.constant('Water',1e-20*u.kg/u.kg,shape3d),),
        #     sizes=(st.AerosolSize.constant('Water_size',1*u.um,shape3d),)
        # )
    )

def test_planet_from_dict():
    d = {
        'shape':{
            'nlayer':'10',
            'nlon':'30',
            'nlat':'20'
        },
        'planet':{
            'epsilon':'5.0',
            'teff_star': '5800 K',
            'albedo': '0.3',
            'emissivity':'1.0',
            'r_star': '1 R_sun',
            'r_orbit': '1 AU',
            'gamma': '1.4',
            'pressure':{
                'psurf': '1 bar',
                'ptop': '1e-5 bar'
            },
            'wind':{
                'U': '1 m/s',
                'V': '1 m/s'
            },
        },
        'molecules':{
            'H2O': '1e-3',
            'CO2': '1e-4'
        },
        'aerosols':{
            'Water':{
                'abn':'1e-2',
                'size':'1 um'
            },
            'WaterIce':{
                'abn':'1e-3',
                'size':'5 um'
            }
        }
    }
    planet = pl.Planet.from_dict(d)

def test_planet_properties(default_planet:pl.Planet):
    planet = default_planet
    assert planet.flat.ndim == 1
    assert isinstance(planet.gcm_properties,str)

def test_planet_content(default_planet:pl.Planet):
    content = default_planet.content
    with open(OUTFILE,'wb') as file:
        file.write(content)


if __name__ in '__main__':
    test_planet_content()