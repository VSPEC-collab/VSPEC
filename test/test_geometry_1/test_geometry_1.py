#
# TEST GEOMETRY
# --------------
# Objective 1
# Test the sub-stellar longitude as a function of phase for various rotation periods
# Expectation:
# lon = lon0 + phase - rotation
# --------------
# Objective 2
# Test the substellar latitude as a function of phase for various obliquities
# Expectation:
# lat = 0 for zero obliquity
# --------------
# Objective 3
# Test the planet sub-observer lon as a function of phase for various rotation periods
# Expectation:
# lon0 = init_substellar_lon - initial_phase
# lon = lon0 - rotation
# --------------
# Objective 4
# Test the planet sub-observer lat as a function of phase for various inclinations
# Expectation:
# lat = -i (i=0 for edge on)
# --------------
# Objective 5
# Test the stellar sub-obs lon as func of time for various rotation periods
# Expectation:
# lon = 0 - rotation
# --------------

import matplotlib.pyplot as plt
from astropy import units as u
import numpy as np
from pathlib import Path

from os import chdir, remove


from VSPEC import ObservationModel
from VSPEC.helpers import isclose



def test_obj1():
    """OBJECTIVE 1"""
    fig,ax = plt.subplots(2,1,figsize=(10,5))
    original_rot_period = model.params.planet_rotational_period
    orbit_period = model.params.planet_orbital_period
    lon0 = model.params.planet_init_substellar_lon
    i=0
    for period in np.array([0.1,0.5,1,2,1e6,-1])*orbit_period:
        model.params.planet_rotational_period = period
        geo = model.get_observation_parameters()
        plan = model.get_observation_plan(geo)
        ax[0].scatter(plan['phase'],plan['sub_stellar_lon'],c=f'C{i}',label=f'period={period.value:.1f} {period.unit}')

        N=model.params.total_images
        phase = plan['phase']
        time = np.linspace(0,1,N)*orbit_period
        expected_lon = (lon0 + (phase-phase[0]) - 360*u.deg*time/period) % (360*u.deg)
        ax[0].scatter(phase,expected_lon,c=f'C{i}',label=f'period={period.value:.1f} {period.unit}',marker='s',alpha=0.5)
        ax[0].axvline(phase[0].value,c='k',ls='--')
        res = plan['sub_stellar_lon'] - expected_lon
        ax[1].scatter(phase,res,c=f'C{i}')
        ax[1].axhline(1e-6,c='k',ls='--')
        ax[1].axhline(-1e-6,c='k',ls='--')
        assert np.all(isclose(plan['sub_stellar_lon'],expected_lon,0.1*u.deg)), f'sub-stellar lon test failed for {period}'

        i+=1
    model.params.planet_rotational_period = original_rot_period


    print('OBJECTIVE 1: Passed sub-stellar longitude test')
    ax[1].set_xlabel('phase (deg)')
    ax[0].set_ylabel('sub-stellar longitude (deg)')
    ax[1].set_ylabel('sub-residual (deg)')
    ax[0].set_title(f'orbital period {orbit_period}')
    ax[0].legend()
    fig.savefig('substellar_lon.png',facecolor='w')

def test_obj2():
    """OBJECTIVE 2"""
    original_obliquity = model.params.planet_obliquity
    for obl in [0]*u.deg:
        model.params.planet_obliquity = obl
        geo = model.get_observation_parameters()
        plan = model.get_observation_plan(geo)
        expected = plan['phase']*0
        assert np.all(isclose(plan['sub_stellar_lat'],expected,0.1*u.deg))
    model.params.planet_obliquity = original_obliquity
    print('OBJECTIVE 2: Passed sub-stellar latitude test')

def test_obj3():
    """OBJECTIVE 3"""
    fig,ax = plt.subplots(2,1,figsize=(12,5))
    original_rot_period = model.params.planet_rotational_period
    orbit_period = model.params.planet_orbital_period
    lon0 = model.params.planet_init_substellar_lon - model.params.planet_initial_phase
    i=0
    for period in np.array([0.1,0.5,1,2,1e6,-1])*orbit_period:
        model.params.planet_rotational_period = period
        geo = model.get_observation_parameters()
        plan = model.get_observation_plan(geo)
        phase = plan['phase']
        ax[0].scatter(plan['phase'],plan['planet_sub_obs_lon']  % (360*u.deg),c=f'C{i}',label=f'period={period.value:.1f} {period.unit}')
        ax[0].axvline(phase[0].value,c='k',ls='--')
        N=model.params.total_images
        time = np.linspace(0,1,N)*orbit_period
        expected_lon = (lon0 - 360*u.deg*time/period) % (360*u.deg)
        res = plan['planet_sub_obs_lon'] - expected_lon
        ax[1].scatter(phase,res,c=f'C{i}')
        ax[1].axhline(1e-6,c='k',ls='--')
        ax[1].axhline(-1e-6,c='k',ls='--')
        ax[0].scatter(phase,expected_lon,c=f'C{i}',label=f'period={period.value:.1f} {period.unit}',marker='s',alpha=0.5)
        assert np.all(isclose(plan['planet_sub_obs_lon'] % (360*u.deg),expected_lon,0.1*u.deg)), f'Planet sub-obs lon test failed for {period}'
        i+=1
    model.params.planet_rotational_period = original_rot_period
    print('OBJECTIVE 3: Passed planet sub-observer longitude test')

    ax[1].set_xlabel('phase (deg)')
    ax[1].set_ylabel('residual (deg)')

    ax[0].set_ylabel('sub-obs longitude (deg)')
    ax[0].set_title(f'orbital period {orbit_period}')
    ax[0].legend()
    fig.savefig('subobs_lon.png',facecolor='w')


def test_obj4():
    """OBJECTIVE 4"""
    original_i = model.params.system_inclination
    for i in [90,45,0,-45,-90]*u.deg:
        model.params.system_inclination = i
        model.params.system_inclination_psg = 90*u.deg-i
        geo = model.get_observation_parameters()
        plan = model.get_observation_plan(geo)
        expected = plan['phase']*0 - i
        assert np.all(isclose(plan['planet_sub_obs_lat'],expected,0.1*u.deg)), f'Planet Sub-obs lat test failed for i={i.value:.1f} {i.unit}'
    model.params.system_inclination = original_i
    model.params.system_inclination_psg = 90*u.deg - original_i
    print('OBJECTIVE 4: Passed planet sub-observer latitude test')

def test_obj5():
    """OBJECTIVE 5"""
    original_rot_per = model.params.star_rot_period
    for period in [1,5,10,20,40]*u.day:
        model.params.star_rot_period = period
        geo = model.get_observation_parameters()
        plan = model.get_observation_plan(geo)
        time = plan['time']
        expected_lon = 0*u.deg - (time/period)*360*u.deg
        message = f'Stellar sub-obs lon test failed for period = {period.value:.1f} {period.unit}'
        assert np.all(isclose(plan['sub_obs_lon'],expected_lon,0.1*u.deg)), message
    model.params.star_rot_period = original_rot_per
    print('OBJECTIVE 5: Passed stellar sub-observer longitude test')
    



# axes = []
# phases = np.linspace(0,90,9,endpoint=True)
# phases = [40]
# for phase in phases:
#     filename = f'geoplot_phase{phase}.png'
#     fig = geo.plot(phase*u.deg)
#     fig.savefig(f'geoplot_phase{phase}.png',facecolor='w')
# nplots = len(phases)
# fig,axes = plt.subplots(nplots,1,figsize = (10,5*nplots),tight_layout=True)
# axes = np.ravel(axes)
# for i in range(nplots):
#     phase = phases[i]
#     filename = f'geoplot_phase{phase}.png'
#     dat = plt.imread(filename)
#     axes[i].imshow(dat)
#     axes[i].set_title(f'phase={phase}')
#     axes[i].axis('off')
#     remove(filename)
# fig.savefig('geophases.png',facecolor='w')


# axes = []
# obl = [0,10,30,60,90]
# for ob in obl:
#     filename = f'geoplot_phase{phase}.png'
#     geo.obliquity = ob*u.deg
#     fig = geo.plot(0*u.deg)
#     fig.savefig(f'geoplot_obl{ob}.png',facecolor='w')
# nplots = len(obl)
# fig,axes = plt.subplots(nplots,1,figsize = (10,5*nplots))
# axes = np.ravel(axes)
# for i in range(nplots):
#     ob = obl[i]
#     filename = f'geoplot_obl{ob}.png'
#     dat = plt.imread(filename)
#     axes[i].imshow(dat)
#     axes[i].set_title(f'obl={ob}')
#     axes[i].axis('off')
#     remove(filename)
# fig.savefig('geoobl.png',facecolor='w')



if __name__ in '__main__':
    CONFIG_FILENAME = 'test_geometry_1.cfg'
    WORKING_DIRECTORY = Path(__file__).parent
    CONFIG_PATH = WORKING_DIRECTORY / CONFIG_FILENAME

    chdir(WORKING_DIRECTORY)


    model = ObservationModel(CONFIG_FILENAME)

    ### This should all work regardless of initial conditions
    ### The following code will assign random conditions for this run
    model.params.planet_init_substellar_lon = np.random.random() * 360*u.deg
    model.params.planet_initial_phase = np.random.random() * 360*u.deg

    print(f'Initial conditions:')
    print(f'lon = {model.params.planet_init_substellar_lon.value:.1f} {model.params.planet_init_substellar_lon.unit}')
    print(f'phase = {model.params.planet_initial_phase.value:.1f} {model.params.planet_initial_phase.unit}')
    test_obj1()
    test_obj2()
    test_obj3()
    test_obj4()
    test_obj5()