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
#########################################

import matplotlib.pyplot as plt
from astropy import units as u
import numpy as np
from pathlib import Path

from os import chdir, remove


from VSPEC import ObservationModel
from VSPEC.helpers import isclose

CONFIG_FILENAME = 'test_geometry_1.cfg'
WORKING_DIRECTORY = Path(__file__).parent
CONFIG_PATH = WORKING_DIRECTORY / CONFIG_FILENAME

chdir(WORKING_DIRECTORY)


model = ObservationModel(CONFIG_FILENAME)

# OBJECTIVE 1
fig,ax = plt.subplots(1,1,figsize=(12,5))
original_rot_period = model.params.planet_rotational_period
orbit_period = model.params.planet_orbital_period
lon0 = model.params.planet_init_substellar_lon
i=0
for period in np.array([0.1,0.5,1,2,1e6,-1])*orbit_period:
    model.params.planet_rotational_period = period
    geo = model.get_observation_parameters()
    plan = model.get_observation_plan(geo)
    ax.scatter(plan['phase'],plan['sub_stellar_lon'],c=f'C{i}',label=f'period={period.value:.1f} {period.unit}')

    N=model.params.total_images
    phase = plan['phase']
    time = np.linspace(0,1,N)*orbit_period
    expected_lon = (lon0 + phase - 360*u.deg*time/period) % (360*u.deg)
    assert np.all(isclose(plan['sub_stellar_lon'],expected_lon,0.1*u.deg)), f'sub-stellar lon test failed for {period}'
    ax.plot(phase,expected_lon,c=f'C{i}',label=f'period={period.value:.1f} {period.unit}')
    i+=1
model.params.planet_rotational_period = original_rot_period

ax.set_xlabel('phase (deg)')
ax.set_ylabel('sub-stellar longitude (deg)')
ax.set_title(f'orbital period {orbit_period}')
ax.legend()
fig.savefig('substellar_lon.png',facecolor='w')

# OBJECTIVE 2
original_obliquity = model.params.planet_obliquity
for obl in [0]*u.deg:
    model.params.planet_obliquity = obl
    geo = model.get_observation_parameters()
    plan = model.get_observation_plan(geo)
    expected = plan['phase']*0
    assert np.all(isclose(plan['sub_stellar_lat'],expected,0.1*u.deg))
model.params.planet_obliquity = original_obliquity

# OBJECTIVE 3
fig,ax = plt.subplots(1,1,figsize=(12,5))
original_rot_period = model.params.planet_rotational_period
orbit_period = model.params.planet_orbital_period
lon0 = model.params.planet_init_substellar_lon - model.params.planet_initial_phase
i=0
for period in np.array([0.1,0.5,1,2,1e6,-1])*orbit_period:
    model.params.planet_rotational_period = period
    geo = model.get_observation_parameters()
    plan = model.get_observation_plan(geo)
    ax.scatter(plan['phase'],plan['planet_sub_obs_lon']  % (360*u.deg),c=f'C{i}',label=f'period={period.value:.1f} {period.unit}')

    N=model.params.total_images
    phase = plan['phase']
    time = np.linspace(0,1,N)*orbit_period
    expected_lon = (lon0 - 360*u.deg*time/period) % (360*u.deg)
    assert np.all(isclose(plan['planet_sub_obs_lon'] % (360*u.deg),expected_lon,0.1*u.deg)), f'sub-obs lon test failed for {period}'
    ax.plot(phase,expected_lon,c=f'C{i}',label=f'period={period.value:.1f} {period.unit}')
    i+=1
model.params.planet_rotational_period = original_rot_period

ax.set_xlabel('phase (deg)')
ax.set_ylabel('sub-obs longitude (deg)')
ax.set_title(f'orbital period {orbit_period}')
ax.legend()
fig.savefig('subobs_lon.png',facecolor='w')

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



