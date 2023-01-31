
import matplotlib.pyplot as plt
from astropy import units as u
import numpy as np
# import cartopy.crs as ccrs
from pathlib import Path

from os import chdir, remove


from VSPEC import ObservationModel, PhaseAnalyzer

CONFIG_FILENAME = 'test_geometry_1.cfg'
WORKING_DIRECTORY = Path(__file__).parent
CONFIG_PATH = WORKING_DIRECTORY / CONFIG_FILENAME

chdir(WORKING_DIRECTORY)

# model = ObservationModel(CONFIG_PATH,debug=False)

model = ObservationModel(CONFIG_FILENAME)

geo = model.get_observation_parameters()

plan = model.get_observation_plan(geo)

fig,ax = plt.subplots(1,1,figsize=(12,5))
# ax.plot(plan['time'],plan['phase'],label='phase')
ax.scatter(plan['phase'],plan['sub_stellar_lon'],label='sub_stellar_lon')
# ax.scatter(plan['phase'],plan['planet_sub_obs_lon'],label='sub_obs_lon')
# ax.scatter(plan['time'],plan['sub_stellar_lon'],label='sub_stellar_lon')
# ax.scatter(plan['time'],plan['sub_stellar_lon'] - plan['phase'],label='sub_obs_lon')
ax.set_xlabel('deg')
ax.legend()
fig.savefig('orbit.png',facecolor='w')


axes = []
phases = np.linspace(0,90,9,endpoint=True)
phases = [40]
for phase in phases:
    filename = f'geoplot_phase{phase}.png'
    fig = geo.plot(phase*u.deg)
    fig.savefig(f'geoplot_phase{phase}.png',facecolor='w')
nplots = len(phases)
fig,axes = plt.subplots(nplots,1,figsize = (10,5*nplots),tight_layout=True)
axes = np.ravel(axes)
for i in range(nplots):
    phase = phases[i]
    filename = f'geoplot_phase{phase}.png'
    dat = plt.imread(filename)
    axes[i].imshow(dat)
    axes[i].set_title(f'phase={phase}')
    axes[i].axis('off')
    remove(filename)
fig.savefig('geophases.png',facecolor='w')


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



