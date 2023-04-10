# GLOBES TEST 1
#
#
# Test the phase dependence of reflected light
##############################################

from pathlib import Path
import matplotlib.pyplot as plt
from astropy import units as u
from os import chdir
import numpy as np


from VSPEC import ObservationModel, PhaseAnalyzer

CONFIG_FILENAME = 'test_globes_1.cfg'
WORKING_DIRECTORY = Path(__file__).parent
CONFIG_PATH = WORKING_DIRECTORY / CONFIG_FILENAME

chdir(WORKING_DIRECTORY)

model = ObservationModel(CONFIG_PATH)
model.build_directories()
model.bin_spectra()
# model.build_star()
fig,ax = plt.subplots(2,1)
fig_filename = 'lightcurve_debug.png'
pixel = (135,145)
# pixel = (0,30)

# for i in np.linspace(-90,90,2):
j=0
for a in [3,12,45]:
    # i_psg = 90 - i
    # model.params.system_inclination = i*u.deg
    # model.params.system_inclination_psg = i_psg*u.deg
    model.params.gcm_binning = a
    model.build_planet()
    model.build_spectra()
    data_path = model.dirs['all_model']
    data = PhaseAnalyzer(data_path)
    # make lightcurve
    color = f'C{j}'
    j+=1
    ax[0].plot(data.unique_phase-360*u.deg,data.lightcurve('reflected',pixel),label=f'{a}, ref',zorder=-1*a,c=color)
    ax[1].plot(data.unique_phase-360*u.deg,data.lightcurve('thermal',pixel),label=f'{a}, therm',zorder=-1*a,c=color,ls='--')

bandpass = data.wavelength[slice(*pixel)]

# model.params.gcm_path = '../test_gcms/proxcen_1d.cfg'
# model.params.use_globes = False
# model.build_planet()
# model.build_spectra()
# data_path = model.dirs['all_model']
# data = PhaseAnalyzer(data_path)
# # make lightcurve
# ax.plot(data.unique_phase-360*u.deg,data.lightcurve('reflected',pixel),label=f'1D PSG model')


for a in ax:
    a.set_xlabel('phase (deg)')
    a.set_ylabel('flux (W m-2 um-1)')
    a.set_xticks(np.linspace(-180,180,9))
    # a.axvline(-90,c='k',ls='--',lw=1)
    # a.axvline(0,c='k',ls='--',lw=1)
    # a.axvline(90,c='k',ls='--',lw=1)
    a.legend()
ax[0].set_title(f'bandpass from {bandpass.min()} to {bandpass.max()}')

fig.savefig(fig_filename,dpi=120,facecolor='w')



# make 2d phase plot

# fig,ax = plt.subplots(1,1)
# fig_filename = 'phase_contour.png'
# cf = ax.contourf(data.unique_phase,data.wavelength,data.reflected,levels=60)
# ax.set_xlabel(f'phase ({data.unique_phase.unit})')
# ax.set_ylabel(f'wavelength ({data.wavelength.unit})')
# ax.set_title('Reflected light')
# cbar = fig.colorbar(cf)
# cbar.ax.set_label(str(data.reflected.unit))
# fig.savefig(fig_filename,dpi=120,facecolor='w')
