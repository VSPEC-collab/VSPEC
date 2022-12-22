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

model = ObservationModel(CONFIG_PATH,debug=False)
model.build_directories()
model.build_star()
model.warm_up_star(0*u.day,0*u.day)
# model.bin_spectra()
fig,ax = plt.subplots(1,1)
fig_filename = 'lightcurve.png'
pixel = (0,70)

# for i in np.linspace(-90,90,2):
for a in [6,7,8,9]:
    # i_psg = 90 - i
    # model.params.system_inclination = i*u.deg
    # model.params.system_inclination_psg = i_psg*u.deg
    model.params.gcm_binning = a
    model.build_planet()
    model.build_spectra()
    data_path = model.dirs['all_model']
    data = PhaseAnalyzer(data_path)
    # make lightcurve
    ax.plot(data.unique_phase-360*u.deg,data.lightcurve('reflected',pixel),label=f'binning={a}')

bandpass = data.wavelength[slice(*pixel)]

# model.params.gcm_path = '../test_gcms/proxcen_1d.cfg'
# model.params.use_globes = False
# model.build_planet()
# model.build_spectra()
# data_path = model.dirs['all_model']
# data = PhaseAnalyzer(data_path)
# # make lightcurve
# ax.plot(data.unique_phase-360*u.deg,data.lightcurve('reflected',pixel),label=f'1D PSG model')



ax.set_xlabel('phase (deg)')
ax.set_ylabel('flux (W m-2 um-1)')
ax.set_title(f'reflected with\nbandpass from {bandpass.min()} to {bandpass.max()}')
ax.axvline(-90,c='k',ls='--',lw=1)
ax.axvline(0,c='k',ls='--',lw=1)
ax.axvline(90,c='k',ls='--',lw=1)
ax.legend()
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
