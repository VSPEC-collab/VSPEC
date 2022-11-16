# GLOBES TEST 1
#
#
# Test the phase dependence of reflected light
##############################################

from pathlib import Path
import matplotlib.pyplot as plt
from astropy import units as u
from os import chdir


from VSPEC import ObservationModel, PhaseAnalyzer

CONFIG_FILENAME = 'test_globes_1.cfg'
WORKING_DIRECTORY = Path(__file__).parent
CONFIG_PATH = WORKING_DIRECTORY / CONFIG_FILENAME

chdir(WORKING_DIRECTORY)

model = ObservationModel(CONFIG_PATH,debug=True)
model.build_directories()
model.build_star()
model.warm_up_star(0*u.day,0*u.day)
# model.bin_spectra()
model.build_planet()
model.build_spectra()

data_path = model.dirs['all_model']
data = PhaseAnalyzer(data_path)

# make 2d phase plot

fig,ax = plt.subplots(1,1)
fig_filename = 'phase_contour.png'
cf = ax.contourf(data.unique_phase,data.wavelength,data.reflected,levels=60)
ax.set_xlabel(f'phase ({data.unique_phase.unit})')
ax.set_ylabel(f'wavelength ({data.wavelength.unit})')
ax.set_title('Reflected light')
cbar = fig.colorbar(cf)
cbar.ax.set_label(str(data.reflected.unit))
fig.savefig(fig_filename,dpi=120,facecolor='w')

# make lightcurve

fig,ax = plt.subplots(1,1)
fig_filename = 'lightcurve.png'
pixel = (0,70)
bandpass = data.wavelength[slice(*pixel)]
ax.scatter(data.time,data.lightcurve('reflected',pixel))
ax.set_xlabel('time since periasteron (s)')
ax.set_ylabel('flux (W m-2 um-1)')
ax.set_title(f'reflected with\nbandpass from {bandpass.min()} to {bandpass.max()}')
ax.set_ylim(-0.1e-18,3e-18)
fig.savefig(fig_filename,dpi=120,facecolor='w')



