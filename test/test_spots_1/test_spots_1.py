import VSPEC
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import pandas as pd

from pathlib import Path
from os import chdir

CONFIG_FILENAME = 'test_spots_1.cfg'

WORKING_DIRECTORY = Path(__file__).parent
CONFIG_PATH = WORKING_DIRECTORY / CONFIG_FILENAME

chdir(WORKING_DIRECTORY)

model = VSPEC.ObservationModel(CONFIG_PATH,debug=False)

model.build_directories()
model.build_star()
model.warm_up_star(60*u.day,0*u.day)
# model.bin_spectra()

model.build_planet()
model.build_spectra()


data = VSPEC.PhaseAnalyzer(model.dirs['all_model'])

fig,ax = plt.subplots(1,1)
fig_filename = 'lightcurve.png'
pixel = (0,70)
ax.plot(data.time*u.deg,data.lightcurve('star',pixel))
bandpass = data.wavelength[slice(*pixel)]
ax.set_xlabel('time (s)')
ax.set_ylabel('flux (W m-2 um-1)')
ax.set_title(f'reflected with\nbandpass from {bandpass.min()} to {bandpass.max()}')
fig.savefig(fig_filename,dpi=120,facecolor='w')