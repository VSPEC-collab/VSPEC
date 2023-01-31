from VSPEC.variable_star_model import FlareGenerator, FlareCollection, StellarFlare
import VSPEC
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from pathlib import Path
from os import chdir

# import warnings
# warnings.filterwarnings("error")

CONFIG_PATH = 'test_flares_1.cfg'
WORKING_DIRECTORY = Path(__file__).parent

chdir(WORKING_DIRECTORY)


# flare = StellarFlare(3*u.hr,1e34*u.erg,0*u.deg,0*u.deg,9000*u.K,1*u.day)
# time = np.linspace(0,6,100)*u.day
# print(flare.calc_peak_area())
# # print(flare.areacurve(time))
# print(flare.get_timearea(time))

model = VSPEC.ObservationModel(CONFIG_PATH,debug=False)

# model.build_directories()
# model.build_star()
# model.warm_up_star(30*u.day,0*u.day)
# model.bin_spectra()

model.build_planet()
model.build_spectra()

# data = VSPEC.PhaseAnalyzer(model.dirs['all_model'])

# fig,ax = plt.subplots(1,1)
# fig_filename = 'lightcurve.png'
# pixel = (0,70)
# pixel2 = (100,140)
# ax.plot(data.time*u.deg,data.lightcurve('star',pixel),label='Star, NearIR',c='C0')
# ax.plot(data.time*u.deg,data.lightcurve('total',pixel),label='Total, NearIR',c='C1')
# # ax.plot(data.time*u.deg,data.lightcurve('reflected',pixel),label='Reflected, NearIR',c='C2')
# # ax.plot(data.time*u.deg,data.lightcurve('star',pixel2),label='Star, MidIR',c='C0',ls='--')
# # ax.plot(data.time*u.deg,data.lightcurve('total',pixel2),label='Total, MidIR',c='C1',ls='--')
# # ax.plot(data.time*u.deg,data.lightcurve('reflected',pixel2),label='Reflected, MidIR',c='C2',ls='--')
# ax.set_yscale('log')
# ax.legend()

# bandpass = data.wavelength[slice(*pixel)]
# ax.set_xlabel('time (s)')
# ax.set_ylabel('flux (W m-2 um-1)')
# ax.set_title(f'reflected with\nbandpass from {bandpass.min()} to {bandpass.max()}')
# fig.savefig(fig_filename,dpi=120,facecolor='w')

# fig,ax = plt.subplots(1,1)
# fig_filename = 'phase_contour.png'
# cf = ax.contourf(data.unique_phase,data.wavelength,data.star,levels=60)
# ax.set_xlabel(f'phase ({data.unique_phase.unit})')
# ax.set_ylabel(f'wavelength ({data.wavelength.unit})')
# ax.set_title('Star light')
# cbar = fig.colorbar(cf)
# cbar.ax.set_label(str(data.star.unit))
# fig.savefig(fig_filename,dpi=120,facecolor='w')