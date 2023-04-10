#
# TEST GEOMETRY
# --------------
# Objective 1
# Create a gif with some random parameters

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from astropy import units as u
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from os import system


from os import chdir, remove


from VSPEC import ObservationModel, PhaseAnalyzer
from VSPEC.helpers import isclose, to_float
from VSPEC import variable_star_model
from pathlib import Path


def make_plots(i, model):
    model.params.system_inclination = i*u.deg
    reffig, refax = plt.subplots(2, 1, sharex=True)
    thermfig, thermax = plt.subplots(2, 1, sharex=True)
    binning_list = [3,6,12,40]
    colorlist = [f'C{i}' for i in range(len(binning_list))]
    ref = None
    therm = None
    for b, c in zip(binning_list, colorlist):
        model.params.gcm_binning = b
        model.build_planet()
        model.build_spectra()

        data = PhaseAnalyzer(model.dirs['all_model'])

        nu = data.unique_phase - 360*u.deg

        observed = data.lightcurve('reflected', 0)
        refax[0].scatter(nu, observed, marker='o', label=f'binning={b}', c=c)
        if ref is None:
            ref = observed
        else:
            observed = (observed-ref)/ref * 100
            refax[1].scatter(nu, observed, marker='o', c=c)

        observed = data.lightcurve('thermal', 140)
        thermax[0].scatter(nu, observed, marker='o', label=f'binning={b}', c=c)
        if therm is None:
            therm = observed
        else:
            observed = (observed-therm)/therm * 100
            thermax[1].scatter(nu, observed, marker='o', c=c)

    refax[1].set_xlabel('phase (deg)')
    thermax[1].set_xlabel('phase (deg)')
    refax[0].set_ylabel(f'Reflected at {data.wavelength[0]:.1f} (W m-2 um-1)')
    thermax[0].set_ylabel(
        f'Thermal at {data.wavelength[140]:.1f} (W m-2 um-1)')
    refax[1].set_ylabel('Residual (%)')
    thermax[1].set_ylabel('Residual (%)')

    refax[0].legend()
    thermax[0].legend()
    reffig.savefig(
        f'ref_binningi{str(i).zfill(3)}.png', facecolor='w', dpi=120)
    thermfig.savefig(
        f'therm_binningi{str(i).zfill(3)}.png', facecolor='w', dpi=120)


if __name__ in '__main__':
    CONFIG_FILENAME = 'test_gcm_binning.cfg'
    WORKING_DIRECTORY = Path(__file__).parent
    CONFIG_PATH = WORKING_DIRECTORY / CONFIG_FILENAME

    chdir(WORKING_DIRECTORY)

    model = ObservationModel(CONFIG_FILENAME)

    model.bin_spectra()

    make_plots(0, model)
    make_plots(90, model)
