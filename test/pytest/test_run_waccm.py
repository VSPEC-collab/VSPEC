
from pathlib import Path
from os import chdir
from astropy import units as u
import matplotlib.pyplot as plt

import VSPEC

chdir(Path(__file__).parent)
DEFAULT_CFG = Path(__file__).parent / 'default.cfg'
WACCM_CFG = Path(__file__).parent / 'waccm.cfg'
T0_GCM = Path(__file__).parent / 'test_gcms'/'waccmt0.gcm'


def run_static_waccm():
    model = VSPEC.ObservationModel(DEFAULT_CFG)
    model.params.gcm_path = T0_GCM
    model.params.planet_init_substellar_lon = 180*u.deg
    model.params.gcm_binning = 200
    model.build_planet()
    model.build_spectra()
    data = VSPEC.PhaseAnalyzer(model.directories['all_model'])
    plt.pcolormesh(data.time.value,data.wavelength.value,data.thermal.value)
    plt.colorbar()
    0

def run_nc_test():
    model = VSPEC.ObservationModel(WACCM_CFG)
    model.build_planet()
    model.build_spectra()
    data = VSPEC.PhaseAnalyzer(model.directories['all_model'])
    plt.pcolormesh(data.time.value,data.wavelength.value,data.thermal.value)
    plt.colorbar()
    0


if __name__ in '__main__':
    # run_static_waccm()
    run_nc_test()