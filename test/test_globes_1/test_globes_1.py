# GLOBES TEST 1
#
#
# Test the phase dependence of reflected light
##############################################

from pathlib import Path
import matplotlib.pyplot as plt
from astropy import units as u


from VSPEC import ObservationModel, PhaseAnalyzer

CONFIG_FILENAME = 'test_globes_1.cfg'
CONFIG_PATH = Path(__file__).parent / CONFIG_FILENAME

model = ObservationModel(CONFIG_PATH)
model.build_directories()
model.build_star()
model.warm_up(0*u.day,0*u.day)
model.bin_spectra()
model.build_planet()
model.build_spectra()

