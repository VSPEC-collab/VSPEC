import VSPEC
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import pandas as pd
from io import StringIO
import xarray
from pathlib import Path
from os import chdir

CONFIG_FILENAME = 'test_globes_2.cfg'

WORKING_DIRECTORY = Path(__file__).parent
CONFIG_PATH = WORKING_DIRECTORY / CONFIG_FILENAME

chdir(WORKING_DIRECTORY)

model = VSPEC.ObservationModel(CONFIG_PATH,debug=False)
# model.build_directories()
# model.build_star()
# model.warm_up_star(0*u.day,0*u.day)
# model.bin_spectra()

# model.build_planet()

# model.build_spectra()

data = VSPEC.PhaseAnalyzer(model.dirs['all_model'])

print(data)