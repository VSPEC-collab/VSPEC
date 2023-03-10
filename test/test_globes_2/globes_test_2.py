import VSPEC
from pathlib import Path
from os import chdir

CONFIG_FILENAME = 'test_globes_2.cfg'

WORKING_DIRECTORY = Path(__file__).parent
CONFIG_PATH = WORKING_DIRECTORY / CONFIG_FILENAME

chdir(WORKING_DIRECTORY)

model = VSPEC.ObservationModel(CONFIG_PATH,debug=False)
model.bin_spectra()

model.build_planet()

model.build_spectra()

# data = VSPEC.PhaseAnalyzer(model.dirs['all_model'])

# print(data)