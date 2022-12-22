
import matplotlib.pyplot as plt
from astropy import units as u
import numpy as np
import cartopy.crs as ccrs
from pathlib import Path

from os import chdir


from VSPEC import ObservationModel, PhaseAnalyzer

CONFIG_FILENAME = 'test_geometry_1.cfg'
WORKING_DIRECTORY = Path(__file__).parent
CONFIG_PATH = WORKING_DIRECTORY / CONFIG_FILENAME

chdir(WORKING_DIRECTORY)

model = ObservationModel(CONFIG_PATH,debug=False)

model = ObservationModel(CONFIG_FILENAME)

geo = model.get_observation_parameters()

geo.plot(120*u.deg).show()

