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


from VSPEC import ObservationModel
from VSPEC.helpers import isclose
from VSPEC import variable_star_model
from pathlib import Path



if __name__ in '__main__':
    CONFIG_FILENAME = 'test_star_1.cfg'
    WORKING_DIRECTORY = Path(__file__).parent
    CONFIG_PATH = WORKING_DIRECTORY / CONFIG_FILENAME

    chdir(WORKING_DIRECTORY)


    model = ObservationModel(CONFIG_FILENAME)


    model.build_star()
    # area = 0.5*4*np.pi*model.params.star_radius**2
    # model.star.add_spot([variable_star_model.StarSpot(
    #     0*u.deg,0*u.deg,
    #     area,area,
    #     2900*u.K,2500*u.K,
    #     growth_rate=0/u.day,decay_rate=0*area/u.day,

    # )])
    model.warm_up_star(0*u.day,0*u.day)

    geo = model.get_observation_parameters()
    plan = model.get_observation_plan(geo)
    
    make_gif(plan,'geometry.gif')
    # geo.plot(0*u.deg)
    # i=0
    # phase = plan['phase'][i]
    # lat = plan['sub_obs_lat'][i]
    # lon = plan['sub_obs_lon'][i]
    # coords = {
    #     'lat':lat,
    #     'lon':lon
    # }
    # make_fig(phase,coords, 'phase1.png')
