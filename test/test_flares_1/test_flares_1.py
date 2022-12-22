from VSPEC.variable_star_model import FlareGenerator, FlareCollection, StellarFlare
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from pathlib import Path
from os import chdir


WORKING_DIRECTORY = Path(__file__).parent

chdir(WORKING_DIRECTORY)

star_teff = 3300*u.K
star_rot = 80*u.day
flaregen = FlareGenerator(star_teff,star_rot)
E_dist = flaregen.generage_E_dist()
time = 3*u.day
flares = flaregen.generate_flare_series(E_dist,time)
flares = FlareCollection(flares)
tstart = 1*u.day
tfinish = 2*u.day
fluxes = flares.get_flare_integral_in_timeperiod(tstart,tfinish)
print(fluxes)