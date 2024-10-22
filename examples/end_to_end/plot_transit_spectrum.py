"""
Plot the spectrum of a transiting planet
========================================

This example runs VSPEC with a transiting planet scenario.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import libpypsg

from VSPEC import ObservationModel,PhaseAnalyzer

try:
    CFG_PATH = Path(__file__).parent / 'transit_spectrum.yaml'
except NameError:
    CFG_PATH = Path('transit_spectrum.yaml')

libpypsg.docker.set_url_and_run()

# %%
# Initialize the VSPEC run
# ------------------------
#
# We read in the config file and run the model.

model = ObservationModel.from_yaml(CFG_PATH)
model.build_planet()
model.build_spectra()

# %%
# Load in the data
# ----------------
#
# We can use VSPEC to read in the synthetic
# data we just created.

data = PhaseAnalyzer(model.directories['all_model'])

# %%
# Calculate the transit depth
# ---------------------------
#
# Since this star model has no limb darkeing, no spots,
# and no noise, we don't need to fit a model to our data
# to extract the transit depth.

cmb_data = data.total
continuum = cmb_data[:,0] # the first epoch
data_min = np.min(cmb_data,axis=1)
transit_depth = (continuum-data_min)/continuum
rp_rs = np.sqrt(transit_depth)
wavelength = data.wavelength

plt.plot(wavelength,rp_rs*100)
plt.xlabel(f'Wavelength {wavelength.unit}')
plt.ylabel(r'$\frac{R_p}{R_*}$ (%)')