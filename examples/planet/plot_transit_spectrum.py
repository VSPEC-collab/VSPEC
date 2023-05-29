"""
Plot the spectrum of a transiting planet
========================================

This example runs VSPEC with a transiting planet scenario.
"""

from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
# from os import chdir
# from pathlib import Path
# chdir(Path(__file__).parent)

from VSPEC import ObservationModel,PhaseAnalyzer

CFG_PATH = 'transit_spectrum.yaml'

# %%
# Initialize the VSPEC run
# ------------------------
#
# We read in the config file and run the model.

model = ObservationModel(CFG_PATH)
model.bin_spectra()
model.build_planet()
model.build_spectra()

# %%
# Load in the data
# ----------------
#
# We can use VSPEC to read in the synthetic
# data we just created.

data = PhaseAnalyzer(model.dirs['all_model'])

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