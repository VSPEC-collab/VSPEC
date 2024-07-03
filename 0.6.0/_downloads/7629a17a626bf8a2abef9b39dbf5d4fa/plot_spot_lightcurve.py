"""
Plot the lightcurve of a spotted star
=====================================

This example plots the lightcurve caused by a
spotted photosphere.
"""
from pathlib import Path
from astropy import units as u
import matplotlib.pyplot as plt
import pypsg

from VSPEC import ObservationModel,PhaseAnalyzer

try:
    CFG_PATH = Path(__file__).parent / 'spot_lightcurve.yaml'
except NameError:
    CFG_PATH = 'spot_lightcurve.yaml'

pypsg.docker.set_url_and_run()

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
# Get the lightcurve
# ------------------
#
# We will look in a few different wavelengths.

wl_pixels = [0,300,500,700]
time = data.time.to(u.day)
for i in wl_pixels:
    wl = data.wavelength[i]
    lc = data.lightcurve(
        source='star',
        pixel=i,
        normalize=0
    )
    plt.plot(time,lc,label=f'{wl:.1f}')
plt.legend()
plt.xlabel(f'time ({time.unit})')
_=plt.ylabel('Flux (normalized)')