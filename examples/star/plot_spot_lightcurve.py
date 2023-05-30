"""
Plot the lightcurve of a spotted star
=====================================

This example plots the lightcurve caused by a
spotted photosphere.
"""

from astropy import units as u
import matplotlib.pyplot as plt

from VSPEC import ObservationModel,PhaseAnalyzer

CFG_PATH = 'spot_lightcurve.yaml'

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
plt.ylabel(f'Flux (normalized)')