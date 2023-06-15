"""
Plot flare frequency function
=============================

This example generates flares and compares them to the expected distribution.
"""

from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt

from VSPEC.variable_star_model import FlareGenerator

SEED = 10

# %%
# Generate the flares
# -------------------

dt = 10000*u.day # a long time.
gen = FlareGenerator(
    dist_teff_mean=9000*u.K,
    dist_teff_sigma=1000*u.K,
    dist_fwhm_mean=3*u.hr,
    dist_fwhm_logsigma=0.3,
    min_energy=1e33*u.erg,
    cluster_size=4,
    rng=np.random.default_rng(seed=SEED)
)
Es = np.logspace(np.log10(gen.min_energy.to_value(u.erg)),np.log10(gen.min_energy.to_value(u.erg))+4)*u.erg

flares = gen.generate_flare_series(dt)

# %%
# Get the energies
# ----------------

energies = np.array([flare.energy.to_value(u.erg) for flare in flares])

energies_ge_E = np.array([np.sum(energies>=E) for E in Es.to_value(u.erg)])

measured_freq = energies_ge_E/dt
measured_freq_err = np.sqrt(energies_ge_E)/dt

# %%
# Plot the results
# ----------------

beta = gen.beta
alpha = gen.alpha

expected_log_freq = beta + alpha*np.log10(Es/u.erg)
expected_freq = 10**expected_log_freq / u.day

ratio = np.where(energies_ge_E>0,measured_freq/expected_freq,np.nan)
ratio_err = np.where(energies_ge_E>0,measured_freq_err/expected_freq,np.nan)

fig,axes = plt.subplots(2,1)

axes[0].plot(Es,expected_freq,c='xkcd:azure',label='Expected')
axes[0].errorbar(Es,measured_freq,yerr=measured_freq_err,fmt='o',color='xkcd:rose pink',label='Observed',markersize=5)
axes[0].set_xlabel('Energy (erg)')
axes[0].set_ylabel('Frequency (1/day)')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].legend()

axes[1].errorbar(Es,ratio,yerr=ratio_err,c='k',fmt='o',markersize=5)
axes[1].set_xlabel('Energy (erg)')
axes[1].set_ylabel('Observed/Expected')
axes[1].set_xscale('log')
axes[1].axhline(1,c='k',ls='--')
axes[1].set_xlim(axes[0].get_xlim())
0
# %%
# Look for clustering
# -------------------

tpeaks = np.array(
    [
        flare.tpeak.to_value(u.day) for flare in flares
    ]
)
tpeaks = np.sort(tpeaks)
tdiffs = np.diff(tpeaks)
plt.hist(tdiffs,bins=np.logspace(-3,3,30))
plt.xscale('log')
0