"""
Plot the lightcurve of a flaring star with spots
================================================

This example plots the lightcurve caused by a
flaring star when it also has spots.
"""

from astropy import units as u
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pypsg

from VSPEC import ObservationModel,PhaseAnalyzer
from VSPEC import params
from VSPEC import config

SEED = 42
pypsg.docker.set_url_and_run()

# %%
# Initialize the VSPEC run parameters
# -----------------------------------
#
# For this example, we will create the
# parameter objects explicitly. This can also
# be done using a YAML file.

header = params.Header(
    data_path=Path('.vspec/flare_spot_lightcurve'),
    teff_min=2900*u.K,
    teff_max=3400*u.K,
    seed=SEED,verbose=1
)

star = params.StarParameters(
    psg_star_template='M',
    teff=3300*u.K,
    mass = 0.1*u.M_sun,
    radius=0.15*u.R_sun,
    period = 6*u.day,
    misalignment_dir=0*u.deg,
    misalignment=0*u.deg,
    ld = params.LimbDarkeningParameters.solar(),
    faculae=params.FaculaParameters.none(),
    spots=params.SpotParameters(
        distribution='iso',
        initial_coverage=0.1,
        area_mean=300*config.MSH,
        area_logsigma=0.2,
        teff_umbra=2900*u.K,
        teff_penumbra=3000*u.K,
        equillibrium_coverage=0.1,
        burn_in=0*u.day,
        growth_rate=0/u.day,
        decay_rate=0*config.MSH/u.day,
        initial_area=10*config.MSH
        ),
    flares=params.FlareParameters(
        dist_teff_mean=9000*u.K,
        dist_teff_sigma=500*u.K,
        dist_fwhm_mean=3*u.hr,
        dist_fwhm_logsigma=0.4,
        alpha=-0.829,
        beta=26.87,
        min_energy=1e32*u.erg,
        cluster_size=3
    ),
    granulation=params.GranulationParameters.none(),
    grid_params=(500, 1000)
)

planet = params.PlanetParameters.std(init_phase=180*u.deg,init_substellar_lon=0*u.deg)
system = params.SystemParameters(
    distance=1.3*u.pc,
    inclination=80*u.deg,
    phase_of_periasteron=0*u.deg
)
observation = params.ObservationParameters(
    observation_time=10*u.day,
    integration_time=4*u.hr
)
psg_params = params.psgParameters(
    gcm_binning=200,
    phase_binning=1,
    use_molecular_signatures=True,
    nmax=0,
    lmax=0,
    continuum=['Rayleigh', 'Refraction', 'CIA_all'],
)
instrument = params.InstrumentParameters.mirecle()

gcm = params.gcmParameters(
    gcm=params.vspecGCM.earth(molecules={'CO2':1e-4}),
    mean_molec_weight=28
)

parameters = params.InternalParameters(
    header = header,
    star = star,
    planet = planet,
    system = system,
    obs=observation,
    psg = psg_params,
    inst=instrument,
    gcm = gcm
)

#%%
# Run the simulation
# ------------------
#

model = ObservationModel(params=parameters)
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
# Make the figure
# ---------------
#

time = (data.time - data.time[0]).to_value(u.day)
wl = data.wavelength.to_value(u.um)

emission = (data.thermal/data.total).to_value(u.dimensionless_unscaled)*1e6

variation = (data.star/data.star[:,0,np.newaxis]-1).to_value(u.dimensionless_unscaled)*100

fig,ax = plt.subplots(1,2,figsize=(8,4))

im=ax[0].pcolormesh(time,wl,emission,cmap='cividis')
fig.colorbar(im,ax=ax[0],label='Planet Thermal Emission (ppm)',location='top')
# ax[0].set_title('Planet')

im=ax[1].pcolormesh(time,wl,variation,cmap='cividis')
fig.colorbar(im,ax=ax[1],label='Stellar Variation (%)',location='top')
# ax[1].set_title('Star')

ax[0].set_ylabel('Wavelength (${\\rm \\mu m}$)')
fig.text(0.5,0.02,'Time (days)',ha='center')

