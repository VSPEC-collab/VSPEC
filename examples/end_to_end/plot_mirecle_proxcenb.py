"""
Observe the MIRECLE Phase Curve of Proxima Centauri b
=====================================================

This example observes the closest exoplanet with the Mid-Infrared
Exoplanet CLimate Explorer.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import pypsg

from VSPEC import ObservationModel,PhaseAnalyzer
from VSPEC import params

SEED = 10
# pypsg.docker.set_url_and_run()
pypsg.settings.save_settings(url=pypsg.settings.PSG_URL)


# %%
# Create the needed configurations
# --------------------------------
# 
# MIRECLE is described in :cite:t:`2022AJ....164..176M`

# Instrument
inst = params.InstrumentParameters.mirecle()

# Observation

observation = params.ObservationParameters(
    observation_time=12*u.day,
    integration_time=8*u.hr
)

# PSG
psg_params = params.psgParameters(
    use_molecular_signatures=True,
    gcm_binning=200,
    phase_binning=1,
    use_continuum_stellar=True,
    nmax=0,
    lmax=0,
    continuum=['Rayleigh', 'Refraction','CIA_all'],
    )
# Star and Planet

star_teff = 2900*u.K
star_rad = 0.141*u.R_sun
inclination = 85*u.deg
planet_mass = 1.07*u.M_earth/np.sin(inclination)
planet_rad = 1*u.R_earth * planet_mass.to_value(u.M_earth)**3
orbit_rad = 0.04856*u.AU
orbit_period = 11.18*u.day
planet_rot_period = orbit_period
star_rot_period = 90*u.day
planet_mass = 1.07*u.M_earth
star_mass = 0.122*u.M_sun



initial_phase = 180*u.deg

planet_params = params.PlanetParameters(
    name='proxcenb',
    radius=planet_rad,
    gravity=params.GravityParameters('kg',planet_mass),
    semimajor_axis=orbit_rad,
    orbit_period=orbit_period,
    rotation_period=planet_rot_period,
    eccentricity=0,
    obliquity=0*u.deg,
    obliquity_direction=0*u.deg,
    init_phase=initial_phase,
    init_substellar_lon=0*u.deg
)

system_params = params.SystemParameters(
    distance=1.3*u.pc,
    inclination=inclination,
    phase_of_periasteron=0*u.deg
)


star_dict = {
    'teff': star_teff,
    'radius': star_rad
}
planet_dict = {'semimajor_axis': orbit_rad}

gcm_dict = {
    'nlayer': 30,
    'nlon': 30,
    'nlat': 15,
    'epsilon': 1.5,
    'albedo': 0.3,
    'emissivity': 1.0,
    'lat_redistribution': 0.5,
    'gamma': 1.4,
    'psurf': 1*u.bar,
    'ptop': 1e-8*u.bar,
    'wind': {'U': '0 m/s','V':'0 m/s'},
    'molecules':{'CO2':1e-4}
}

gcm_params = params.gcmParameters.from_dict({
    'star':star_dict,
    'planet':planet_dict,
    'gcm':{'vspec':gcm_dict,'mean_molec_weight':28}
})
quiet_star = params.StarParameters(
    spots=params.SpotParameters.none(),
    psg_star_template='M',
    teff=star_teff,
    mass=star_mass,
    radius=star_rad,
    period=star_rot_period,
    misalignment=0*u.deg,
    misalignment_dir=0*u.deg,
    ld=params.LimbDarkeningParameters.proxima(),
    faculae=params.FaculaParameters.none(),
    flares=params.FlareParameters.none(),
    granulation=params.GranulationParameters.none(),
    grid_params=(500, 1000),
    spectral_grid='default'
)

# Set parameters for simulation

internal_params = params.InternalParameters(
    header=params.Header(
        data_path=Path('.vspec/proxcenb'),
        teff_min=2300*u.K,teff_max=3400*u.K,
        seed = SEED),
    star = quiet_star,
    psg=psg_params,
    planet=planet_params,
    system=system_params,
    obs=observation,
    gcm=gcm_params,
    inst=inst
)

# %%
# Run VSPEC
# ---------
#
# We read in the config file and run the model.

model = ObservationModel(internal_params)
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
# Plot the phase curve
# --------------------
#

fig,ax = plt.subplots(1,1,figsize=(4,4),tight_layout=True)

emission = (data.thermal/data.total).to_value(u.dimensionless_unscaled)*1e6
noise = (data.noise/data.total).to_value(u.dimensionless_unscaled)*1e6
sim_noise = model.rng.normal(loc=0,scale=noise)
sim_data = emission + sim_noise

time = (data.time - data.time[0]).to_value(u.day)
wl = data.wavelength.to_value(u.um)

im = ax.pcolormesh(time,wl,sim_data,cmap='viridis')
fig.colorbar(im,ax=ax,label='Emission (ppm)')

ax.set_xlabel('Time (days)')
ax.set_ylabel('Wavelength ($\\mu m$)')

# %%
# Plot the integrated spectrum
# ----------------------------
#

true = np.mean(emission,axis=1)
observed = np.mean(sim_data,axis=1)
err = np.sqrt(np.sum(noise**2,axis=1))/noise.shape[1]

fig,ax = plt.subplots(1,1,figsize=(4,3),tight_layout=True)

ax.plot(wl,true,c='xkcd:azure',label='True')
ax.errorbar(wl,observed,yerr=err,fmt='o',color='xkcd:rose pink',label='Observed',markersize=2)
ax.set_xlabel('Wavelength ($\\mu m$)')
ax.set_ylabel('Planetary Emission (ppm)')
ax.legend()
